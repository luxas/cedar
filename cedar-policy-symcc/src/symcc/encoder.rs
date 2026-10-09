/*
 * Copyright Cedar Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! This module defines the Cedar encoder, which translates a list of boolean Terms
//! into a list of SMT assertions. Term encoding is trusted.
//!
//!  We use the following type representations for primitive types:
//!  * `TermType.bool`:     builtin SMT `Bool` type
//!  * `TermType.string`:   builtin SMT `String` type
//!  * `TermType.bitvec n`: builtin SMT `(_ BitVec n)` type
//!
//!  We will represent non-primitive types as SMT algebraic data types:
//!  * `TermType.option T`: a parameterized SMT algebraic datatype of the same name,
//!    and with the constructors `(some (val T))` and `(none)`. For each constructor
//!    argument, SMTLib introduces a corresponding (total) selector function. We
//!    will translate `Term.some` nodes in the Term language as applications of the
//!    `val` selector function.
//!  * `TermType.entity E`: we represent Cedar entities of entity type E as values
//!    of the SMT algebraic datatype E with a single constructor, `(E (E_eid String))`.
//!    The selector is named `E_eid`, after the entity type, since SMT-LIB requires
//!    unique selector and constructor names across all datatypes.
//!    Each entity type E gets an uninterpreted function `f: E → Record_E` that maps
//!    instances of E to their attributes.  Similarly, each E
//!    gets N uninterpreted functions `g₁: E → Set E₁, ..., gₙ: E → Set Eₙ` that map
//!    each instance of E to its ancestor sets of the given types, as specified by
//!    the `memberOf` relation in the schema.
//!  * `TermType.Record (Map Attr TermType)`: we represent each record term type as
//!    an SMT algebraic datatype with a single constructor. The order of arguments
//!    to the constructor (the attributes) is important, so we fix that to be the
//!    lexicographic order on the attribute names of the underlying record type. We
//!    use the argument selector functions to translate `record.get` applications.
//!    We can't use raw Cedar attribute names for argument names because they may
//!    not be valid SMT identifiers. So, we'll keep a mapping from the attribute
//!    names to their unique SMT ids. In general, we'll name SMT record types as
//!    "R`<i>`" where `<i>` is a natural number and attributes within the record as
//!    "R`<i>`a`<j>`", where `<j>` is the attribute's position in the constructor argument
//!    list.
//!
//!  Similarly to types and attributes, all uninterpreted functions, variables, and
//!  Terms are mapped to their SMT encoding that conforms to the SMTLib syntax. We
//!  keep track of these mappings to ensure that each Term construct is translated
//!  to its SMT encoding exactly once.  This translation invariant is necessary for
//!  correctness in the case of record type declarations, UF names, and variable
//!  names; and it is necessary for compactness in the case of terms. In
//!  particular, the resulting SMT encoding will be in A-normal form (ANF): the body
//!  of every s-expression in the encoding consists of atomic subterms (identifiers
//!  or literals).

use async_recursion::async_recursion;
use itertools::Itertools;
use miette::Diagnostic;
use smol_str::{format_smolstr, SmolStr, ToSmolStr};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use thiserror::Error;

use cedar_policy_core::ast::PatternElem;

use super::{
    bitvec::{BitVec, BitVecError},
    env::SymEnv,
    ext::Ext,
    extension_types::ipaddr::{CIDRv4, CIDRv6, IPNet, IPv4Prefix, IPv6Prefix},
    op::{ExtOp, Op, Uuf},
    smtlib_script::SmtLibScript,
    term::{Term, TermPrim, TermVar},
    term_type::TermType,
    type_abbrevs::*,
};

use super::extension_types::ipaddr::{V4_WIDTH, V6_WIDTH};

/// Errors during encoding, i.e., converting [`Term`]
/// to SMT-LIB 2 format.
#[derive(Debug, Diagnostic, Error)]
pub enum EncodeError {
    /// IO error.
    #[error("IO error during SMT encoding")]
    Io(#[from] std::io::Error),
    /// Missing member in enum entity.
    #[error("missing member {0} in enum entity")]
    EnumMissingMember(EntityUID),
    /// Record missing attribute.
    #[error("record missing attribute {0}")]
    RecordMissingAttr(Attr),
    /// Expecting a record type.
    #[error("expecting a record type, got {0:?}")]
    ExpectRecord(TermType),
    /// `set.all` (feature `anyall`) term reached the encoder, which does not yet
    /// emit the two-`set.filter` `HO_ALL` encoding. The Rust `.all` compile arm
    /// (Phase 6 W3) builds the term; the byte-for-byte encoder emission (and the
    /// `HO_ALL` logic switch) land with the SymCC differential (Phase 6 W8),
    /// gated on Phase 5B re-landing and D-74.
    #[cfg(feature = "anyall")]
    #[error("SMT encoding of `set.all` is not yet wired (Phase 6 W8)")]
    SetAllNotEncoded,
    /// Missing type encoding.
    #[error("missing type encoding for {0:?}")]
    MissingTypeEncoding(TermType),
    /// Malformed record get.
    #[error("malformed record get")]
    MalformedRecordGet,
    /// Unable to encode string.
    #[error("unable to encode string \"{0}\" in SMT as it exceeds the max supported code point")]
    EncodeStringFailed(SmolStr),
    /// Unable to encode pattern.
    #[error("unable to encode pattern {0:?} in SMT as it exceeds the max supported code point")]
    EncodePatternFailed(OrdPattern),
    /// Bit-vector error.
    #[error("bit-vector error")]
    BitVecError(#[from] BitVecError),
}

type Result<T> = std::result::Result<T, EncodeError>;

#[derive(Debug)]
pub struct Encoder<'a, S> {
    pub(super) terms: BTreeMap<Term, SmolStr>,
    pub(super) types: BTreeMap<TermType, SmolStr>,
    pub(super) uufs: BTreeMap<Uuf, SmolStr>,
    pub(super) enums: BTreeMap<&'a EntityType, &'a BTreeSet<SmolStr>>,
    script: S,
}

fn term_id(n: usize) -> SmolStr {
    format_smolstr!("t{n}")
}

fn uuf_id(n: usize) -> SmolStr {
    format_smolstr!("f{n}")
}

fn entity_type_id(n: usize) -> SmolStr {
    format_smolstr!("E{n}")
}

pub(super) fn enum_id(e: &str, n: usize) -> SmolStr {
    format_smolstr!("{e}_m{n}")
}

fn record_type_id(n: usize) -> SmolStr {
    format_smolstr!("R{n}")
}

fn record_attr_id(r: &str, n: usize) -> SmolStr {
    format_smolstr!("{r}_a{n}")
}

// We don't need these
// def typeNum : EncoderM Nat := do return (← get).types.size
// def termNum : EncoderM Nat := do return (← get).terms.size
// def uufNum  : EncoderM Nat := do return (← get).uufs.size

impl<'a, S> Encoder<'a, S> {
    /// Corresponds to `EncoderState.init` in Lean
    pub fn new(env: &'a SymEnv, script: S) -> Result<Self> {
        Ok(Encoder {
            terms: BTreeMap::new(),
            types: BTreeMap::new(),
            uufs: BTreeMap::new(),
            enums: env
                .entities
                .iter()
                .filter_map(|(ety, d)| Some((ety, d.members.as_ref()?)))
                .collect(),
            script,
        })
    }

    /// "Finalize" the encoder, removing the ability to write to the `script`, and thus also dropping all borrows inherent in the `S` type.
    /// The resulting encoder can have its state inspected (e.g., by the decoder), but can no longer encode anything new.
    pub fn finalize(self) -> Encoder<'a, ()> {
        Encoder {
            terms: self.terms,
            types: self.types,
            uufs: self.uufs,
            enums: self.enums,
            script: (),
        }
    }
}

impl<S: tokio::io::AsyncWrite + Unpin + Send> Encoder<'_, S> {
    /// Returns `id` to match the Lean
    pub async fn declare_type<T: AsRef<str>>(
        &mut self,
        id: T,
        mks: impl IntoIterator<Item = &str>,
    ) -> Result<T> {
        self.script
            .declare_datatype(id.as_ref(), vec![], mks)
            .await?;
        Ok(id)
    }

    pub async fn declare_entity_type(&mut self, ety: &EntityType) -> Result<SmolStr> {
        let ety_id = entity_type_id(self.types.len());
        match self.enums.get(ety) {
            Some(members) => {
                self.script
                    .comment(&format_smolstr!("{ety}::[{}]", members.iter().join(", ")))
                    .await?;
                let mks: Vec<_> = members
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format_smolstr!("({})", enum_id(&ety_id, i)))
                    .collect();
                self.declare_type(ety_id, mks.iter().map(|s| s.as_str()))
                    .await
            }
            None => {
                self.script.comment(&ety.to_string()).await?;
                self.declare_type(
                    ety_id.clone(),
                    [format_smolstr!("({ety_id} ({ety_id}_eid String))").as_str()],
                )
                .await
            }
        }
    }

    pub async fn declare_ext_type(&mut self, ext_ty: ExtType) -> Result<&'static str> {
        match ext_ty {
            ExtType::Decimal => {
                self.declare_type("Decimal", ["(Decimal (decimalVal (_ BitVec 64)))"])
                    .await
            }
            ExtType::IpAddr => {
                self.declare_type(
                    "IPAddr",
                    [
                        "(V4 (addrV4 (_ BitVec 32)) (prefixV4 (Option (_ BitVec 5))))",
                        "(V6 (addrV6 (_ BitVec 128)) (prefixV6 (Option (_ BitVec 7))))",
                    ],
                )
                .await
            }
            ExtType::Duration => {
                self.declare_type("Duration", ["(Duration (durationVal (_ BitVec 64)))"])
                    .await
            }
            ExtType::DateTime => {
                self.declare_type("Datetime", ["(Datetime (datetimeVal (_ BitVec 64)))"])
                    .await
            }
        }
    }

    pub async fn declare_record_type<'r>(
        &mut self,
        rty: impl IntoIterator<Item = &'r (Attr, SmolStr)> + Clone,
    ) -> Result<SmolStr> {
        let rty_id = record_type_id(self.types.len());
        let mut attrs = rty
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, (_, ty))| format_smolstr!("({} {})", record_attr_id(&rty_id, i), ty));
        self.script
            .comment(&format_smolstr!(
                "{{{}}}",
                rty.into_iter().map(|(k, _)| k).join(", ")
            ))
            .await?;
        self.declare_type(
            rty_id.clone(),
            [format_smolstr!("({} {})", rty_id, attrs.join(" ")).as_str()],
        )
        .await
    }

    #[async_recursion]
    pub async fn encode_type(&mut self, ty: &TermType) -> Result<SmolStr> {
        match self.types.get(ty) {
            Some(enc) => Ok(enc.clone()),
            None => {
                let enc = match ty {
                    TermType::Bool => {
                        return Ok(SmolStr::new_static("Bool"));
                    }
                    TermType::String => {
                        return Ok(SmolStr::new_static("String"));
                    }
                    TermType::Bitvec { n } => {
                        return Ok(format_smolstr!("(_ BitVec {n})"));
                    }
                    TermType::Option { ref ty } => {
                        return Ok(format_smolstr!("(Option {})", self.encode_type(ty).await?));
                    }
                    TermType::Set { ty } => {
                        return Ok(format_smolstr!("(Set {})", self.encode_type(ty).await?));
                    }
                    TermType::Entity { ety } => self.declare_entity_type(ety).await?,
                    TermType::Ext { xty } => {
                        SmolStr::new_static(self.declare_ext_type(*xty).await?)
                    }
                    TermType::Record { rty } => {
                        let mut record_type = Vec::with_capacity(rty.len());
                        for (k, v) in rty.iter() {
                            record_type.push((k.clone(), self.encode_type(v).await?));
                        }
                        self.declare_record_type(record_type.iter()).await?
                    }
                };
                self.types.insert(ty.clone(), enc.clone());
                Ok(enc)
            }
        }
    }

    pub async fn declare_var(&mut self, v: &TermVar, ty_enc: &str) -> Result<SmolStr> {
        let id = term_id(self.terms.len());
        self.script.comment(&format_smolstr!("{:?}", v.id)).await?;
        self.script.declare_const(&id, ty_enc).await?;
        Ok(id)
    }

    pub async fn define_term(&mut self, ty_enc: &str, t_enc: &str) -> Result<SmolStr> {
        let id = term_id(self.terms.len());
        self.script.define_fun(&id, [], ty_enc, t_enc).await?;
        Ok(id)
    }

    pub async fn define_set<'s>(
        &mut self,
        ty_enc: &str,
        t_encs: impl ExactSizeIterator<Item = &'s str>,
    ) -> Result<SmolStr> {
        let set_term = if t_encs.len() == 0 {
            format!("(as set.empty {ty_enc})")
        } else {
            format!(
                "(set.insert {} (as set.empty {}))",
                t_encs.format(" "),
                ty_enc
            )
        };
        self.define_term(ty_enc, &set_term).await
    }

    pub async fn define_record<'s>(
        &mut self,
        ty_enc: &str,
        t_encs: impl IntoIterator<Item = &'s str>,
    ) -> Result<SmolStr> {
        let t_encs = t_encs.into_iter().join(" ");
        let t_enc = if t_encs.is_empty() {
            ty_enc
        } else {
            &format_smolstr!("({ty_enc} {})", t_encs)
        };
        self.define_term(ty_enc, t_enc).await
    }

    pub async fn encode_uuf(&mut self, uuf: &Uuf) -> Result<SmolStr> {
        match self.uufs.get(uuf) {
            Some(enc) => Ok(enc.clone()),
            None => {
                let id = uuf_id(self.uufs.len());
                self.script.comment(&uuf.id).await?;
                let encoded_arg_type = self.encode_type(&uuf.arg).await?;
                let encoded_out_type = self.encode_type(&uuf.out).await?;
                self.script
                    .declare_fun(&id, [encoded_arg_type.as_str()], &encoded_out_type)
                    .await?;
                self.uufs.insert(uuf.clone(), id.clone());
                Ok(id)
            }
        }
    }

    pub async fn define_entity(&mut self, ty_enc: &str, entity: &EntityUID) -> Result<SmolStr> {
        match self.enums.get(entity.type_name()) {
            Some(members) => {
                let entity_ind = match members
                    .iter()
                    .position(|s| s == <EntityID as AsRef<str>>::as_ref(entity.id()))
                {
                    Some(ind) => ind,
                    None => return Err(EncodeError::EnumMissingMember(entity.clone())),
                };
                Ok(enum_id(ty_enc, entity_ind))
            }
            None => {
                self.define_term(
                    ty_enc,
                    &format_smolstr!(
                        "({ty_enc} \"{}\")",
                        encode_string(<EntityID as AsRef<str>>::as_ref(entity.id())).ok_or_else(
                            || EncodeError::EncodeStringFailed(format_smolstr!(
                                "{:?}",
                                entity.id()
                            ))
                        )?
                    ),
                )
                .await
            }
        }
    }

    fn index_of_attr(a: &Attr, t_ty: &TermType) -> Result<usize> {
        // Getting the index of a key in `BTreeMap` should be ok
        // (it wouldn't be for `HashMap`)
        match t_ty {
            TermType::Record { rty } => match rty.keys().position(|k| k == a) {
                Some(ind) => Ok(ind),
                None => Err(EncodeError::RecordMissingAttr(a.clone())),
            },
            _ => Err(EncodeError::ExpectRecord(t_ty.clone())),
        }
    }

    pub async fn define_record_get(
        &mut self,
        ty_enc: &str,
        a: &Attr,
        t_enc: &str,
        ty: &TermType,
    ) -> Result<SmolStr> {
        let r_id = match self.types.get(ty) {
            Some(t) => t,
            None => return Err(EncodeError::MissingTypeEncoding(ty.clone())),
        };
        let a_id = Self::index_of_attr(a, ty)?;
        self.define_term(
            ty_enc,
            &format_smolstr!("({} {t_enc})", record_attr_id(r_id, a_id)),
        )
        .await
    }

    pub async fn define_app<'b>(
        &mut self,
        ty_enc: &str,
        op: &Op,
        t_encs: impl IntoIterator<Item = SmolStr>,
        ts: impl IntoIterator<Item = &'b Term>,
    ) -> Result<SmolStr> {
        let args = t_encs.into_iter().join(" ");
        match op {
            Op::RecordGet(a) => {
                let ty = match ts.into_iter().next() {
                    Some(t) => t.type_of(),
                    None => return Err(EncodeError::MalformedRecordGet),
                };
                self.define_record_get(ty_enc, a, &args, &ty).await
            }
            Op::StringLike(p) => {
                self.define_term(
                    ty_enc,
                    &format_smolstr!(
                        "(str.in_re {args} {})",
                        encode_pattern(p)
                            .ok_or_else(|| EncodeError::EncodePatternFailed(p.clone()))?
                    ),
                )
                .await
            }
            Op::Uuf(f) => {
                let encoded_uuf = self.encode_uuf(f).await?;
                self.define_term(ty_enc, &format_smolstr!("({} {args})", encoded_uuf))
                    .await
            }
            _ => {
                self.define_term(ty_enc, &format_smolstr!("({} {args})", encode_op(op)))
                    .await
            }
        }
    }

    #[async_recursion]
    pub async fn encode_term(&mut self, t: &Term) -> Result<SmolStr> {
        if let Some(enc) = self.terms.get(t) {
            return Ok(enc.clone());
        }
        let ty_enc = self.encode_type(&t.type_of()).await?;
        let enc = match &t {
            Term::Var(v) => self.declare_var(v, &ty_enc).await?,
            Term::Prim(p) => match p {
                TermPrim::Bool(b) => {
                    return Ok({
                        if *b {
                            SmolStr::new_static("true")
                        } else {
                            SmolStr::new_static("false")
                        }
                    });
                }
                TermPrim::Bitvec(bv) => {
                    return Ok(encode_bitvec(bv));
                }
                TermPrim::String(s) => {
                    return Ok(format_smolstr!(
                        "\"{}\"",
                        encode_string(s)
                            .ok_or_else(|| EncodeError::EncodeStringFailed(s.clone()))?
                    ));
                }
                TermPrim::Entity(e) => self.define_entity(&ty_enc, e).await?,
                TermPrim::Ext(x) => self.define_term(&ty_enc, &encode_ext(x)).await?,
            },
            Term::None(_) => {
                self.define_term(&ty_enc, &format_smolstr!("(as none {ty_enc})"))
                    .await?
            }
            Term::Some(t1) => {
                let encoded_term = self.encode_term(t1).await?;
                self.define_term(&ty_enc, &format_smolstr!("(some {encoded_term})"))
                    .await?
            }
            Term::Set { elts, .. } => {
                let mut encoded_terms = Vec::with_capacity(elts.len());
                for elt in elts.iter() {
                    encoded_terms.push(self.encode_term(elt).await?);
                }
                self.define_set(&ty_enc, encoded_terms.iter().map(|s| s.as_str()))
                    .await?
            }
            Term::Record(ats) => {
                let mut encoded_terms = Vec::with_capacity(ats.len());
                for t in ats.values() {
                    encoded_terms.push(self.encode_term(t).await?);
                }
                self.define_record(&ty_enc, encoded_terms.iter().map(|s| s.as_str()))
                    .await?
            }
            Term::App {
                op: Op::Bvnego,
                args,
                ret_ty: TermType::Bool,
            } if args.len() == 1 => {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "Slice of length 1 can be indexed by 0"
                )]
                let t = &args[0]; // guaranteed to exist because we already checked that `args.len() == 1`

                // don't encode bvnego itself, for compatibility with older CVC5 (bvnego was
                // introduced in CVC5 1.1.2)
                // this rewrite is done in the encoder and is thus trusted; see notes here in
                // the Lean
                match t.type_of() {
                    TermType::Bitvec { n } => {
                        // more fancy and possibly more optimized, but hard to prove termination in Lean:
                        // self.encode_term(&factory::eq(t, &BitVec::int_min(n))).await?
                        let t_enc = self.encode_term(t).await?;
                        self.define_app(
                            &ty_enc,
                            &Op::Eq,
                            [t_enc, encode_bitvec(&BitVec::int_min(n))],
                            [t, &BitVec::int_min(n).into()],
                        )
                        .await?
                    }
                    _ => {
                        debug_assert!(false, "`Bvnego` should only be applied to `Bitvec`");
                        // we could put anything here and be sound, because `Bvnego` should only be
                        // applied to Terms of type `Bitvec`
                        SmolStr::new_static("false")
                    }
                }
            }
            Term::App { op, args, .. } => {
                #[cfg(feature = "anyall")]
                if matches!(op, Op::SetAll) {
                    // D-52 tri-valued `set.filter` encoding (HO_ALL), mirroring
                    // Lean's `Encoder.lean` `set.all` arm:
                    //   ite (not (set.filter (lambda ((!anyall!it T)) ERR) S = empty))
                    //       none
                    //       (some (set.filter (lambda ((!anyall!it T)) P) S = S))
                    #[expect(clippy::indexing_slicing, reason = "set.all always has 3 args")]
                    let (set_t, pred_t, err_t) = (&args[0], &args[1], &args[2]);
                    let elem_ty = match set_t.type_of() {
                        TermType::Set { ty } => std::sync::Arc::unwrap_or_clone(ty),
                        _ => return Err(EncodeError::SetAllNotEncoded),
                    };
                    let elem_ty_enc = self.encode_type(&elem_ty).await?;
                    let s_enc = self.encode_term(set_t).await?;
                    let pred_enc = self.encode_term_inline(pred_t).await?;
                    let err_enc = self.encode_term_inline(err_t).await?;
                    let val_filter = format_smolstr!(
                        "(set.filter (lambda ((!anyall!it {elem_ty_enc})) {pred_enc}) {s_enc})"
                    );
                    let err_filter = format_smolstr!(
                        "(set.filter (lambda ((!anyall!it {elem_ty_enc})) {err_enc}) {s_enc})"
                    );
                    let v = format_smolstr!("(= {val_filter} {s_enc})");
                    let e = format_smolstr!(
                        "(not (= {err_filter} (as set.empty (Set {elem_ty_enc}))))"
                    );
                    let enc = self
                        .define_term(&ty_enc, &format_smolstr!("(ite {e} (as none {ty_enc}) (some {v}))"))
                        .await?;
                    self.terms.insert(t.clone(), enc.clone());
                    return Ok(enc);
                }
                let mut encoded_terms = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    encoded_terms.push(self.encode_term(arg).await?);
                }
                self.define_app(&ty_enc, op, encoded_terms, args.iter())
                    .await?
            }
        };
        self.terms.insert(t.clone(), enc.clone());
        Ok(enc)
    }

    /// Encode a Term *inline* (not in A-normal form) for use inside a
    /// `set.filter` lambda body (the `set.all` encoding, feature `anyall`,
    /// D-52/D-53). Mirrors Lean `Encoder.encodeTermInline`. Subterms that do NOT
    /// mention the reserved bound variable `!anyall!it` are closed w.r.t. the
    /// binder and delegated to the ANF `encode_term` (returning a top-level id,
    /// valid inside the lambda); only subterms that mention it are inlined.
    #[cfg(feature = "anyall")]
    #[async_recursion]
    pub async fn encode_term_inline(&mut self, t: &Term) -> Result<SmolStr> {
        if !mentions_anyall_it(t) {
            return self.encode_term(t).await;
        }
        match t {
            Term::Var(v) => {
                if v.id == "!anyall!it" {
                    Ok(SmolStr::new_static("!anyall!it"))
                } else {
                    self.encode_term(t).await
                }
            }
            Term::Some(inner) => {
                Ok(format_smolstr!("(some {})", self.encode_term_inline(inner).await?))
            }
            Term::Set { elts, elts_ty } => {
                let set_ty_enc = self.encode_type(&TermType::set_of(elts_ty.clone())).await?;
                if elts.is_empty() {
                    Ok(format_smolstr!("(as set.empty {set_ty_enc})"))
                } else {
                    let mut encs = Vec::with_capacity(elts.len());
                    for e in elts.iter() {
                        encs.push(self.encode_term_inline(e).await?);
                    }
                    Ok(format_smolstr!(
                        "(set.insert {} (as set.empty {set_ty_enc}))",
                        encs.into_iter().join(" ")
                    ))
                }
            }
            Term::App { op, args, .. } => {
                let args_slice = args.as_slice();
                match (op, args_slice) {
                    (Op::RecordGet(a), [r]) => {
                        let r_enc = self.encode_term_inline(r).await?;
                        let r_ty = r.type_of();
                        let r_id = match self.types.get(&r_ty) {
                            Some(id) => id.clone(),
                            None => return Err(EncodeError::MissingTypeEncoding(r_ty.clone())),
                        };
                        let a_id = Self::index_of_attr(a, &r_ty)?;
                        Ok(format_smolstr!("({} {r_enc})", record_attr_id(&r_id, a_id)))
                    }
                    (Op::StringLike(p), [a]) => {
                        let a_enc = self.encode_term_inline(a).await?;
                        let p_enc = encode_pattern(p).ok_or_else(|| {
                            EncodeError::EncodePatternFailed(p.clone())
                        })?;
                        Ok(format_smolstr!("(str.in_re {a_enc} {p_enc})"))
                    }
                    (Op::OptionGet, [a]) => {
                        Ok(format_smolstr!("(val {})", self.encode_term_inline(a).await?))
                    }
                    (Op::Uuf(f), [a]) => {
                        let f_enc = self.encode_uuf(f).await?;
                        Ok(format_smolstr!("({f_enc} {})", self.encode_term_inline(a).await?))
                    }
                    (Op::Ext(xop), [a]) => {
                        Ok(format_smolstr!(
                            "({} {})",
                            encode_ext_op(xop),
                            self.encode_term_inline(a).await?
                        ))
                    }
                    (Op::Eq, [a, b]) => {
                        let a_enc = self.encode_term_inline(a).await?;
                        let b_enc = self.encode_term_inline(b).await?;
                        Ok(format_smolstr!("(= {a_enc} {b_enc})"))
                    }
                    (Op::ZeroExtend(n), [a]) => {
                        Ok(format_smolstr!(
                            "((_ zero_extend {n}) {})",
                            self.encode_term_inline(a).await?
                        ))
                    }
                    _ => {
                        let mut encs = Vec::with_capacity(args.len());
                        for a in args.iter() {
                            encs.push(self.encode_term_inline(a).await?);
                        }
                        Ok(format_smolstr!("({} {})", encode_op(op), encs.into_iter().join(" ")))
                    }
                }
            }
            // `prim`/`none` never mention the bound var (handled by the fast path
            // above); `record` with the bound var falls here only if a record
            // literal mentions `it`, which the set-free predicate fragment does
            // produce (a record of `it`-attributes) — encode it inline.
            Term::Record(ats) => {
                let r_ty = t.type_of();
                let r_ty_enc = self.encode_type(&r_ty).await?;
                if ats.is_empty() {
                    Ok(r_ty_enc)
                } else {
                    let mut encs = Vec::with_capacity(ats.len());
                    for v in ats.values() {
                        encs.push(self.encode_term_inline(v).await?);
                    }
                    Ok(format_smolstr!("({r_ty_enc} {})", encs.into_iter().join(" ")))
                }
            }
            Term::Prim(_) | Term::None(_) => self.encode_term(t).await,
        }
    }

    /// Once you've generated `Asserts` with one of the functions in verifier.rs, you
    /// can use this function to encode them as SMTLib assertions.
    ///
    /// Note that `encode()` itself first resets the solver in order to define datatypes
    /// etc.
    ///
    /// In Lean, this is a standalone function which takes a `SymEnv`, uses that to
    /// construct an `Encoder` (`EncoderState` in Lean), and then does the encoding.
    /// Here in Rust, we have this as a method on `Encoder`, so the caller first
    /// constructs an `Encoder` themselves with the `SymEnv`, then calls this.
    pub async fn encode(&mut self, ts: impl ExactSizeIterator<Item = &Term>) -> Result<()> {
        self.script
            .declare_datatype("Option", ["X"], ["(none)", "(some (val X))"])
            .await?;
        let mut ids: Vec<_> = Vec::with_capacity(ts.len());
        for t in ts {
            let id = self.encode_term(t).await?;
            ids.push(id);
        }
        for id in ids {
            self.script.assert(&id).await?;
        }
        Ok(())
    }
}

/// The maximum Unicode code point supported in SMT-LIB 2.7.
/// Also see `num_codes` in cvc5:
/// https://github.com/cvc5/cvc5/blob/b78e7ed23348659db52a32765ad181ae0c26bbd5/src/util/string.h#L53
pub const SMT_LIB_MAX_CODE_POINT: u32 = 196607;

/// This function needs to encode unicode strings with two levels of
/// escape sequences:
/// - At the string theory level, we need to encode all non-printable
///   unicode characters as `\u{xxxx}`, where a character is printable
///   if its code point is within [32, 126] (see also the note on string
///   literals in https://smt-lib.org/theories-UnicodeStrings.shtml).
/// - At the parser level, we need to replace any single `"` character
///   with `""`, according to the SMT-LIB 2.7 standard on string literals:
///   https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-07-07.pdf
///
/// Note in particular that `\\` is NOT an escape sequence,
/// so cvc5 will read `\\u{0}` as a two-character string with
/// characters `\u{5c}` and `\0`.
pub(super) fn encode_string(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '"' {
            out.push_str("\"\"");
        } else if c == '\\' {
            // This is to avoid unexpectedly escape some characters
            out.push_str("\\u{5c}");
        } else if 32 as char <= c && c <= 126 as char {
            out.push(c);
        } else {
            // Encode non-printable character
            if c as u32 > SMT_LIB_MAX_CODE_POINT {
                return None; // Invalid code point for SMT-LIB
            }
            #[expect(clippy::unwrap_used, reason = "writing string cannot fail")]
            write!(out, "\\u{{{:x}}}", c as u32).unwrap();
        }
    }
    Some(out)
}

fn encode_bitvec(bv: &BitVec) -> SmolStr {
    format_smolstr!("(_ bv{} {})", bv.as_nat(), bv.width())
}

fn encode_ipaddr_prefix_v4(pre: &IPv4Prefix) -> SmolStr {
    match pre.as_bitvec() {
        Some(pre) => format_smolstr!("(some {})", encode_bitvec(pre)),
        None => format_smolstr!("(as none (Option (_ BitVec {V4_WIDTH})))"),
    }
}

fn encode_ipaddr_prefix_v6(pre: &IPv6Prefix) -> SmolStr {
    match pre.as_bitvec() {
        Some(pre) => format_smolstr!("(some {})", encode_bitvec(pre)),
        None => format_smolstr!("(as none (Option (_ BitVec {V6_WIDTH})))"),
    }
}

fn encode_ext(e: &Ext) -> SmolStr {
    match e {
        Ext::Decimal { d } => {
            let bv_enc = encode_bitvec(&BitVec::of_int(SIXTY_FOUR, d.0.into()));
            format_smolstr!("(Decimal {bv_enc})")
        }
        Ext::Ipaddr {
            ip: IPNet::V4(CIDRv4 { addr, prefix }),
        } => {
            let addr = encode_bitvec(addr.as_bitvec());
            let pre = encode_ipaddr_prefix_v4(prefix);
            format_smolstr!("(V4 {addr} {pre})")
        }
        Ext::Ipaddr {
            ip: IPNet::V6(CIDRv6 { addr, prefix }),
        } => {
            let addr = encode_bitvec(addr.as_bitvec());
            let pre = encode_ipaddr_prefix_v6(prefix);
            format_smolstr!("(V6 {addr} {pre})")
        }
        Ext::Duration { d } => {
            let bv_enc = encode_bitvec(&BitVec::of_int(SIXTY_FOUR, d.to_milliseconds().into()));
            format_smolstr!("(Duration {bv_enc})")
        }
        Ext::Datetime { dt } => {
            let bv_enc = encode_bitvec(&BitVec::of_i128(SIXTY_FOUR, i64::from(dt).into()));
            format_smolstr!("(Datetime {bv_enc})")
        }
    }
}

fn encode_ext_op(ext_op: &ExtOp) -> &'static str {
    match ext_op {
        ExtOp::DecimalVal => "decimalVal",
        ExtOp::IpaddrIsV4 => "(_ is V4)",
        ExtOp::IpaddrAddrV4 => "addrV4",
        ExtOp::IpaddrPrefixV4 => "prefixV4",
        ExtOp::IpaddrAddrV6 => "addrV6",
        ExtOp::IpaddrPrefixV6 => "prefixV6",
        ExtOp::DatetimeVal => "datetimeVal",
        ExtOp::DatetimeOfBitVec => "Datetime",
        ExtOp::DurationVal => "durationVal",
        ExtOp::DurationOfBitVec => "Duration",
    }
}

/// Does the term mention the reserved `.all` bound element variable
/// `!anyall!it`? Mirrors Lean `Term.mentionsAnyAllIt`. A subterm that does not
/// is closed w.r.t. the `set.filter` binder and can be ANF-lifted (encoded with
/// `encode_term`); only subterms that do must be encoded inline (D-53).
#[cfg(feature = "anyall")]
fn mentions_anyall_it(t: &Term) -> bool {
    match t {
        Term::Var(v) => v.id == "!anyall!it",
        Term::Prim(_) | Term::None(_) => false,
        Term::Some(inner) => mentions_anyall_it(inner),
        Term::Set { elts, .. } => elts.iter().any(mentions_anyall_it),
        Term::Record(ats) => ats.values().any(mentions_anyall_it),
        Term::App { args, .. } => args.iter().any(mentions_anyall_it),
    }
}

fn encode_op(op: &Op) -> SmolStr {
    match op {
        Op::Eq => SmolStr::new_static("="),
        Op::ZeroExtend(n) => format_smolstr!("(_ zero_extend {n})"),
        Op::OptionGet => SmolStr::new_static("val"),
        Op::Ext(xop) => SmolStr::new_static(encode_ext_op(xop)),
        _ => SmolStr::new_static(op.mk_name()),
    }
}

fn encode_pat_elem(pat_elem: PatternElem) -> Option<SmolStr> {
    Some(match pat_elem {
        PatternElem::Wildcard => SmolStr::new_static("(re.* re.allchar)"),
        PatternElem::Char(c) => {
            format_smolstr!("(str.to_re \"{}\")", encode_string(&c.to_smolstr())?)
        }
    })
}

fn encode_pattern(pattern: &OrdPattern) -> Option<SmolStr> {
    if pattern.get_elems().is_empty() {
        Some(SmolStr::new_static("(str.to_re \"\")"))
    } else if pattern.get_elems().len() == 1 {
        #[expect(
            clippy::indexing_slicing,
            reason = "Slice of length 1 can be indexed by 0"
        )]
        encode_pat_elem(pattern.get_elems()[0])
    } else {
        Some(format_smolstr!(
            "(re.++ {})",
            pattern
                .iter()
                .copied()
                .map(encode_pat_elem)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .join(" ")
        ))
    }
}

#[cfg(test)]
mod unit_tests {
    use std::{collections::BTreeSet, str::FromStr};

    use crate::symcc::env::{SymEntities, SymEnv, SymRequest};
    use cedar_policy::EntityTypeName;
    use smol_str::SmolStr;

    use super::Encoder;
    use crate::symcc::term_type::TermType;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn declare_type() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder
            .declare_type("foo", ["(Bar1 (baz String))"])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn declare_entity_type() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        let ety = cedar_policy::EntityTypeName::from_str("User").unwrap();
        let empty_set = BTreeSet::new();
        encoder.enums.insert(&ety, &empty_set);
        encoder.declare_entity_type(&ety).await.unwrap();
    }

    #[tokio::test]
    async fn declare_empty_record_type() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder.declare_record_type(vec![]).await.unwrap();
    }

    #[tokio::test]
    async fn declare_record_type() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder
            .declare_record_type(std::iter::once(&("foo".into(), SmolStr::new_static("bar"))))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn encode_bool_type() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder.encode_type(&TermType::Bool).await.unwrap();
    }

    #[tokio::test]
    async fn encode_string_type() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder.encode_type(&TermType::String).await.unwrap();
    }

    #[tokio::test]
    async fn encode_uuf() {
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        let my_uuf = crate::symcc::op::Uuf {
            id: "my_fun".into(),
            arg: TermType::Bool,
            out: TermType::Bool,
        };
        encoder.encode_uuf(&my_uuf).await.unwrap();
    }

    #[tokio::test]
    async fn define_entity() {
        use cedar_policy::EntityUid;
        let symenv = SymEnv {
            request: SymRequest::empty_sym_req(),
            entities: Arc::new(SymEntities(BTreeMap::new())),
        };
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        let entity_type_name = EntityTypeName::from_str("User").unwrap();
        let entity = EntityUid::from_type_name_and_id(
            entity_type_name.clone(),
            cedar_policy::EntityId::from_str("alice").unwrap(),
        );
        let entity_ty_enc = encoder
            .encode_type(&TermType::Entity {
                ety: entity_type_name,
            })
            .await
            .unwrap();
        encoder
            .define_entity(&entity_ty_enc, &entity)
            .await
            .unwrap();
    }

    /// Compiles `expr` against the schema shared with
    /// `compiler::ext_has_attr_tests` and returns the SMT text the encoder
    /// emits for it.
    async fn compile_and_encode(expr: &str) -> String {
        use crate::symcc::compiler::{
            compile,
            ext_has_attr_tests::{parse_expr, sym_env},
        };

        let symenv = sym_env();
        let term = compile(&parse_expr(expr), &symenv).unwrap();

        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder.encode_term(&term).await.unwrap();

        String::from_utf8(encoder.script).unwrap()
    }

    #[tokio::test]
    async fn ext_has_attr_compiles_to_expected_smt() {
        insta::assert_snapshot!(compile_and_encode("context has rec.x").await, @"(define-fun t0 () (Option Bool) (some true))");
    }

    // entity base, optional then present
    #[tokio::test]
    async fn ext_has_attr_entity_optional_then_present_smt() {
        insta::assert_snapshot!(compile_and_encode("principal has thing1.id").await, @r#"
        ; Thing
        (declare-datatype E0 (
          (E0 (E0_eid String))))
        ; Thing2
        (declare-datatype E1 (
          (E1 (E1_eid String))))
        ; {id, thing2, thing2bis}
        (declare-datatype R2 (
          (R2 (R2_a0 String) (R2_a1 E1) (R2_a2 (Option E1)))))
        ; {name, thing1, thing2, x, xopt}
        (declare-datatype R3 (
          (R3 (R3_a0 String) (R3_a1 (Option E0)) (R3_a2 E1) (R3_a3 R2) (R3_a4 (Option R2)))))
        ; User
        (declare-datatype E4 (
          (E4 (E4_eid String))))
        ; "principal"
        (declare-const t0 E4)
        ; attrs[User]
        (declare-fun f0 (E4) R3)
        (define-fun t1 () R3 (f0 t0))
        (define-fun t2 () (Option E0) (R3_a1 t1))
        (define-fun t3 () (Option E0) (as none (Option E0)))
        (define-fun t4 () Bool (= t2 t3))
        (define-fun t5 () Bool (not t4))
        (define-fun t6 () (Option Bool) (as none (Option Bool)))
        (define-fun t7 () (Option Bool) (some false))
        (define-fun t8 () (Option Bool) (ite t4 t6 t7))
        (define-fun t9 () (Option Bool) (ite t5 t8 t7))
        "#);
    }

    // entity base, present then optional
    #[tokio::test]
    async fn ext_has_attr_entity_present_then_optional_smt() {
        insta::assert_snapshot!(compile_and_encode("principal has thing2.opt").await, @r#"
        ; {id, opt}
        (declare-datatype R0 (
          (R0 (R0_a0 String) (R0_a1 (Option (_ BitVec 64))))))
        ; Thing2
        (declare-datatype E1 (
          (E1 (E1_eid String))))
        ; Thing
        (declare-datatype E2 (
          (E2 (E2_eid String))))
        ; {id, thing2, thing2bis}
        (declare-datatype R3 (
          (R3 (R3_a0 String) (R3_a1 E1) (R3_a2 (Option E1)))))
        ; {name, thing1, thing2, x, xopt}
        (declare-datatype R4 (
          (R4 (R4_a0 String) (R4_a1 (Option E2)) (R4_a2 E1) (R4_a3 R3) (R4_a4 (Option R3)))))
        ; User
        (declare-datatype E5 (
          (E5 (E5_eid String))))
        ; "principal"
        (declare-const t0 E5)
        ; attrs[User]
        (declare-fun f0 (E5) R4)
        (define-fun t1 () R4 (f0 t0))
        (define-fun t2 () E1 (R4_a2 t1))
        ; attrs[Thing2]
        (declare-fun f1 (E1) R0)
        (define-fun t3 () R0 (f1 t2))
        (define-fun t4 () (Option (_ BitVec 64)) (R0_a1 t3))
        (define-fun t5 () (Option (_ BitVec 64)) (as none (Option (_ BitVec 64))))
        (define-fun t6 () Bool (= t4 t5))
        (define-fun t7 () Bool (not t6))
        (define-fun t8 () (Option Bool) (some t7))
        "#);
    }

    // record base, present then optional
    #[tokio::test]
    async fn ext_has_attr_record_present_then_optional_smt() {
        insta::assert_snapshot!(compile_and_encode("principal.x has thing2.opt").await, @r#"
        ; {id, opt}
        (declare-datatype R0 (
          (R0 (R0_a0 String) (R0_a1 (Option (_ BitVec 64))))))
        ; Thing2
        (declare-datatype E1 (
          (E1 (E1_eid String))))
        ; {id, thing2, thing2bis}
        (declare-datatype R2 (
          (R2 (R2_a0 String) (R2_a1 E1) (R2_a2 (Option E1)))))
        ; Thing
        (declare-datatype E3 (
          (E3 (E3_eid String))))
        ; {name, thing1, thing2, x, xopt}
        (declare-datatype R4 (
          (R4 (R4_a0 String) (R4_a1 (Option E3)) (R4_a2 E1) (R4_a3 R2) (R4_a4 (Option R2)))))
        ; User
        (declare-datatype E5 (
          (E5 (E5_eid String))))
        ; "principal"
        (declare-const t0 E5)
        ; attrs[User]
        (declare-fun f0 (E5) R4)
        (define-fun t1 () R4 (f0 t0))
        (define-fun t2 () R2 (R4_a3 t1))
        (define-fun t3 () E1 (R2_a1 t2))
        ; attrs[Thing2]
        (declare-fun f1 (E1) R0)
        (define-fun t4 () R0 (f1 t3))
        (define-fun t5 () (Option (_ BitVec 64)) (R0_a1 t4))
        (define-fun t6 () (Option (_ BitVec 64)) (as none (Option (_ BitVec 64))))
        (define-fun t7 () Bool (= t5 t6))
        (define-fun t8 () Bool (not t7))
        (define-fun t9 () (Option Bool) (some t8))
        "#);
    }

    // record base, optional then present
    #[tokio::test]
    async fn ext_has_attr_record_optional_then_present_smt() {
        insta::assert_snapshot!(compile_and_encode("principal.x has thing2bis.id").await, @r#"
        ; Thing2
        (declare-datatype E0 (
          (E0 (E0_eid String))))
        ; {id, thing2, thing2bis}
        (declare-datatype R1 (
          (R1 (R1_a0 String) (R1_a1 E0) (R1_a2 (Option E0)))))
        ; Thing
        (declare-datatype E2 (
          (E2 (E2_eid String))))
        ; {name, thing1, thing2, x, xopt}
        (declare-datatype R3 (
          (R3 (R3_a0 String) (R3_a1 (Option E2)) (R3_a2 E0) (R3_a3 R1) (R3_a4 (Option R1)))))
        ; User
        (declare-datatype E4 (
          (E4 (E4_eid String))))
        ; "principal"
        (declare-const t0 E4)
        ; attrs[User]
        (declare-fun f0 (E4) R3)
        (define-fun t1 () R3 (f0 t0))
        (define-fun t2 () R1 (R3_a3 t1))
        (define-fun t3 () (Option E0) (R1_a2 t2))
        (define-fun t4 () (Option E0) (as none (Option E0)))
        (define-fun t5 () Bool (= t3 t4))
        (define-fun t6 () Bool (not t5))
        (define-fun t7 () (Option Bool) (as none (Option Bool)))
        (define-fun t8 () (Option Bool) (some true))
        (define-fun t9 () (Option Bool) (ite t5 t7 t8))
        (define-fun t10 () (Option Bool) (some false))
        (define-fun t11 () (Option Bool) (ite t6 t9 t10))
        "#);
    }
}

#[cfg(test)]
mod deep_extended_has_chain_tests {
    use crate::symcc::compiler::compile;
    use crate::symcc::test_utils::{deep_chain_sym_env, deep_has_chain_expr};

    use super::Encoder;

    async fn compile_encde_at_depth(depth: usize) -> String {
        let symenv = deep_chain_sym_env(depth);
        let term =
            compile(&deep_has_chain_expr(depth), &symenv).expect("expression should compile");
        let mut encoder = Encoder::new(&symenv, Vec::<u8>::new()).unwrap();
        encoder.encode_term(&term).await.unwrap();
        String::from_utf8(encoder.script).unwrap()
    }

    #[tokio::test]
    async fn nested_has_chain_encodes_linearly_not_exponentially() {
        let smt_at_2 = compile_encde_at_depth(2).await;
        let smt_at_3 = compile_encde_at_depth(3).await;
        let smt_at_4 = compile_encde_at_depth(4).await;
        let n_2 = smt_at_2.matches("define-fun").count();
        let n_3 = smt_at_3.matches("define-fun").count();
        let n_4 = smt_at_4.matches("define-fun").count();
        // Size increases linearly, not exponentially
        assert_eq!(n_3 - n_2, n_4 - n_3);
    }
}

#[cfg(all(test, feature = "anyall"))]
#[expect(clippy::panic, reason = "unit tests")]
mod anyall_encoder_tests {
    use std::str::FromStr;

    use cedar_policy::{RequestEnv, Schema};

    use super::*;
    use crate::symcc::compiler::compile;
    use crate::symcc::env::SymEnv;

    fn sym_env() -> SymEnv {
        let schema = Schema::from_cedarschema_str(
            r#"
            entity User;
            entity Thing;
            action View appliesTo {
                principal: [User],
                resource: [Thing],
                context: { ports: Set<Long> }
            };
            "#,
        )
        .unwrap()
        .0;
        let req = RequestEnv::new(
            "User".parse().unwrap(),
            "Action::\"View\"".parse().unwrap(),
            "Thing".parse().unwrap(),
        );
        SymEnv::new(&schema, &req).expect("malformed sym env")
    }

    async fn encode_expr(s: &str) -> String {
        let env = sym_env();
        let term = compile(&cedar_policy_core::ast::Expr::from_str(s).unwrap(), &env).unwrap();
        let mut encoder = Encoder::new(&env, Vec::<u8>::new()).unwrap();
        encoder.encode_term(&term).await.unwrap();
        String::from_utf8(encoder.script).unwrap()
    }

    // A symbolic `.all` emits the tri-valued two-`set.filter` encoding (D-52):
    // a value filter `(= (set.filter (lambda ((!anyall!it ...)) P) S) S)` and an
    // error filter `(not (= (set.filter (lambda ...) ERR) (as set.empty ...)))`,
    // combined `(ite err none (some val))`. We assert the structural pieces
    // rather than a full snapshot (ids/types are schema-dependent), which is
    // enough to prove the HO `set.filter` lambdas are emitted (not the generic
    // `(set.all ...)` App) and that both value and error filters appear.
    #[tokio::test]
    async fn all_emits_two_set_filter_lambdas() {
        let smt = encode_expr("context.ports.all(it > 0)").await;
        let filters = smt.matches("set.filter (lambda ((!anyall!it").count();
        assert_eq!(filters, 2, "expected two set.filter lambdas, got SMT:\n{smt}");
        assert!(smt.contains("(as set.empty"), "error filter should compare to empty set:\n{smt}");
        assert!(smt.contains("(as none"), "tri-valued result should carry a none branch:\n{smt}");
        // The generic App fallback `(set.all ...)` must NOT appear — the op is
        // emitted as the two filters, never as a bare `set.all` application.
        assert!(!smt.contains("(set.all "), "set.all must encode via set.filter, not as a bare op:\n{smt}");
    }

    // F-2 (mutation 3a-i): the VALUE filter body must be the PREDICATE and the
    // ERROR filter body must be the ERROR term — swapping the two lambdas
    // (`pred_enc`↔`err_enc`) is a silent SMT-semantics bug the lambda-COUNT
    // test above does not catch. For `it > 0` the value filter body is the
    // compiled predicate `(not (bvsle !anyall!it (_ bv0 64)))` and the error
    // filter body is `false` (the predicate never errors), so they are
    // distinguishable. The whole `set.all` encoding is pinned by snapshot so
    // ANY lambda-body swap (or template drift) diffs.
    #[tokio::test]
    async fn all_value_filter_is_predicate_error_filter_is_error() {
        let smt = encode_expr("context.ports.all(it > 0)").await;
        // Emission order (from the `(ite (not (= (set.filter … ERR …) empty))
        // none (some (= (set.filter … VAL …) S)))` shape): the ERROR filter
        // comes FIRST (in the `ite` condition), the VALUE filter SECOND (in the
        // `some`). This pins both the body→filter mapping and that ordering.
        let err_body = extract_filter_body(&smt, 0);
        let val_body = extract_filter_body(&smt, 1);
        assert!(
            val_body.contains("bvsle"),
            "VALUE filter body must be the predicate `(not (bvsle …))`, got `{val_body}`\nSMT:\n{smt}"
        );
        assert_eq!(
            err_body, "false",
            "ERROR filter body must be `false` for a non-erroring predicate, got `{err_body}`\nSMT:\n{smt}"
        );
        // And the predicate (bvsle) must NOT appear in the error filter.
        assert!(
            !err_body.contains("bvsle"),
            "a lambda swap put the predicate in the ERROR filter: `{err_body}`\nSMT:\n{smt}"
        );
    }

    /// Return the body of the `n`-th `set.filter (lambda ((!anyall!it T)) BODY)`
    /// in `smt` (0-indexed, emission order), via balanced-paren scanning. `T`
    /// may itself be parenthesized (e.g. `(_ BitVec 64)`).
    fn extract_filter_body(smt: &str, n: usize) -> String {
        let needle = "set.filter (lambda ((!anyall!it ";
        let mut from = 0usize;
        for _ in 0..=n {
            let i = smt[from..].find(needle).expect("set.filter lambda not found");
            from += i + needle.len();
        }
        // Now positioned at the start of the type token `T` inside
        // `((!anyall!it T)) BODY)`. Skip T (a balanced parens group, or an atom).
        let rest = &smt[from..];
        let mut p = 0usize;
        if rest.starts_with('(') {
            let mut depth = 0i32;
            for (i, c) in rest.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            p = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            p = rest.find(|c: char| c == ')' || c.is_whitespace()).unwrap_or(rest.len());
        }
        // After T comes `))` closing the binding and the binding-list, then the body.
        let rest = &rest[p..];
        let after = rest.find("))").expect("binding-list close `))`") + 2;
        let body_rel = after
            + rest[after..]
                .find(|c: char| !c.is_whitespace())
                .expect("lambda body");
        let b = &rest[body_rel..];
        if !b.starts_with('(') {
            let end = b.find(|c: char| c == ')' || c.is_whitespace()).unwrap_or(b.len());
            return b[..end].to_string();
        }
        let mut depth = 0i32;
        for (i, c) in b.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return b[..=i].to_string();
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced lambda body");
    }

    // `.any` lowers to `!all(!p)`, so its encoding also carries the two filters.
    #[tokio::test]
    async fn any_emits_two_set_filter_lambdas() {
        let smt = encode_expr("context.ports.any(it > 0)").await;
        let filters = smt.matches("set.filter (lambda ((!anyall!it").count();
        assert_eq!(filters, 2, "`.any` should still emit two set.filter lambdas:\n{smt}");
    }
}
