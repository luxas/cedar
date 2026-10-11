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

//! This module contains the type-aware partial evaluator.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    ast::{self, BinaryOp, EntityUID, PartialValue, Set, Value, ValueKind, Var},
    evaluator::stack_size_check,
    extensions::Extensions,
    validator::types::Type,
};

use crate::{
    tpe::entities::PartialEntities,
    tpe::request::PartialRequest,
    tpe::residual::{Residual, ResidualKind},
};

/// The partial evaluator
#[derive(Debug)]
pub struct Evaluator<'e> {
    pub(crate) request: &'e PartialRequest,
    pub(crate) entities: &'e PartialEntities,
    pub(crate) extensions: &'e Extensions<'e>,
}

impl Evaluator<'_> {
    /// Interpret a typed expression into a residual
    /// This function always succeeds because it wraps an error encountered
    /// into a `ResidualKind::Error`
    #[expect(clippy::cognitive_complexity, reason = "experimental feature")]
    pub fn interpret(&self, r: &Residual) -> Residual {
        let kind = match r {
            Residual::Concrete { .. } => {
                return r.clone();
            }
            Residual::Error(_) => {
                return r.clone();
            }
            Residual::Partial { kind, .. } => kind,
        };
        // Do not define a ty variable in this scope, to avoid ambiguity, but instead propagate
        // the ty in the return value, the type does not change during evaluation.
        let mk_error = || Residual::Error(r.ty().clone());
        let mk_residual = |kind: ResidualKind| Residual::Partial {
            kind,
            ty: r.ty().clone(),
        };
        let mk_concrete = |v: Value| Residual::Concrete {
            value: normalize_ext_value(v),
            ty: r.ty().clone(),
        };

        // Guard against stack overflows (just like the concrete evaluator), given the recursive
        // nature of interpret.
        match stack_size_check() {
            Ok(_) => (),
            Err(_) => return mk_error(),
        }

        match kind {
            ResidualKind::Var(Var::Action) => mk_concrete(self.request.action().clone().into()),
            ResidualKind::Var(Var::Principal) => {
                if let Ok(principal) = EntityUID::try_from(self.request.principal().clone()) {
                    mk_concrete(principal.into())
                } else {
                    mk_residual(ResidualKind::Var(Var::Principal))
                }
            }
            ResidualKind::Var(Var::Resource) => {
                if let Ok(resource) = EntityUID::try_from(self.request.resource().clone()) {
                    mk_concrete(resource.into())
                } else {
                    mk_residual(ResidualKind::Var(Var::Resource))
                }
            }
            ResidualKind::Var(Var::Context) => {
                if let Some(context) = self.request.context_attrs() {
                    mk_concrete(Value::record_arc(context.clone(), None))
                } else {
                    mk_residual(ResidualKind::Var(Var::Context))
                }
            }
            ResidualKind::And { left, right } => {
                let left = self.interpret(left);
                match &left {
                    Residual::Concrete { value, .. } => match value.get_as_bool() {
                        Ok(false) => mk_concrete(false.into()), // false && <right> => false
                        Ok(true) => self.interpret(right),      // true && <right> => <right>
                        Err(_) => mk_error(),                   // <error> && <right> => <error>
                    },
                    Residual::Partial { .. } => {
                        let right = self.interpret(right);
                        match &right {
                            Residual::Concrete { value, .. } => match value.get_as_bool() {
                                // <left-residual> && true => <left-residual>
                                Ok(true) => left,
                                Ok(false) => {
                                    if !left.can_error_assuming_well_formed() {
                                        // simplify <error-free> && false == false
                                        mk_concrete(false.into())
                                    } else {
                                        // cannot simplify <non-error-free> && false
                                        mk_residual(ResidualKind::And {
                                            left: Arc::new(left),
                                            right: Arc::new(mk_concrete(false.into())),
                                        })
                                    }
                                }
                                // "<left-residual> && <nonbool>" => "<left-residual> && <error>"
                                // TODO(luxas): Introduce a Residual::PartialError variant that says "the expression definitely errors, but with an unknown error"
                                Err(_) => mk_residual(ResidualKind::And {
                                    left: Arc::new(left),
                                    right: Arc::new(mk_error()),
                                }),
                            },
                            // Cannot simplify "<left-residual> && <right-residual>" or "<left-residual> && <error>"
                            // The latter expression could become a Residual::PartialError later.
                            Residual::Partial { .. } | Residual::Error(_) => {
                                mk_residual(ResidualKind::And {
                                    left: Arc::new(left),
                                    right: Arc::new(right),
                                })
                            }
                        }
                    }
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::Or { left, right } => {
                let left = self.interpret(left);
                match &left {
                    Residual::Concrete { value, .. } => match value.get_as_bool() {
                        Ok(true) => mk_concrete(true.into()), // true || <right> => true
                        Ok(false) => self.interpret(right),   // false || <right> => <right>
                        Err(_) => mk_error(),                 // <error> || <right> => <error>
                    },
                    Residual::Partial { .. } => {
                        let right = self.interpret(right);
                        match &right {
                            Residual::Concrete { value, .. } => match value.get_as_bool() {
                                // <left-residual> || false == <left-residual>
                                Ok(false) => left,
                                Ok(true) => {
                                    if !left.can_error_assuming_well_formed() {
                                        // simplify <error-free> || true == true
                                        mk_concrete(true.into())
                                    } else {
                                        // cannot simplify <non-error-free> || true
                                        mk_residual(ResidualKind::Or {
                                            left: Arc::new(left),
                                            right: Arc::new(mk_concrete(true.into())),
                                        })
                                    }
                                }
                                // "<left-residual> || <nonbool>" => "<left-residual> || <error>"
                                // Note that this is not necessarily a Residual::PartialError, as "<left-residual>" might evaluate to true,
                                // in which case the whole expression is true, regardless of the RHS error.
                                Err(_) => mk_residual(ResidualKind::Or {
                                    left: Arc::new(left),
                                    right: Arc::new(mk_error()),
                                }),
                            },
                            // Cannot simplify "<left-residual> || <right-residual>" or "<left-residual> || <error>"
                            Residual::Partial { .. } | Residual::Error(_) => {
                                mk_residual(ResidualKind::Or {
                                    left: Arc::new(left),
                                    right: Arc::new(right),
                                })
                            }
                        }
                    }
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::If {
                test_expr,
                then_expr,
                else_expr,
            } => {
                let test_expr = self.interpret(test_expr);
                match &test_expr {
                    Residual::Concrete { value, .. } => match value.get_as_bool() {
                        Ok(true) => self.interpret(then_expr), // (if true then <then> else <else>) => <then>
                        Ok(false) => self.interpret(else_expr), // (if false then <then> else <else>) => <else>
                        Err(_) => mk_error(), // (if <error> then <then> else <else>) => <error>
                    },
                    Residual::Partial { .. } => mk_residual(ResidualKind::If {
                        test_expr: Arc::new(test_expr),
                        then_expr: Arc::new(self.interpret(then_expr)),
                        else_expr: Arc::new(self.interpret(else_expr)),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::Is { expr, entity_type } => {
                let expr = self.interpret(expr);
                match &expr {
                    Residual::Concrete { value, .. } => match value.get_as_entity() {
                        Ok(uid) => mk_concrete((uid.entity_type() == entity_type).into()),
                        Err(_) => mk_error(), // <error> is <entity_type> => <error>
                    },
                    Residual::Partial {
                        kind: ResidualKind::Var(Var::Principal),
                        ..
                    } => mk_concrete((entity_type == self.request.principal_type()).into()),
                    Residual::Partial {
                        kind: ResidualKind::Var(Var::Resource),
                        ..
                    } => mk_concrete((entity_type == self.request.resource_type()).into()),
                    Residual::Partial { .. } => mk_residual(ResidualKind::Is {
                        expr: Arc::new(expr),
                        entity_type: entity_type.clone(),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::Like { expr, pattern } => {
                let expr = self.interpret(expr);
                match &expr {
                    Residual::Concrete { value, .. } => match value.get_as_string() {
                        Ok(s) => mk_concrete(pattern.wildcard_match(s).into()),
                        Err(_) => mk_error(), // <error> like <pattern> => <error>
                    },
                    Residual::Partial { .. } => mk_residual(ResidualKind::Like {
                        expr: Arc::new(expr),
                        pattern: pattern.clone(),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::BinaryApp { op, arg1, arg2 } => {
                let arg1 = self.interpret(arg1);
                let arg2 = self.interpret(arg2);
                let binapp_residual = |arg1, arg2| {
                    mk_residual(ResidualKind::BinaryApp {
                        op: *op,
                        arg1: Arc::new(arg1),
                        arg2: Arc::new(arg2),
                    })
                };
                match (&arg1, &arg2) {
                    (
                        Residual::Concrete { value: v1, .. },
                        Residual::Concrete { value: v2, .. },
                    ) => match op {
                        BinaryOp::Eq | BinaryOp::Less | BinaryOp::LessEq => {
                            if let Ok(v) =
                                crate::evaluator::binary_relation(*op, v1, v2, self.extensions)
                            {
                                mk_concrete(v)
                            } else {
                                mk_error()
                            }
                        }
                        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                            if let Ok(v) =
                                crate::evaluator::binary_arith(*op, v1.clone(), v2.clone(), None)
                            {
                                mk_concrete(v)
                            } else {
                                mk_error()
                            }
                        }
                        BinaryOp::In => {
                            let Ok(uid1) = v1.get_as_entity() else {
                                return mk_error();
                            };
                            if let Ok(uid2) = v2.get_as_entity() {
                                if uid1 == uid2 {
                                    return mk_concrete(true.into());
                                } else if let Some(ancestors) = self.entities.get_ancestors(uid1) {
                                    return mk_concrete(ancestors.contains(uid2).into());
                                }
                                binapp_residual(arg1, arg2)
                            } else if let Ok(uids) = v2.get_as_entity_set() {
                                let ancestors = self.entities.get_ancestors(uid1);
                                if uids.contains(&uid1)
                                    || ancestors.is_some_and(|ancestors| {
                                        uids.iter().any(|uid2| ancestors.contains(uid2))
                                    })
                                {
                                    // The set contains `uid1`, or `uid1` has known ancestors and the set contains one of these
                                    mk_concrete(true.into())
                                } else if !uids.is_empty() && ancestors.is_none() {
                                    // `uid1` was not in the set and has unknown ancestors, so we can't reach a concrete value
                                    // unless the set happens to be empty.
                                    binapp_residual(arg1, arg2)
                                } else {
                                    // `uid1` and it's ancestors are not in set
                                    mk_concrete(false.into())
                                }
                            } else {
                                mk_error()
                            }
                        }
                        BinaryOp::GetTag => {
                            let Ok(uid) = v1.get_as_entity() else {
                                return mk_error();
                            };
                            let Ok(tag) = v2.get_as_string() else {
                                return mk_error();
                            };
                            match self.entities.get_tags(uid) {
                                Some(tags) => match tags.get(tag) {
                                    Some(v) => mk_concrete(v.clone()),
                                    None => mk_error(),
                                },
                                None => binapp_residual(arg1, arg2),
                            }
                        }
                        BinaryOp::HasTag => {
                            let Ok(uid) = v1.get_as_entity() else {
                                return mk_error();
                            };
                            let Ok(tag) = v2.get_as_string() else {
                                return mk_error();
                            };
                            match self.entities.get_tags(uid) {
                                Some(tags) => mk_concrete(tags.contains_key(tag).into()),
                                None => binapp_residual(arg1, arg2),
                            }
                        }
                        BinaryOp::Contains => match v1.get_as_set() {
                            Ok(s) => mk_concrete(s.contains(v2).into()),
                            Err(_) => mk_error(),
                        },
                        BinaryOp::ContainsAll => match (v1.get_as_set(), v2.get_as_set()) {
                            (Ok(arg1_set), Ok(arg2_set)) => {
                                mk_concrete(arg2_set.is_subset(arg1_set).into())
                            }
                            _ => mk_error(),
                        },
                        BinaryOp::ContainsAny => match (v1.get_as_set(), v2.get_as_set()) {
                            (Ok(arg1_set), Ok(arg2_set)) => {
                                mk_concrete((!arg1_set.is_disjoint(arg2_set)).into())
                            }
                            _ => mk_error(),
                        },
                    },
                    (Residual::Error(_), _) => mk_error(),
                    (_, Residual::Error(_)) => mk_error(),
                    (_, _) => binapp_residual(arg1, arg2),
                }
            }
            ResidualKind::GetAttr { expr, attr } => {
                let expr = self.interpret(expr);
                match &expr {
                    Residual::Concrete { value, .. } => {
                        if let Ok(r) = value.get_as_record() {
                            if let Some(val) = r.as_ref().get(attr) {
                                mk_concrete(val.clone())
                            } else {
                                mk_error()
                            }
                        } else if let Ok(uid) = value.get_as_entity() {
                            match self.entities.get_attrs(uid) {
                                Some(attrs) => match attrs.get(attr) {
                                    Some(val) => mk_concrete(val.clone()),
                                    None => mk_error(),
                                },
                                None => mk_residual(ResidualKind::GetAttr {
                                    expr: Arc::new(expr),
                                    attr: attr.clone(),
                                }),
                            }
                        } else {
                            mk_error()
                        }
                    }
                    Residual::Partial { .. } => mk_residual(ResidualKind::GetAttr {
                        expr: Arc::new(expr),
                        attr: attr.clone(),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::HasAttr { expr, attr } => {
                let expr = self.interpret(expr);
                match &expr {
                    Residual::Concrete { value, .. } => {
                        if let Ok(r) = value.get_as_record() {
                            mk_concrete(r.as_ref().contains_key(attr).into())
                        } else if let Ok(uid) = value.get_as_entity() {
                            match self.entities.get_attrs(uid) {
                                Some(attrs) => mk_concrete(attrs.contains_key(attr).into()),
                                None => mk_residual(ResidualKind::HasAttr {
                                    expr: Arc::new(expr),
                                    attr: attr.clone(),
                                }),
                            }
                        } else {
                            mk_error()
                        }
                    }
                    Residual::Partial { .. } => mk_residual(ResidualKind::HasAttr {
                        expr: Arc::new(expr),
                        attr: attr.clone(),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::ExtHasAttr { expr, attrs } => {
                let expr = self.interpret(expr);
                match &expr {
                    Residual::Concrete { value, .. } => {
                        // Walk the attribute chain with short-circuit. The last
                        // attribute is only checked for membership (never
                        // dereferenced), matching the spec's `loop`.
                        let mut current_val = value;
                        let mut iter = attrs.iter();
                        while let Some(attr) = iter.next() {
                            // Resolve the attributes of the current value, whether
                            // it is a record or a (known) entity.
                            let current_attrs = if let Ok(r) = current_val.get_as_record() {
                                r.as_ref()
                            } else if let Ok(uid) = current_val.get_as_entity() {
                                match self.entities.get_attrs(uid) {
                                    Some(entity_attrs) => entity_attrs,
                                    None => {
                                        // Entity not in store: reduce the residual to the
                                        // unknown entity and the attributes not yet resolved
                                        // (the current attr plus the remaining ones).
                                        let reduced_attrs = nonempty::NonEmpty {
                                            head: attr.clone(),
                                            tail: iter.cloned().collect(),
                                        };
                                        let entity_ty =
                                            Type::named_entity_reference(uid.entity_type().clone());
                                        return mk_residual(ResidualKind::ExtHasAttr {
                                            expr: Arc::new(Residual::Concrete {
                                                value: current_val.clone(),
                                                ty: entity_ty,
                                            }),
                                            attrs: reduced_attrs,
                                        });
                                    }
                                }
                            } else {
                                return mk_error();
                            };
                            let Some(next_val) = current_attrs.get(attr) else {
                                return mk_concrete(false.into());
                            };
                            current_val = next_val;
                        }
                        // All attributes are present
                        mk_concrete(true.into())
                    }
                    Residual::Partial { .. } => mk_residual(ResidualKind::ExtHasAttr {
                        expr: Arc::new(expr),
                        attrs: attrs.clone(),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::UnaryApp { op, arg } => {
                let arg = self.interpret(arg);
                match arg {
                    Residual::Concrete { value, .. } => {
                        if let Ok(v) = crate::evaluator::unary_app(*op, value, None) {
                            mk_concrete(v)
                        } else {
                            mk_error()
                        }
                    }
                    Residual::Partial { .. } => mk_residual(ResidualKind::UnaryApp {
                        op: *op,
                        arg: Arc::new(arg),
                    }),
                    Residual::Error(_) => mk_error(),
                }
            }
            ResidualKind::ExtensionFunctionApp { fn_name, args } => {
                let args: Vec<_> = args.iter().map(|a| self.interpret(a)).collect();
                // If the arguments are all concrete values, we proceed to evaluate the function call
                if args.iter().all(Residual::is_concrete) {
                    let vals : Vec<_> = args.into_iter().map(|a|{
                        #[expect(
                            clippy::unwrap_used,
                            reason = "`if` condition guarantees that all set elements are concrete, so `Value::try_from` cannot error"
                        )]
                        Value::try_from(a).unwrap()
                    }).collect();
                    // Attempt to look up the extension function and apply it
                    // Failed lookup or application errors both lead to
                    // `Residual::Error` of appropriate types
                    if let Ok(ext_fn) = self.extensions.func(fn_name) {
                        if let Ok(PartialValue::Value(value)) = ext_fn.call(&vals) {
                            return mk_concrete(normalize_ext_value(value));
                        }
                    }
                    mk_error()
                } else if args.iter().any(Residual::is_error) {
                    mk_error()
                } else {
                    mk_residual(ResidualKind::ExtensionFunctionApp {
                        fn_name: fn_name.clone(),
                        args: Arc::new(args),
                    })
                }
            }
            ResidualKind::Set(es) => {
                let es: Vec<_> = es.iter().map(|e| self.interpret(e)).collect();
                if es.iter().all(Residual::is_concrete) {
                    let vals = es.into_iter().map(|a|{
                        #[expect(
                            clippy::unwrap_used,
                            reason = "`if` condition guarantees that all set elements are concrete, so `Value::try_from` cannot error"
                        )]
                        Value::try_from(a).unwrap()
                    });
                    mk_concrete(Value::set(vals, None))
                } else if es.iter().any(Residual::is_error) {
                    mk_error()
                } else {
                    mk_residual(ResidualKind::Set(Arc::new(es)))
                }
            }
            ResidualKind::Record(m) => {
                let record: BTreeMap<_, _> = m
                    .iter()
                    .map(|(a, e)| (a.clone(), self.interpret(e)))
                    .collect();
                if record.iter().all(|(_, r)| r.is_concrete()) {
                    let m = record
                        .into_iter()
                        .map(|(a, r)| {
                            #[expect(
                                clippy::unwrap_used,
                                reason = "`if` condition guarantees that all attribute values are concrete, so `Value::try_from` cannot error"
                            )]
                            (a, Value::try_from(r).unwrap())
                        });
                    mk_concrete(Value::record(m, None))
                } else if record.iter().any(|(_, r)| r.is_error()) {
                    mk_error()
                } else {
                    mk_residual(ResidualKind::Record(Arc::new(record)))
                }
            }
            // Phase 6.5 (D-80, reqs 7.1/7.2/7.3/7.4). `.all`/`.any`.
            #[cfg(feature = "anyall")]
            ResidualKind::All { expr, pred } => {
                let recv = self.interpret(expr);
                match &recv {
                    // Receiver error propagates (matches the concrete evaluator,
                    // which fails on a receiver error).
                    Residual::Error(_) => mk_error(),
                    // Residual receiver: keep the whole quantifier node (7.2).
                    // The receiver is already partially evaluated.
                    Residual::Partial { .. } => mk_residual(ResidualKind::All {
                        expr: Arc::new(recv),
                        pred: pred.clone(),
                    }),
                    // Concrete receiver: refine the quantifier per element
                    // (req 7.3 / D-80 full / D-92). Non-set receiver => error.
                    // For each element, instantiate the predicate and interpret
                    // it:
                    //   * a concrete `true`  => the element is discharged (drop);
                    //   * a concrete `false` => pin `decided_false` (the final
                    //     value is `false` unless a pending element errors);
                    //   * a concrete error / non-bool => the whole node is the
                    //     `QuantifierError` image (`mk_error`), immediately
                    //     (reqs 2.5/2.6);
                    //   * a residual => keep it in `pending`.
                    // There is NO short-circuit on a `false` element: every
                    // element is interpreted, because a later element may still
                    // error and that error outranks the pinned `false` (2.8).
                    // After the loop: empty `pending` collapses to
                    // `Concrete(!decided_false)` (the fold completed — all-true
                    // => `true`, some-false => `false`); otherwise emit a refined
                    // `AllPartial` carrying the pinned `false` and the undecided
                    // elements (7.3). This supersedes the Phase-6.5 conservative
                    // "keep the whole node" behaviour.
                    Residual::Concrete { value, .. } => {
                        let set = match value.get_as_set() {
                            Ok(set) => set,
                            Err(_) => return mk_error(),
                        };
                        let mut decided_false = false;
                        let mut pending: Vec<Residual> = Vec::new();
                        for elem in set.iter() {
                            let inst = pred.instantiate(elem);
                            let elem_residual =
                                self.interpret(&Residual::from_untyped_expr(&inst, r.ty()));
                            match &elem_residual {
                                Residual::Error(_) => return mk_error(),
                                Residual::Concrete { value, .. } => match value.get_as_bool() {
                                    Ok(true) => {}             // discharged
                                    Ok(false) => decided_false = true,
                                    // A non-boolean per-element result is a
                                    // QuantifierError, matching Lean/Rust (D-16).
                                    Err(_) => return mk_error(),
                                },
                                Residual::Partial { .. } => pending.push(elem_residual),
                            }
                        }
                        if pending.is_empty() {
                            mk_concrete((!decided_false).into())
                        } else {
                            mk_residual(ResidualKind::AllPartial {
                                decided_false,
                                pending: Arc::new(pending),
                            })
                        }
                    }
                }
            }
            // A refined quantifier (D-92) is re-interpreted idempotently: each
            // still-pending element predicate is interpreted again (an unknown
            // it reads may now be known), applying the same per-element fold —
            // newly-`true` elements drop, a newly-`false` element pins
            // `decided_false`, an element error collapses the whole node to the
            // `QuantifierError` image, and still-residual elements stay pending.
            // No short-circuit (2.8). Empty `pending` collapses to
            // `Concrete(!decided_false)`. This keeps a second TPE pass sound and
            // monotone (a pending element can only become decided, never
            // un-decided).
            #[cfg(feature = "anyall")]
            ResidualKind::AllPartial {
                decided_false,
                pending,
            } => {
                let mut decided_false = *decided_false;
                let mut new_pending: Vec<Residual> = Vec::new();
                for elem in pending.as_ref() {
                    let elem_residual = self.interpret(elem);
                    match &elem_residual {
                        Residual::Error(_) => return mk_error(),
                        Residual::Concrete { value, .. } => match value.get_as_bool() {
                            Ok(true) => {}
                            Ok(false) => decided_false = true,
                            Err(_) => return mk_error(),
                        },
                        Residual::Partial { .. } => new_pending.push(elem_residual),
                    }
                }
                if new_pending.is_empty() {
                    mk_concrete((!decided_false).into())
                } else {
                    mk_residual(ResidualKind::AllPartial {
                        decided_false,
                        pending: Arc::new(new_pending),
                    })
                }
            }
        }
    }
}

/// If the value is an extension value, rebuild the
/// [`RepresentableExtensionValue`] so that the stored `func`/`args` match the
/// canonical form given by [`ExtensionValue::canonical_repr`]. This ensures TPE
/// residuals are deterministic regardless of which constructor originally
/// created the value.
fn normalize_ext_value(value: Value) -> Value {
    normalize_ext_value_inner(&value).unwrap_or(value)
}

/// Returns `Some(normalized)` if the value needed normalization, `None` if it was already fine.
fn normalize_ext_value_inner(value: &Value) -> Option<Value> {
    match &value.value {
        ValueKind::Lit(_) => None,
        ValueKind::ExtensionValue(ev) => Some(Value {
            value: ValueKind::ExtensionValue(Arc::new(ast::RepresentableExtensionValue::new_lazy(
                ev.value.clone(),
            ))),
            loc: value.loc.clone(),
        }),
        ValueKind::Set(s) if s.fast.is_some() => {
            // due to invariant on set, this means all elements are literals, hence nothing to norm
            None
        }

        // The Set and Record normalization attempt to avoid cloning by scanning whether
        // normalization is needed. Cloning to get the normalization only happens when it is
        // actually required.
        ValueKind::Set(s) => {
            // Find the first element that needs normalization or return None.
            let (idx, normalized) = s
                .iter()
                .enumerate()
                .find_map(|(i, x)| normalize_ext_value_inner(x).map(|n| (i, n)))?;
            // Clone elements before `idx` as-is, insert the normalized one,
            // then normalize the rest.
            let vals: Vec<Value> = s
                .iter()
                .take(idx)
                .cloned()
                .chain(std::iter::once(normalized))
                .chain(
                    s.iter()
                        .skip(idx + 1)
                        .map(|v| normalize_ext_value_inner(v).unwrap_or_else(|| v.clone())),
                )
                .collect();
            Some(Value {
                value: ValueKind::Set(Set::new(vals)),
                loc: value.loc.clone(),
            })
        }
        ValueKind::Record(r) => {
            let mut iter = r.iter().enumerate();
            let (idx, key, normalized) = loop {
                let (i, (k, v)) = iter.next()?;
                if let Some(n) = normalize_ext_value_inner(v) {
                    break (i, k.clone(), n);
                }
            };
            let map: BTreeMap<_, _> = r
                .iter()
                .take(idx)
                .map(|(k, v)| (k.clone(), v.clone()))
                .chain(std::iter::once((key, normalized)))
                .chain(r.iter().skip(idx + 1).map(|(k, v)| {
                    (
                        k.clone(),
                        normalize_ext_value_inner(v).unwrap_or_else(|| v.clone()),
                    )
                }))
                .collect();
            Some(Value {
                value: ValueKind::Record(Arc::new(map)),
                loc: value.loc.clone(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use smol_str::SmolStr;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use crate::ast::{Expr, SlotEnv};
    use crate::tpe::err::ExprToResidualError;
    use crate::tpe::test_utils::{parse_partial_euid, parse_typed_expr};
    use crate::validator::types::Type;
    use crate::validator::ValidatorSchema;
    use crate::{
        ast::{Context, ExprBuilder, Value, Var},
        expr_builder::ExprBuilder as _,
        extensions::Extensions,
    };
    use insta::assert_snapshot;

    use crate::{
        ast, tpe::entities::PartialEntities, tpe::request::PartialRequest, tpe::residual::Residual,
    };

    use super::Evaluator;

    #[track_caller]
    fn parse_schema(src: &str) -> ValidatorSchema {
        ValidatorSchema::from_cedarschema_str(src, &Extensions::all_available())
            .unwrap()
            .0
    }

    fn schema() -> ValidatorSchema {
        parse_schema(
            r#"
            entity User in Organization { foo: Bool, str: String, num: Long, period: __cedar::duration, set: Set<String> } tags String;
            entity Organization;
            entity Document in Organization;
            action get appliesTo { principal: [User], resource: [Document] };"#,
        )
    }

    fn concrete_user_req() -> PartialRequest {
        PartialRequest::new(
            parse_partial_euid(r#"User::"foo""#),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid("Document"),
            None,
            &schema(),
        )
        .unwrap()
    }

    #[track_caller]
    fn interpret_typed_str_to_str(
        evaluator: &Evaluator<'_>,
        expr_str: &str,
        schema: &ValidatorSchema,
    ) -> String {
        let expr = parse_typed_expr(expr_str, evaluator.request, schema, &SlotEnv::new());
        interpret_expr_to_str(evaluator, &expr)
    }

    #[track_caller]
    fn interpret_expr_to_str(evaluator: &Evaluator<'_>, expr: &Expr<Option<Type>>) -> String {
        let evaluated: ast::Expr = evaluator.interpret_expr(&expr).unwrap().try_into().unwrap();
        evaluated.to_string()
    }

    impl Evaluator<'_> {
        /// Interpret a typed expression by converting to a [`Residual`] with an empty slot environment.
        ///
        /// This is a test-only utility because other callers should generally
        /// be interpreting a residual with slots bounds appropriately by
        /// `policy_residual_map` or else explicitly bindings slots (with an
        /// empty environment or otherwise).
        fn interpret_expr(&self, e: &Expr<Option<Type>>) -> Result<Residual, ExprToResidualError> {
            Ok(self.interpret(&Residual::try_from_typed_expr(e, &SlotEnv::new())?))
        }
    }

    #[test]
    fn test_var() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        // principal -> User::""
        assert_snapshot!(
            interpret_typed_str_to_str("principal"),
            @r#"User::"foo""#
        );
        // resource -> resource because its eid is unknown
        assert_snapshot!(
            interpret_typed_str_to_str("resource"),
            @"resource"
        );
        // action is always known
        assert_snapshot!(
            interpret_typed_str_to_str("action"),
            @r#"Action::"get""#
        );
        // context is always unknown
        assert_snapshot!(
            interpret_typed_str_to_str("context"),
            @"context"
        );
    }

    #[test]
    fn test_known_context() {
        let schema = parse_schema(
            r#"entity E; action a appliesTo {principal: E, resource: E, context: {l: Long}};"#,
        );
        let eval = Evaluator {
            request: &PartialRequest::new(
                parse_partial_euid(r#"E"#),
                r#"Action::"a""#.parse().unwrap(),
                parse_partial_euid("E"),
                Some(Arc::new(BTreeMap::from([(
                    "l".parse().unwrap(),
                    Value::from(0),
                )]))),
                &schema,
            )
            .unwrap(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str("context"),
            @"{l: 0}"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("context.l"),
            @"0"
        );
    }

    #[track_caller]
    fn builder() -> ExprBuilder<Option<Type>> {
        ExprBuilder::with_data(Some(Type::Never))
    }

    #[test]
    fn test_and() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        // Note: The test expressions are in the same order as the match statements

        // "false && <any>" => "false"
        assert_snapshot!(
            // Note: principal is concrete, and thus can the residual be simplified.
            interpret_typed_str_to_str("principal != principal && principal.foo"),
            @r#"false"#
        );
        // "true && <any>" => "<any>"
        assert_snapshot!(
            interpret_typed_str_to_str("principal == principal && principal.foo"),
            @r#"User::"foo".foo"#
        );
        // "<error> && <residual>" => "<error>"
        assert_snapshot!(
            interpret_typed_str_to_str("(9223372036854775807 * 2 == 0) && principal.foo"),
            @r#"error()"#
        );
        // "<residual> && true" => "<residual>"
        assert_snapshot!(
            interpret_typed_str_to_str("principal.foo && true"),
            @r#"User::"foo".foo"#
        );
        // "<error-free> && false" => "false"
        assert_snapshot!(
            interpret_typed_str_to_str("resource == resource && 41 == 42"),
            @r#"false"#
        );
        // note: resource is unknown, and we haven't (yet) implemented a simplifying algorithm for this,
        // so it yields an error-free residual, hence the previous test makes sense
        assert_snapshot!(
            interpret_typed_str_to_str("resource == resource"),
            @r#"resource == resource"#
        );
        // "<non-error-free> && false" cannot be fully simplified
        assert_snapshot!(
            interpret_typed_str_to_str("principal.num + 1 == 100 && 41 == 42"),
            @r#"((User::"foo".num + 1) == 100) && false"#
        );
        // "<residual> && <nonbool>" => "<residual> && <error>"
        assert_snapshot!(
            interpret_expr_to_str(&eval, &builder().and(
                builder().get_attr(builder().var(Var::Principal), "foo".into()),
                builder().val(42),
            )),
            @r#"User::"foo".foo && error()"#
        );
        // The "<residual> && <residual>" case cannot be simplified
        assert_snapshot!(
            interpret_typed_str_to_str("principal.foo && principal.num == 100"),
            @r#"User::"foo".foo && (User::"foo".num == 100)"#
        );
        // "<residual> && <error>" cannot be simplified
        assert_snapshot!(
            interpret_typed_str_to_str("principal.foo && (9223372036854775807 * 2 == 0)"),
            @r#"User::"foo".foo && error()"#
        );
        // "<error> && <any>" => "<error>"
        assert_snapshot!(
            interpret_typed_str_to_str("(9223372036854775807 * 2 == 0) && false"),
            @"error()"
        );
    }

    #[test]
    fn test_or() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        // Note: The test expressions are in the same order as the match statements

        // "true || <any>" => "true"
        assert_snapshot!(
            // Note: principal is concrete, and thus can the residual be simplified.
            interpret_typed_str_to_str("principal == principal || principal.foo"),
            @r#"true"#
        );
        // "false || <any>" => "<any>"
        assert_snapshot!(
            interpret_typed_str_to_str("principal != principal || principal.foo"),
            @r#"User::"foo".foo"#
        );
        // "<error> || <residual>" => "<error>"
        assert_snapshot!(
            interpret_typed_str_to_str("(9223372036854775807 * 2 == 0) || principal.foo"),
            @r#"error()"#
        );
        // "<residual> || false" => "<residual>"
        assert_snapshot!(
            interpret_typed_str_to_str("principal.foo || false"),
            @r#"User::"foo".foo"#
        );
        // "<error-free> || true" => "true"
        assert_snapshot!(
            interpret_typed_str_to_str("resource == resource || 42 == 42"),
            @r#"true"#
        );
        // note: resource is unknown, and we haven't (yet) implemented a simplifying algorithm for this,
        // so it yields an error-free residual, hence the previous test makes sense
        assert_snapshot!(
            interpret_typed_str_to_str("resource == resource"),
            @r#"resource == resource"#
        );
        // "<non-error-free> || true" cannot be fully simplified
        assert_snapshot!(
            interpret_typed_str_to_str("principal.num + 1 == 100 || 42 == 42"),
            @r#"((User::"foo".num + 1) == 100) || true"#
        );
        // "<residual> || <nonbool>" => "<residual> || <error>"
        assert_snapshot!(
            interpret_expr_to_str(&eval, &builder().or(
                builder().get_attr(builder().var(Var::Principal), "foo".into()),
                builder().val(42),
            )),
            @r#"User::"foo".foo || error()"#
        );
        // The "<residual> || <residual>" case cannot be simplified
        assert_snapshot!(
            interpret_typed_str_to_str("principal.foo || principal.num == 100"),
            @r#"User::"foo".foo || (User::"foo".num == 100)"#
        );
        // "<residual> || <error>" cannot be simplified
        assert_snapshot!(
            interpret_typed_str_to_str("principal.foo || (9223372036854775807 * 2 == 0)"),
            @r#"User::"foo".foo || error()"#
        );
        // "<error> || <any>" => "<error>"
        assert_snapshot!(
            interpret_typed_str_to_str("(9223372036854775807 * 2 == 0) || true"),
            @"error()"
        );
    }

    #[test]
    fn test_ite() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        assert_snapshot!(
            interpret_typed_str_to_str("if true then resource else 2"),
            @"resource"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("if false then resource else 2"),
            @"2"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"if (resource == User::"alice") then Document::"A" else Document::"B""#),
            @r#"if resource == User::"alice" then Document::"B" else Document::"B""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(&r#"if (9223372036854775807 * 2) == 0 then resource else Document::"A""#),
            @"error()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("if true then (9223372036854775807 * 2) else 0"),
            @"error()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("if false then (9223372036854775807 * 2) else 0"),
            @"0"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("if true then 0 else (9223372036854775807 * 2)"),
            @"0"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("if false then 0 else (9223372036854775807 * 2)"),
            @"error()"
        );
    }

    #[test]
    fn test_is() {
        let schema = parse_schema(
            r#"entity E, User { baz : E }; action a appliesTo {principal: User, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"a""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str("resource is User"),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("resource is E"),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("principal is User"),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("principal is E"),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("principal.baz is E"),
            @"principal.baz is E"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("principal.baz is Document"),
            @"principal.baz is Document"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("User::\"alice\" is User"),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("User::\"alice\" is E"),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str("(if (9223372036854775807 * 2 == 0) then User::\"alice\" else principal) is E"),
            @"error()"
        );
    }

    #[test]
    fn test_like() {
        let schema = parse_schema(
            r#"entity E { s : String }; action a appliesTo {principal: E, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("E"),
            r#"Action::"a""#.parse().unwrap(),
            parse_partial_euid("E"),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str(r#""aaa" like "a*""#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#""aaa" like "b*""#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.s like "*""#),
            @r#"principal.s like "*""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.s like "b*""#),
            @r#"principal.s like "b*""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if (9223372036854775807 * 2 == 0) then "a" else "b") like "b*""#),
            @"error()"
        );
    }

    #[test]
    fn test_negation() {
        let schema = parse_schema(
            r#"entity E { l : Long , b: Bool }; action a appliesTo {principal: E, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("E"),
            r#"Action::"a""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"-(42)"#),
            @"(-42)"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"-(-9223372036854775808)"#),
            @"error()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"-(principal.l)"#),
            @"-(principal.l)"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"--(principal.l)"#),
            @"-(-(principal.l))"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"!(false)"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"!(principal.b)"#),
            @"!principal.b"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"!!(principal.b)"#),
            @"!(!principal.b)"
        );
    }

    #[test]
    fn test_is_empty() {
        let schema = parse_schema(
            r#"entity E { s: Set<Long> }; action a appliesTo {principal: E, resource: E, context: {s0: Set<Long>, s1: Set<Long>}};"#,
        );
        let Ok(Context::Value(context)) =
            Context::from_json_value(serde_json::json!({ "s0": [], "s1": [0] }))
        else {
            panic!("expected concrete context")
        };
        let req = PartialRequest::new(
            parse_partial_euid(r#"E::"foo""#),
            r#"Action::"a""#.parse().unwrap(),
            parse_partial_euid("E"),
            Some(context),
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[1].isEmpty()"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context.s0.isEmpty()"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context.s1.isEmpty()"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.s.isEmpty()"#),
            @"resource.s.isEmpty()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if (9223372036854775807 * 2 == 0) then [1] else [2]).isEmpty()"#),
            @"error()"
        );
    }

    #[test]
    fn test_get_attr_simple() {
        let schema = parse_schema(
            r#"entity E { s: String }; action get appliesTo {principal: E, resource: E};"#,
        );
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": { "s": "bar" },
                },
                {
                    "uid": { "type": "E", "id": "e" },
                },
            ]),
            &schema,
        )
        .unwrap();
        let req = PartialRequest::new(
            parse_partial_euid(r#"E::"""#),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid("E"),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.s"#),
            @r#""bar""#
        );
        // When LHS is unknown, the entire expression is
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.s"#),
            @"resource.s"
        );
        // When LHS is not in the entities, the entire expression is unknown
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"f".s"#),
            @r#"E::"f".s"#
        );
        // When LHS is in the entities, but its attributes are `None`, the
        // entire expression is unknown
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"e".s"#),
            @r#"E::"e".s"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if (9223372036854775807 * 2 == 0) then E::"alice" else E::"bob").s"#),
            @"error()"
        );
    }

    #[test]
    fn test_get_attr_chained() {
        let schema = parse_schema(
            r#"entity E { s: String, e: E }; action get appliesTo {principal: E, resource: E};"#,
        );
        // `E::"loop".e` points back at itself, so chained access through it
        // always terminates at `E::"loop"`.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": {
                        "s": "bar",
                        "e": { "__entity": { "type": "E", "id": "x" } },
                    },
                },
                {
                    "uid": { "type": "E", "id": "loop" },
                    "attrs": {
                        "s": "loop_str",
                        "e": { "__entity": { "type": "E", "id": "loop" } },
                    },
                },
            ]),
            &schema,
        )
        .unwrap();
        let req = PartialRequest::new(
            parse_partial_euid(r#"E::"""#),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid("E"),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.e.s"#),
            @r#"E::"x".s"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.e.e.e.s"#),
            @r#"E::"x".e.e.s"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.e.s"#),
            @"resource.e.s"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"loop".e"#),
            @r#"E::"loop""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"loop".e.e.e.e"#),
            @r#"E::"loop""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"loop".e.e.e.e.s"#),
            @r#""loop_str""#
        );
    }

    #[test]
    fn test_has_attr() {
        let schema = parse_schema(
            r#"entity E { s: String };
            entity Hat { size: Long, origin: Country };
            entity Country { name: String };
            entity User { s: String, hat: Hat };
            action get appliesTo {principal: E, resource: User};"#,
        );
        // `User::""` has known attributes while `E::"e"` omits `attrs`, marking
        // them unknown.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "User", "id": "" },
                    "attrs": { "s": "bar", "hat": { "__entity": { "type": "Hat", "id": "cap" } }  },
                },
                {
                    "uid": { "type": "Hat", "id": "cap" },
                    "attrs": { "size": 10, "origin": { "__entity": { "type": "Country", "id": "italy" } } }
                },
                {
                    "uid": { "type": "E", "id": "e" },
                },
            ]),
            &schema,
        )
        .unwrap();
        let req = PartialRequest::new(
            parse_partial_euid("E"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"User::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has s"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has other"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal has s"#),
            @"principal has s"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal has other"#),
            @"principal has other"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has tomato.country"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has hat.origin"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has hat.origin.name"#),
            @r#"Country::"italy" has name"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"f" has s"#),
            @r#"E::"f" has s"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"e" has s"#),
            @r#"E::"e" has s"#
        );

        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: 0} has s"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: 0} has t"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: {t: {u: 0}}, r: 1} has s.t.u"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: {t: {u: 0}}, r: 1} has r"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: {t: {u: 0}}, r: 1} has s.u.t"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if (9223372036854775807 * 2 == 0) then E::"alice" else E::"bob") has s"#),
            @"error()"
        );
    }

    /// Test extended has attr in TPE specifically for entity-chain traversal
    /// and residual production when entities are partially known.
    #[test]
    fn test_ext_has_attr_entity_chain() {
        let schema = parse_schema(
            r#"entity E { s: String };
            entity Hat { size: Long, origin: Country };
            entity Country { name: String };
            entity User { s: String, hat: Hat };
            action get appliesTo {principal: E, resource: User};"#,
        );
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "User", "id": "" },
                    "attrs": { "s": "bar", "hat": { "__entity": { "type": "Hat", "id": "cap" } } },
                },
                {
                    "uid": { "type": "Hat", "id": "cap" },
                    "attrs": { "size": 10, "origin": { "__entity": { "type": "Country", "id": "italy" } } }
                },
                {
                    "uid": { "type": "Country", "id": "italy" },
                    "attrs": { "name": "Italy" }
                },
            ]),
            &schema,
        )
        .unwrap();
        let req = PartialRequest::new(
            parse_partial_euid("E"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"User::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has hat.origin.name"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has hat.size"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has hat.origin.missing"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has hat.missing.name"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource has missing.anything"#),
            @"false"
        );
        assert_snapshot!(
          interpret_typed_str_to_str(r#"Hat::"other" has origin.name"#),
          @r#"Hat::"other" has origin.name"#
        );
    }

    /// This mirrors the tests in `UnitTest.TPE.ExtHasAttr` from cedar-lean.
    #[test]
    fn test_ext_has_attr_residual_reduction() {
        let schema = parse_schema(
            r#"
            entity Leaf { value: String };
            entity Middle { info: { "tag": String }, next: Leaf };
            entity Root { child: Middle, data: { "inner": Middle } };
            entity User { profile: { "address": { "city": String } }, manager: Root };
            entity Document;
            action Do appliesTo {
                principal: [User],
                resource: [Document],
                context: {
                    "ref": Root,
                    "nested": { "deep": { "leaf": String } },
                    "wrap": { "box": { "target": Middle } }
                }
            };
            "#,
        );

        // Entities:
        //   User::"alice"  → { profile: {address: {city: "Seattle"}}, manager: Root::"r1" }
        //   Root::"r1"     → { child: Middle::"m1", data: {inner: Middle::"m2"} }
        //   Middle::"m1"   → { info: {tag: "hello"}, next: Leaf::"l1" }
        //   Leaf::"l1"     → { value: "world" }
        //   Middle::"m2"   — NOT in store (unknown)
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "User", "id": "alice" },
                    "attrs": {
                        "profile": { "address": { "city": "Seattle" } },
                        "manager": { "__entity": { "type": "Root", "id": "r1" } }
                    }
                },
                {
                    "uid": { "type": "Root", "id": "r1" },
                    "attrs": {
                        "child": { "__entity": { "type": "Middle", "id": "m1" } },
                        "data": { "inner": { "__entity": { "type": "Middle", "id": "m2" } } }
                    }
                },
                {
                    "uid": { "type": "Middle", "id": "m1" },
                    "attrs": {
                        "info": { "tag": "hello" },
                        "next": { "__entity": { "type": "Leaf", "id": "l1" } }
                    }
                },
                {
                    "uid": { "type": "Leaf", "id": "l1" },
                    "attrs": { "value": "world" }
                },
            ]),
            &schema,
        )
        .unwrap();

        // Request: principal=User::"alice", action=Action::"Do", resource=Document (unknown eid)
        // context = { "ref": Root::"r1", "nested": {"deep": {"leaf": "yes"}},
        //             "wrap": {"box": {"target": Middle::"m1"}} }
        let req = PartialRequest::new(
            parse_partial_euid(r#"User::"alice""#),
            r#"Action::"Do""#.parse().unwrap(),
            parse_partial_euid("Document"),
            Some(Arc::new(BTreeMap::from([
                (
                    "ref".parse().unwrap(),
                    Value::from(ast::EntityUID::with_eid_and_type("Root", "r1").unwrap()),
                ),
                (
                    "nested".parse().unwrap(),
                    Value::record(
                        [(
                            SmolStr::from("deep"),
                            Value::record(
                                [(smol_str::SmolStr::from("leaf"), Value::from("yes"))],
                                None,
                            ),
                        )],
                        None,
                    ),
                ),
                (
                    "wrap".parse().unwrap(),
                    Value::record(
                        [(
                            SmolStr::from("box"),
                            Value::record(
                                [(
                                    SmolStr::from("target"),
                                    Value::from(
                                        ast::EntityUID::with_eid_and_type("Middle", "m1").unwrap(),
                                    ),
                                )],
                                None,
                            ),
                        )],
                        None,
                    ),
                ),
            ]))),
            &schema,
        )
        .unwrap();

        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        // --- Basic tests ---

        // context has nested.deep.leaf → true (all records, fully known)
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has nested.deep.leaf"#),
            @"true"
        );

        // context has nested.missing.leaf → false (no "missing" in context.nested)
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has nested.missing.leaf"#),
            @"false"
        );

        // context has ref → true (single attr, context has "ref")
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has ref"#),
            @"true"
        );

        // --- Residual reduction tests ---

        // principal.manager has data.inner.info → Middle::"m2" has info
        // Because: Root::"r1".data.inner = Middle::"m2" (not in store), can't check "info"
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.manager has data.inner.info"#),
            @r#"Middle::"m2" has info"#
        );

        // principal.manager has data.inner.next → Middle::"m2" has next
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.manager has data.inner.next"#),
            @r#"Middle::"m2" has next"#
        );

        // principal.manager has data.inner.next.value → Middle::"m2" has next.value
        // Because: Middle::"m2" is unknown, remaining chain stays as residual
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.manager has data.inner.next.value"#),
            @r#"Middle::"m2" has next.value"#
        );

        // principal has manager.child.next.value → true
        // Because: all entities in chain are known, Leaf::"l1" has "value"
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal has manager.child.next.value"#),
            @"true"
        );

        // context has wrap.box.target.next.value → true
        // Because: all entities resolved, Leaf::"l1" has "value"
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has wrap.box.target.next.value"#),
            @"true"
        );

        // context has ref.child.next.value → true
        // Because: Root::"r1" → Middle::"m1" → Leaf::"l1", all known, "value" present
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has ref.child.next.value"#),
            @"true"
        );

        // --- Nesting pattern tests ---

        // context has ref.data.inner → true (R-E-R: context→ref(entity)→data(record)→inner exists)
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has ref.data.inner"#),
            @"true"
        );

        // context has wrap.box.target.info → true (R-R-E: records→records→entity→check)
        // context.wrap.box.target = Middle::"m1", which has "info"
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context has wrap.box.target.info"#),
            @"true"
        );

        // principal has manager.child.info.tag → true (E-E-R: entity→entity→record→check)
        // User::"alice".manager = Root::"r1", .child = Middle::"m1", .info = {tag: "hello"}, has "tag"
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal has manager.child.info.tag"#),
            @"true"
        );
    }

    #[test]
    fn test_has_get_optional_attr() {
        let schema = parse_schema(
            r#"entity E { s?: String }; entity User { b: Bool }; action get appliesTo {principal: E, resource: User, context: {x: String}};"#,
        );
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "0" },
                    "attrs": { "s": "foo" },
                },
                {
                    "uid": { "type": "E", "id": "1" },
                    "attrs": { },
                },
                {
                    "uid": { "type": "E", "id": "2" },
                },
            ]),
            &schema,
        )
        .unwrap();
        let req = PartialRequest::new(
            parse_partial_euid("E"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid("User"),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal has s && principal.s == context.x"#),
            @"(principal has s) && (principal.s == context.x)"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"0" has s && E::"0".s == context.x"#),
            @r#""foo" == context.x"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(resource.b && E::"0" has s) && E::"0".s == context.x"#),
            @r#"resource.b && ("foo" == context.x)"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"1" has s && E::"1".s == context.x"#),
            @"false"
        );
        // Residual `resource.b` means we can't eliminate `error` expression even though it's unreachable
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(resource.b && E::"1" has s) && E::"1".s == context.x"#),
            @"resource.b && false && error()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"2" has s && E::"2".s == context.x"#),
            @r#"(E::"2" has s) && (E::"2".s == context.x)"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"3" has s && E::"3".s == context.x"#),
            @r#"(E::"3" has s) && (E::"3".s == context.x)"#
        );
    }

    #[test]
    fn test_set() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };

        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[resource]"#),
            @"[resource]"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[principal]"#),
            @r#"[User::"foo"]"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[principal, User::"alice"]"#),
            @r#"[User::"alice", User::"foo"]"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[0, -(-9223372036854775808)]"#),
            @"error()"
        );
    }

    #[test]
    fn test_record() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: resource}"#),
            @"{s: resource}"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: principal}"#),
            @r#"{s: User::"foo"}"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: -(-9223372036854775808), "": resource}"#),
            @"error()"
        );
    }

    #[test]
    fn test_get_attr_record() {
        let schema = parse_schema(
            r#"entity E { rec: { s: String, inner: { n: Long } } }; entity User; action get appliesTo {principal: User, resource: E, context: {c: {n: Long}}};"#,
        );

        // Entity `E::""` has a `rec` attribute that is a nested record.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": { "rec": { "s": "bar", "inner": { "n": 1 } } },
                },
                {
                    "uid": { "type": "E", "id": "unk_attrs" },
                }
            ]),
            &schema,
        )
        .unwrap();

        // A context record with a nested record field.
        let Ok(Context::Value(context)) =
            Context::from_json_value(serde_json::json!({ "c": { "n": 2 } }))
        else {
            panic!("expected concrete context")
        };

        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            Some(context),
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        // Nested records read from an entity attribute.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.rec.s"#),
            @r#""bar""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.rec.inner.n"#),
            @"1"
        );

        // Nested records read from the context.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context.c"#),
            @"{n: 2}"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context.c.n"#),
            @"2"
        );

        // Attribute access on record literals.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{s: "lit"}.s"#),
            @r#""lit""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{inner: {n: 3}}.inner.n"#),
            @"3"
        );

        assert_snapshot!(
            interpret_typed_str_to_str(r#"{e: E::"unk"}.e.rec"#),
            @r#"E::"unk".rec"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{e: E::"unk_attrs"}.e.rec"#),
            @r#"E::"unk_attrs".rec"#
        );

        // Residuals inside record literals prevent reduction to preserve error semantics
        // cases to preserve semantics, should often be possible.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{e: if principal == User::"alice" then 9223372036854775807*2 else 0, l: 0}.l"#),
            @r#"{e: if principal == User::"alice" then error() else 0, l: 0}.l"#
        );
        // But it should be possible in other cases
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{p: principal}.p"#),
            @"{p: principal}.p"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"{p: principal, l: 0}.l"#),
            @"{l: 0, p: principal}.l"
        );
    }

    #[test]
    fn test_get_attr_intermediate_entities_emulate_unknown_attrs() {
        let schema = parse_schema(
            r#"
            entity E0 { metadata: { m0: M0, m1: M1 } };
            entity M0 { data: Long };
            entity M1 { data: Long };
            entity User;
            action get appliesTo { principal: User, resource: E0 };"#,
        );

        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E0", "id": "e" },
                    "attrs": {
                        "metadata": {
                            "m0": { "__entity": { "type": "M0", "id": "m0" } },
                            "m1": { "__entity": { "type": "M1", "id": "m1" } },
                        }
                    },
                },
                {
                    "uid": { "type": "M0", "id": "m0" },
                    "attrs": { "data": 42 },
                },
                {
                    "uid": { "type": "M1", "id": "m1" },
                },
            ]),
            &schema,
        )
        .unwrap();

        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E0::"e""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        // The referenced entity uids are always known.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.metadata.m0"#),
            @r#"M0::"m0""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.metadata.m1"#),
            @r#"M1::"m1""#
        );

        // `m0`'s attributes are known, so `data` reduces to a concrete value.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.metadata.m0.data"#),
            @"42"
        );
        // `m1`'s attributes are unknown, so `data` stays a residual: this is how
        // the intermediate entity emulates an unknown attribute.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.metadata.m1.data"#),
            @r#"M1::"m1".data"#
        );
        // Mixing a known and an unknown attribute leaves a partial residual.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.metadata.m0.data + resource.metadata.m1.data"#),
            @r#"42 + M1::"m1".data"#
        );
    }

    #[test]
    fn test_call() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        assert_snapshot!(
            interpret_typed_str_to_str(r#"ip("192.168.0.1").isInRange(ip("192.168.0.0/16"))"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if principal == User::"alice" then ip("127.0.0.1") else ip("192.168.0.1")).isInRange(ip("192.168.0.0/16"))"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if (9223372036854775807 * 2 == 0) then ip("127.0.0.1") else ip("192.168.0.1")).isInRange(ip("192.168.0.0/16"))"#),
            @"error()"
        );
    }

    #[test]
    fn test_binary_app_eq() {
        let schema = parse_schema(
            r#"entity E; entity User; action get appliesTo {principal: User, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource == E::"""#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource == User::"""#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal == User::"""#),
            @r#"principal == User::"""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal == E::"""#),
            @r#"principal == E::"""#
        );

        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal == (if (9223372036854775807 * 2 == 0) then User::"alice" else User::"bob")"#),
            @"error()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"(if (9223372036854775807 * 2 == 0) then User::"alice" else User::"bob") == principal"#),
            @"error()"
        );
    }

    #[test]
    fn test_binary_app_arith() {
        let schema = parse_schema(
            r#"entity E; entity User; action get appliesTo {principal: User, resource: E};"#,
        );
        let req = {
            let schema: &ValidatorSchema = &schema;
            PartialRequest::new(
                parse_partial_euid("User"),
                r#"Action::"get""#.parse().unwrap(),
                parse_partial_euid(r#"E::"""#),
                None,
                schema,
            )
            .unwrap()
        };
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"9223372036854775807 + 9223372036854775807"#),
            @"error()"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"1 + 1"#),
            @"2"
        );
    }

    #[test]
    fn test_binary_app_contains() {
        let schema = parse_schema(
            r#"entity E; entity User; action get appliesTo {principal: User, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[E::""].contains(resource)"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[User::""].contains(principal)"#),
            @r#"[User::""].contains(principal)"#
        );
    }

    #[test]
    fn test_binary_app_in() {
        let schema = parse_schema(
            r#"entity E in E; entity User in E; action get appliesTo {principal: User, resource: E, context: {e: E}};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"resource""#),
            None,
            &schema,
        )
        .unwrap();
        // `E::"e"` has known (empty) ancestors; `E::"child"` has a declared
        // parent `E::""`. The other referenced entities are absent, so their
        // ancestors are unknown and `in` stays a residual.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "no_parents" },
                    "parents": [],
                },
                {
                    "uid": { "type": "E", "id": "resource_child" },
                    "parents": [ { "type": "E", "id": "" } ],
                },
                {
                    "uid": { "type": "E", "id": "parents_none" },
                },
            ]),
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"no_parents" in resource"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"resource" in E::"resource""#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"resource_child" in resource"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"resource_child" in E::"not_parent""#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"resource_child" in [E::"not_parent", resource]"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"resource_child" in context.e"#),
            @r#"E::"resource_child" in context.e"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"resource_child" in [context.e]"#),
            @r#"E::"resource_child" in [context.e]"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"context.e in E::"resource""#),
            @r#"context.e in E::"resource""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in resource"#),
            @r#"E::"undefined" in E::"resource""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"parents_none" in resource"#),
            @r#"E::"parents_none" in E::"resource""#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in [resource]"#),
            @r#"E::"undefined" in [E::"resource"]"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"parents_none" in [resource]"#),
            @r#"E::"parents_none" in [E::"resource"]"#
        );

        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in E::"undefined""#),
            @r#"true"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in [E::"undefined"]"#),
            @r#"true"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in [E::"undefined", E::"other"]"#),
            @r#"true"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in [E::"other", E::"undefined"]"#),
            @r#"true"#
        );

        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in [E::"undefined", context.e]"#),
            @r#"E::"undefined" in [E::"undefined", context.e]"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined" in [E::"undefined", (if 9223372036854775807 * 2 == 0 then E::"undefined" else E::"undefined")]"#),
            @"error()"
        );
    }

    #[test]
    fn test_binary_app_in_empty_set() {
        let schema = parse_schema(
            r#"entity E in E; entity User in E; action get appliesTo {principal: User, resource: E, context: {empty: Set<E>}};"#,
        );
        let Ok(Context::Value(context)) =
            Context::from_json_value(serde_json::json!({ "empty": [] }))
        else {
            panic!("expected concrete context")
        };
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid("E"),
            Some(context),
            &schema,
        )
        .unwrap();
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "has_parents" },
                    "parents": [ { "type": "E", "id": "" } ],
                }
            ]),
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"has_parents" in context.empty"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"unknown" in context.empty"#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource in context.empty"#),
            @"resource in []"
        );
    }

    fn binary_app_tags_env() -> (ValidatorSchema, PartialEntities, PartialRequest) {
        let schema = parse_schema(
            r#"entity E; entity User tags String; action get appliesTo {principal: User, resource: E, context: {tag: String}};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"resource""#),
            None,
            &schema,
        )
        .unwrap();
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "User", "id": "some_tags" },
                    "tags": { "s": "bar" },
                },
                {
                    "uid": { "type": "User", "id": "none_tags" },
                },
                {
                    "uid": { "type": "E", "id": "none_tags" },
                },
                {
                    "uid": { "type": "E", "id": "empty_tags" },
                    "tags": {},
                },
            ]),
            &schema,
        )
        .unwrap();
        (schema, entities, req)
    }

    #[test]
    fn test_binary_app_has_tag() {
        let (schema, entities, req) = binary_app_tags_env();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        // Known tags: `hasTag` resolves to a concrete bool.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"some_tags".hasTag("s")"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"some_tags".hasTag("bogus")"#),
            @"false"
        );
        // Unknown entity uid, unknown tags, or absent entity: `hasTag` stays a residual.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.hasTag("s")"#),
            @r#"principal.hasTag("s")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"none_tags".hasTag("s")"#),
            @r#"User::"none_tags".hasTag("s")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"undefined".hasTag("s")"#),
            @r#"User::"undefined".hasTag("s")"#
        );
    }

    #[test]
    fn test_binary_app_get_tag() {
        let (schema, entities, req) = binary_app_tags_env();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        // Known tags: resolves to a concrete bool.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"some_tags".hasTag("s") && User::"some_tags".getTag("s") == "bar" "#),
            @"true"
        );
        // Known tags, missing key: `hasTag` is false and short-circuits.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"some_tags".hasTag("bogus") && User::"some_tags".getTag("bogus") == "bar" "#),
            @"false"
        );
        // Unknown cases
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.hasTag("s") && principal.getTag("s") == "bar" "#),
            @r#"principal.hasTag("s") && (principal.getTag("s") == "bar")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"none_tags".hasTag("s") && User::"none_tags".getTag("s") == "bar" "#),
            @r#"User::"none_tags".hasTag("s") && (User::"none_tags".getTag("s") == "bar")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"undefined".hasTag("s") && User::"undefined".getTag("s") == "bar" "#),
            @r#"User::"undefined".hasTag("s") && (User::"undefined".getTag("s") == "bar")"#
        );

        // `E` entities can't have tags, but `hasTag` is still well-typed. These could all reduce to
        // `false`, but only the explicit empty tag case does atm.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"empty_tags".hasTag("s") && E::"empty_tags".getTag("s") == "bar" "#),
            @"false"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"none_tags".hasTag("s") && E::"none_tags".getTag("s") == "bar" "#),
            @r#"E::"none_tags".hasTag("s")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined_tags".hasTag("s") && E::"undefined_tags".getTag("s") == "bar" "#),
            @r#"E::"undefined_tags".hasTag("s")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"undefined".hasTag("s") && E::"undefined".getTag("s") == "bar" "#),
            @r#"E::"undefined".hasTag("s")"#
        );

        // Residual on the left prevents eliminating `error()` expression even through it's unreachable
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"none_tags".hasTag("tag") && User::"none_tags".getTag("tag") == "foo" && User::"some_tags".hasTag("bogus") && User::"some_tags".getTag("bogus") == "bar" "#),
            @r#"User::"none_tags".hasTag("tag") && (User::"none_tags".getTag("tag") == "foo") && false && error()"#
        );
    }

    #[test]
    fn test_binary_app_tag_unknown_key() {
        let (schema, entities, req) = binary_app_tags_env();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.hasTag(context.tag) && principal.getTag(context.tag) == "bar" "#),
            @r#"principal.hasTag(context.tag) && (principal.getTag(context.tag) == "bar")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"some_tags".hasTag(context.tag) && User::"some_tags".getTag(context.tag) == "bar" "#),
            @r#"User::"some_tags".hasTag(context.tag) && (User::"some_tags".getTag(context.tag) == "bar")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"empty_tags".hasTag(context.tag) && E::"empty_tags".getTag(context.tag) == "bar" "#),
            @r#"E::"empty_tags".hasTag(context.tag)"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"E::"none_tags".hasTag(context.tag) && E::"none_tags".getTag(context.tag) == "bar" "#),
            @r#"E::"none_tags".hasTag(context.tag)"#
        );
    }

    #[test]
    fn test_binary_app_get_tag_record() {
        let schema = parse_schema(
            r#"entity E; entity User tags { s: String, inner: { n: Long } }; action get appliesTo {principal: User, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        // `User::"rec"` has a known tag `k` holding a nested record.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "User", "id": "rec" },
                    "tags": { "k": { "s": "bar", "inner": { "n": 1 } } },
                },
            ]),
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"rec".hasTag("k") && User::"other".hasTag("k") && User::"rec".getTag("k") == User::"other".getTag("k")"#),
            @r#"User::"other".hasTag("k") && ({inner: {n: 1}, s: "bar"} == User::"other".getTag("k"))"#
        );
    }

    #[test]
    fn test_binary_app_get_tag_entity() {
        let schema = parse_schema(
            r#"entity E { data: Long }; entity User tags E; action get appliesTo {principal: User, resource: E};"#,
        );
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid("E"),
            None,
            &schema,
        )
        .unwrap();
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "User", "id": "ent" },
                    "tags": {
                        "k": { "__entity": { "type": "E", "id": "target" } },
                        "unk": { "__entity": { "type": "E", "id": "unk" } },
                    },
                },
                {
                    "uid": { "type": "E", "id": "target" },
                    "attrs": { "data": 42 },
                },
            ]),
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);

        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"ent".hasTag("k") && User::"ent".getTag("k").data == 42"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"ent".hasTag("k") && User::"ent".getTag("k") == resource"#),
            @r#"E::"target" == resource"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"User::"ent".hasTag("unk") && User::"ent".getTag("unk").data == 42"#),
            @r#"E::"unk".data == 42"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"principal.hasTag("k") && principal.getTag("k").data == 42"#),
            @r#"principal.hasTag("k") && (principal.getTag("k").data == 42)"#
        );
    }

    // Test containsAll/containsAny operations
    #[test]
    fn test_set_ops() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[true, false].containsAll([false, true])"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[true, false].containsAll([false, resource == Document::"mine"])"#),
            @r#"[false, true].containsAll([false, resource == Document::"mine"])"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[true, false].containsAny([false])"#),
            @"true"
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"[true].containsAny([resource == Document::"alice"])"#),
            @r#"[true].containsAny([resource == Document::"alice"])"#
        );
    }

    #[test]
    fn test_ext_normalization() {
        let eval = Evaluator {
            request: &concrete_user_req(),
            entities: &PartialEntities::new(),
            extensions: Extensions::all_available(),
        };
        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema());
        // When the TPE evaluator evaluates datetime("6640-02-11") with all
        // concrete args, the resulting residual should use the canonical
        // offset(datetime("1970-01-01"), duration("Nms")) form — not the
        // original datetime("6640-02-11") string.
        // The Lean side will for now always produce the offset(datetime("1970-01-01"), _) repr,
        // while the Rust side representation without canonicalization can be either the
        // direct datetime or the offset from the Unix Epoch, depending on where the term
        // originated from.
        assert_snapshot!(
            interpret_typed_str_to_str(r#"datetime("6640-02-11")"#),
            @r#"datetime("1970-01-01").offset(duration("147374467200000ms"))"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"decimal("0.0")"#),
            @r#"decimal("0.0000")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"ip("::1")"#),
            @r#"ip("::1/128")"#
        );
        assert_snapshot!(
            interpret_typed_str_to_str(r#"duration("1d")"#),
            @r#"duration("86400000ms")"#
        );
    }

    #[test]
    fn test_datetime_attr_from_entity_is_normalized() {
        let schema = parse_schema(
            r#"entity E { dt: datetime }; entity User; action get appliesTo {principal: User, resource: E};"#,
        );

        // Create an entity whose attribute "dt" holds a non-canonical datetime
        // (constructed via `datetime(..)` rather than the canonical
        // `datetime(..).offset(..)` form).
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": {
                        "dt": { "__extn": { "fn": "datetime", "arg": "2026-10-01" } },
                    },
                },
            ]),
            &schema,
        )
        .unwrap();

        // Principal is unknown so that the overall expression stays partial,
        // making the datetime value appear in the residual.
        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };

        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.dt"#),
            @r#"datetime("1970-01-01").offset(duration("1790812800000ms"))"#
        );
    }

    #[test]
    fn test_datetime_in_record_attr_from_entity_is_normalized() {
        let schema = parse_schema(
            r#"entity E { rec: { dt: datetime } }; entity User; action get appliesTo {principal: User, resource: E};"#,
        );

        // Entity attribute "rec" is a record containing the non-canonical datetime.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": {
                        "rec": {
                            "dt": { "__extn": { "fn": "datetime", "arg": "2026-10-01" } },
                        },
                    },
                },
            ]),
            &schema,
        )
        .unwrap();

        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };

        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.rec"#),
            @r#"{dt: datetime("1970-01-01").offset(duration("1790812800000ms"))}"#
        );
    }

    #[test]
    fn test_datetime_in_set_attr_from_entity_is_normalized() {
        let schema = parse_schema(
            r#"entity E { s: Set<datetime> }; entity User; action get appliesTo {principal: User, resource: E};"#,
        );

        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": {
                        "s": [
                            { "__extn": { "fn": "datetime", "arg": "1970-01-02" } },
                            { "__extn": { "fn": "datetime", "arg": "2026-10-01" } },
                        ],
                    },
                },
            ]),
            &schema,
        )
        .unwrap();

        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };

        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.s"#),
            @r#"[datetime("1970-01-01").offset(duration("86400000ms")), datetime("1970-01-01").offset(duration("1790812800000ms"))]"#
        );
    }

    #[test]
    fn test_datetime_in_multi_field_record_is_normalized() {
        let schema = parse_schema(
            r#"entity E { rec: { a: Long, dt: datetime, z: Long } }; entity User; action get appliesTo {principal: User, resource: E};"#,
        );

        // Record with "a" before "dt" and "z" after, so the skip(idx+1) path is exercised.
        let entities = PartialEntities::from_json_value(
            serde_json::json!([
                {
                    "uid": { "type": "E", "id": "" },
                    "attrs": {
                        "rec": {
                            "a": 1,
                            "dt": { "__extn": { "fn": "datetime", "arg": "2026-10-01" } },
                            "z": 2,
                        },
                    },
                },
            ]),
            &schema,
        )
        .unwrap();

        let req = PartialRequest::new(
            parse_partial_euid("User"),
            r#"Action::"get""#.parse().unwrap(),
            parse_partial_euid(r#"E::"""#),
            None,
            &schema,
        )
        .unwrap();
        let eval = Evaluator {
            request: &req,
            entities: &entities,
            extensions: Extensions::all_available(),
        };

        let interpret_typed_str_to_str = |e| interpret_typed_str_to_str(&eval, e, &schema);
        assert_snapshot!(
            interpret_typed_str_to_str(r#"resource.rec"#),
            @r#"{a: 1, dt: datetime("1970-01-01").offset(duration("1790812800000ms")), z: 2}"#
        );
    }

    // ---- Phase 6.5: TPE of `.all` / `.any` (reqs 7.1/7.2/7.3/7.4, D-80) ----
    #[cfg(feature = "anyall")]
    mod anyall {
        use super::*;

        fn anyall_schema() -> ValidatorSchema {
            parse_schema(
                r#"entity E { recs: Set<{ n: Long }> };
                   action a appliesTo {
                     principal: E, resource: E,
                     context: { nums: Set<Long>, strs: Set<String>, recs: Set<{ n: Long }> }
                   };"#,
            )
        }

        /// Build an evaluator whose context (hence `context.*` set attributes)
        /// is concrete, so a `.all` receiver reduces to a `Concrete` set.
        fn concrete_ctx_eval(ctx: serde_json::Value) -> (PartialRequest, ValidatorSchema) {
            let schema = anyall_schema();
            let Ok(Context::Value(context)) = Context::from_json_value(ctx) else {
                panic!("expected concrete context")
            };
            let req = PartialRequest::new(
                parse_partial_euid(r#"E::"p""#),
                r#"Action::"a""#.parse().unwrap(),
                parse_partial_euid(r#"E::"r""#),
                Some(context),
                &schema,
            )
            .unwrap();
            (req, schema)
        }

        #[track_caller]
        fn run(ctx: serde_json::Value, expr: &str) -> String {
            let (req, schema) = concrete_ctx_eval(ctx);
            let eval = Evaluator {
                request: &req,
                entities: &PartialEntities::new(),
                extensions: Extensions::all_available(),
            };
            interpret_typed_str_to_str(&eval, expr, &schema)
        }

        // --- W2: concrete-receiver fold (7.1/7.3) ---

        #[test]
        fn concrete_all_true() {
            // every element satisfies the predicate => true
            assert_snapshot!(run(serde_json::json!({"nums":[8000,9000],"strs":[],"recs":[]}),
                r#"context.nums.all(it >= 8000)"#), @"true");
        }

        #[test]
        fn concrete_all_false() {
            assert_snapshot!(run(serde_json::json!({"nums":[1,2],"strs":[],"recs":[]}),
                r#"context.nums.all(it >= 8000)"#), @"false");
        }

        #[test]
        fn concrete_all_empty_is_true() {
            // empty set => true, no QuantifierError (req 2.7)
            assert_snapshot!(run(serde_json::json!({"nums":[],"strs":[],"recs":[]}),
                r#"context.nums.all(it >= 8000)"#), @"true");
        }

        #[test]
        fn concrete_any_lowering() {
            // `.any(p)` lowers to `!all(!p)`; some element >= 8000 => true
            assert_snapshot!(run(serde_json::json!({"nums":[1,9000],"strs":[],"recs":[]}),
                r#"context.nums.any(it >= 8000)"#), @"true");
            assert_snapshot!(run(serde_json::json!({"nums":[1,2],"strs":[],"recs":[]}),
                r#"context.nums.any(it >= 8000)"#), @"false");
        }

        #[test]
        fn concrete_record_elements() {
            // B1: elements are records; predicate reads an attribute of `it` (7.1)
            assert_snapshot!(run(serde_json::json!({"nums":[],"strs":[],"recs":[{"n":1},{"n":2}]}),
                r#"context.recs.all(it.n >= 1)"#), @"true");
            assert_snapshot!(run(serde_json::json!({"nums":[],"strs":[],"recs":[{"n":0},{"n":2}]}),
                r#"context.recs.all(it.n >= 1)"#), @"false");
        }

        #[test]
        fn concrete_string_elements_like() {
            assert_snapshot!(run(serde_json::json!({"nums":[],"strs":["ab","ac"],"recs":[]}),
                r#"context.strs.all(it like "a*")"#), @"true");
        }

        // --- 2.8: no short-circuit on false; an erroring element outranks false ---

        #[test]
        fn false_then_error_is_quantifier_error_not_false() {
            // 0 makes `it >= 8000` false; the division-by-zero-style error here
            // is a non-boolean/erroring predicate on another element. We use an
            // integer-overflow predicate to force an element error.
            // Element 0: 0 >= 8000 = false. Element with overflow: errors.
            // The whole `.all` must be the QuantifierError image (Error), NOT
            // `false` — proving no short-circuit on the false element (2.8).
            // `9223372036854775807 + 1` overflows => predicate errors.
            let out = run(serde_json::json!({"nums":[0_i64, 9223372036854775807_i64],"strs":[],"recs":[]}),
                r#"context.nums.all(it >= 8000 && it + 1 > it)"#);
            // Error residual prints as the synthetic `error()` extension call.
            assert_snapshot!(out, @"error()");
        }

        // --- W3: residual-receiver passthrough (7.2) + reauthorization parity ---

        #[test]
        fn residual_receiver_keeps_node() {
            // `resource` is unknown => `resource.recs` stays residual => the
            // whole `.all` is retained (not errored, not discarded) (7.2).
            let schema = anyall_schema();
            let Ok(Context::Value(context)) =
                Context::from_json_value(serde_json::json!({"nums":[],"strs":[],"recs":[]}))
            else {
                panic!("expected concrete context")
            };
            let req = PartialRequest::new(
                parse_partial_euid(r#"E::"p""#),
                r#"Action::"a""#.parse().unwrap(),
                parse_partial_euid("E"), // resource eid unknown
                Some(context),
                &schema,
            )
            .unwrap();
            let eval = Evaluator {
                request: &req,
                entities: &PartialEntities::new(),
                extensions: Extensions::all_available(),
            };
            assert_snapshot!(
                interpret_typed_str_to_str(&eval, r#"resource.recs.all(it.n >= 1)"#, &schema),
                @"resource.recs.all(!(it.n < 1))"
            );
        }

        #[test]
        fn residual_all_and_false_stays_residual() {
            // `<residual All> && false` must NOT fold to `Concrete(false)`: the
            // `.all` can raise a QuantifierError, so the `&&` simplifier keeps
            // the node (behavioural guard for `can_error_assuming_well_formed`,
            // review Finding 1). If `All` wrongly reported it cannot error, this
            // would collapse to `false`.
            let schema = anyall_schema();
            let Ok(Context::Value(context)) =
                Context::from_json_value(serde_json::json!({"nums":[],"strs":[],"recs":[]}))
            else {
                panic!("expected concrete context")
            };
            let req = PartialRequest::new(
                parse_partial_euid(r#"E::"p""#),
                r#"Action::"a""#.parse().unwrap(),
                parse_partial_euid("E"), // resource eid unknown => receiver residual
                Some(context),
                &schema,
            )
            .unwrap();
            let eval = Evaluator {
                request: &req,
                entities: &PartialEntities::new(),
                extensions: Extensions::all_available(),
            };
            // Kept as `<All> && false`, not simplified to `false`.
            assert_snapshot!(
                interpret_typed_str_to_str(&eval, r#"resource.recs.all(it.n >= 1) && false"#, &schema),
                @"resource.recs.all(!(it.n < 1)) && false"
            );
        }

        #[test]
        fn concrete_receiver_residual_predicate_refines() {
            // Concrete receiver, but the predicate reads an UNKNOWN request
            // value (`principal`) through `in`, so the per-element predicate
            // stays residual. Under the Phase-6.5 conservative fold (D-80 A)
            // this kept the WHOLE node (`[E::"x"].all(it in principal)`); under
            // the req-7.3 / D-92 refinement it now REFINES to an `AllPartial`
            // carrying only the undecided elements, re-expanded for
            // reauthorization as `[<pending bools>, <!decided_false>].all(it)`.
            // One concrete element whose predicate is residual, nothing yet
            // decided `false`, so: pending = [`E::"x" in principal`],
            // decided_false = false (marker `true`).
            let schema = parse_schema(
                r#"entity E in E;
                   action a appliesTo {
                     principal: E, resource: E, context: { es: Set<E> }
                   };"#,
            );
            let Ok(Context::Value(context)) =
                Context::from_json_value(serde_json::json!({ "es": [ {"__entity":{"type":"E","id":"x"}} ] }))
            else {
                panic!("expected concrete context")
            };
            let req = PartialRequest::new(
                parse_partial_euid("E"), // principal unknown
                r#"Action::"a""#.parse().unwrap(),
                parse_partial_euid(r#"E::"r""#),
                Some(context),
                &schema,
            )
            .unwrap();
            let eval = Evaluator {
                request: &req,
                entities: &PartialEntities::new(),
                extensions: Extensions::all_available(),
            };
            // Refined: the receiver is consumed, the single pending element
            // predicate and the `true` (nothing-pinned) marker become set
            // elements, quantified by the identity predicate `it`.
            assert_snapshot!(
                interpret_typed_str_to_str(&eval, r#"context.es.all(it in principal)"#, &schema),
                @r#"[E::"x" in principal, true].all(it)"#
            );
        }

        // --- W4: central stray-`it` backstop (D-32) ---

        // --- Finding 2: reauthorization-parity for the residual soundness invariant (7.2) ---

        #[test]
        fn reauthorization_parity_residual_all() {
            use crate::ast::{Context, EntityUIDEntry, Request};
            // Produce a residual `All` with a genuinely UNKNOWN receiver
            // (`resource.recs`, resource eid unknown), round-trip it through
            // `From<Residual> for Expr`, then COMPLETE the receiver with several
            // concrete sets and assert the completed residual's concrete
            // evaluation equals full concrete evaluation of the ORIGINAL
            // expression on the same input (the req-7.2 soundness invariant).
            let schema = anyall_schema();
            let src = r#"resource.recs.all(it.n >= 1)"#;

            // 1. TPE with unknown resource => residual, then Expr::from.
            let Ok(Context::Value(ctx)) =
                Context::from_json_value(serde_json::json!({"nums":[],"strs":[],"recs":[]}))
            else {
                panic!("concrete context")
            };
            let preq = PartialRequest::new(
                parse_partial_euid(r#"E::"p""#),
                r#"Action::"a""#.parse().unwrap(),
                parse_partial_euid("E"), // resource unknown
                Some(ctx),
                &schema,
            )
            .unwrap();
            let peval = Evaluator {
                request: &preq,
                entities: &PartialEntities::new(),
                extensions: Extensions::all_available(),
            };
            let residual = peval
                .interpret_expr(&parse_typed_expr(src, peval.request, &schema, &SlotEnv::new()))
                .unwrap();
            assert!(residual.is_partial(), "receiver unknown => residual All");
            let roundtripped: ast::Expr = residual.into();
            let original: ast::Expr = src.parse().unwrap();

            // 2. Complete the receiver with >=4 concrete sets and compare the
            // concrete evaluation of the round-tripped residual against full
            // evaluation of the original. The erroring set places the error
            // AFTER a false element to also exercise req 2.8.
            let cases: Vec<(&str, serde_json::Value)> = vec![
                ("all_true", serde_json::json!([{"n": 1}, {"n": 2}])),
                ("has_false", serde_json::json!([{"n": 0}, {"n": 2}])),
                ("empty", serde_json::json!([])),
                ("false_then_ok", serde_json::json!([{"n": 0}, {"n": 5}])),
            ];
            let exts = Extensions::all_available();
            for (name, recs) in cases {
                let parser: crate::entities::EntityJsonParser<'_, '_> =
                    crate::entities::EntityJsonParser::new(
                        None,
                        exts,
                        crate::entities::TCComputation::ComputeNow,
                    );
                let entities = parser
                    .from_json_value(serde_json::json!([
                        { "uid": {"type":"E","id":"r"}, "attrs": { "recs": recs }, "parents": [] }
                    ]))
                    .unwrap();
                let ctx = Context::from_json_value(
                    serde_json::json!({"nums":[],"strs":[],"recs":[]}),
                )
                .unwrap();
                let req = Request::new_unchecked(
                    EntityUIDEntry::known(r#"E::"p""#.parse().unwrap(), None),
                    EntityUIDEntry::known(r#"Action::"a""#.parse().unwrap(), None),
                    EntityUIDEntry::known(r#"E::"r""#.parse().unwrap(), None),
                    Some(ctx),
                );
                let ev = crate::evaluator::Evaluator::new(req, &entities, &exts);
                let got = ev.interpret(&roundtripped, &SlotEnv::new());
                let want = ev.interpret(&original, &SlotEnv::new());
                match (&got, &want) {
                    (Ok(g), Ok(w)) => assert_eq!(g, w, "case {name}: value mismatch"),
                    (Err(_), Err(_)) => {} // both error (QuantifierError image)
                    _ => panic!("case {name}: residual {got:?} vs original {want:?} disagree"),
                }
            }

            // Error-after-false: an overflow predicate. Element {n:0} => `0>=1`
            // false (no error); element {n:i64::MAX} => `>=1` true then
            // `n + i64::MAX` overflows => QuantifierError. The error outranks the
            // earlier false (req 2.8), end-to-end through the round-tripped
            // residual, and must match full evaluation of the original.
            let src_err = r#"resource.recs.all(it.n >= 1 && it.n + 9223372036854775807 >= 0)"#;
            let residual_err = peval
                .interpret_expr(&parse_typed_expr(src_err, peval.request, &schema, &SlotEnv::new()))
                .unwrap();
            let rt_err: ast::Expr = residual_err.into();
            let orig_err: ast::Expr = src_err.parse().unwrap();
            let parser: crate::entities::EntityJsonParser<'_, '_> =
                crate::entities::EntityJsonParser::new(
                    None,
                    exts,
                    crate::entities::TCComputation::ComputeNow,
                );
            let entities = parser
                .from_json_value(serde_json::json!([
                    { "uid": {"type":"E","id":"r"},
                      "attrs": { "recs": [{"n": 0_i64}, {"n": 9223372036854775807_i64}] },
                      "parents": [] }
                ]))
                .unwrap();
            let ctx = Context::from_json_value(serde_json::json!({"nums":[],"strs":[],"recs":[]}))
                .unwrap();
            let req = Request::new_unchecked(
                EntityUIDEntry::known(r#"E::"p""#.parse().unwrap(), None),
                EntityUIDEntry::known(r#"Action::"a""#.parse().unwrap(), None),
                EntityUIDEntry::known(r#"E::"r""#.parse().unwrap(), None),
                Some(ctx),
            );
            let ev = crate::evaluator::Evaluator::new(req, &entities, &exts);
            let got = ev.interpret(&rt_err, &SlotEnv::new());
            let want = ev.interpret(&orig_err, &SlotEnv::new());
            // Both must error (QuantifierError), proving no short-circuit on the
            // earlier false and residual/original agreement.
            assert!(
                got.is_err() && want.is_err(),
                "false-then-error: residual {got:?} vs original {want:?} must both error"
            );
        }

        #[test]
        fn stray_it_is_rejected_by_tpe_backstop() {
            use crate::ast::Unknown;
            use crate::expr_builder::ExprBuilder as _;
            use crate::tpe::err::ExprToResidualError;
            use cool_asserts::assert_matches;
            // Hand-build a typed expr that is a stray `it` sentinel OUTSIDE any
            // predicate, bypassing the parser's `reject_stray_it` entry guard,
            // to prove the TPE backstop (D-32) catches it. The sentinel is the
            // reserved `IT_SENTINEL` unknown.
            let b: ExprBuilder<Option<Type>> =
                ExprBuilder::with_data(Some(Type::primitive_boolean()));
            let stray = b.unknown(Unknown::new_untyped(crate::ast::IT_SENTINEL));
            assert_matches!(
                Residual::try_from_typed_expr(&stray, &SlotEnv::new()),
                Err(ExprToResidualError::StrayIt(_))
            );
        }

        // --- Finding 3: the TPE concrete fold is RE-IMPLEMENTED over `Residual`
        // (not delegating to the Phase-3 `evaluator.rs:780` arm), so the two
        // folds are independent paths that this test pins to the same answers.
        // Each case is a fully-concrete `.all` driven through BOTH the Phase-3
        // concrete evaluator and the TPE concrete-receiver fold; the two must
        // agree on true/false/error (incl. the req-2.8 ordering cases). ---
        #[test]
        fn concrete_fold_matches_phase3_evaluator() {
            use crate::ast::{Context, EntityUIDEntry, Request};
            #[derive(PartialEq, Debug)]
            enum Verdict {
                True,
                False,
                Error,
            }
            // Every case receives `context.nums` (Set<Long>). The predicate and
            // the set together exercise true/false/empty/error and the two
            // error-ordering cases (false-then-error, error-then-false).
            let cases: Vec<(&str, &str, serde_json::Value)> = vec![
                ("all_true", "it >= 8000", serde_json::json!([8000_i64, 9000_i64])),
                ("all_false", "it >= 8000", serde_json::json!([1_i64, 2_i64])),
                ("empty", "it >= 8000", serde_json::json!([])),
                ("single_true", "it >= 8000", serde_json::json!([8000_i64])),
                ("single_false", "it >= 8000", serde_json::json!([1_i64])),
                ("two_false", "it >= 8000", serde_json::json!([1_i64, 2_i64])),
                // overflow => every element errors
                ("overflow", "it + 9223372036854775807 >= 0", serde_json::json!([1_i64, 2_i64])),
                // false element FIRST, erroring element AFTER (req 2.8)
                (
                    "false_then_error",
                    "it >= 8000 && it + 9223372036854775807 >= 0",
                    serde_json::json!([0_i64, 9223372036854775807_i64]),
                ),
                // erroring element FIRST, false element AFTER (req 2.8)
                (
                    "error_then_false",
                    "it + 9223372036854775807 >= 0 && it >= 8000",
                    serde_json::json!([9223372036854775807_i64, 0_i64]),
                ),
            ];
            let exts = Extensions::all_available();
            for (name, pred, nums) in cases {
                let src = format!("context.nums.all({pred})");
                let ctx_json = serde_json::json!({ "nums": nums, "strs": [], "recs": [] });

                // --- TPE concrete-receiver fold ---
                let Ok(Context::Value(ctx)) = Context::from_json_value(ctx_json.clone()) else {
                    panic!("concrete context")
                };
                let preq = PartialRequest::new(
                    parse_partial_euid(r#"E::"p""#),
                    r#"Action::"a""#.parse().unwrap(),
                    parse_partial_euid(r#"E::"r""#),
                    Some(ctx),
                    &anyall_schema(),
                )
                .unwrap();
                let peval = Evaluator {
                    request: &preq,
                    entities: &PartialEntities::new(),
                    extensions: exts,
                };
                let tpe = peval
                    .interpret_expr(&parse_typed_expr(
                        &src,
                        peval.request,
                        &anyall_schema(),
                        &SlotEnv::new(),
                    ))
                    .unwrap();
                let tpe_verdict = if tpe.is_true() {
                    Verdict::True
                } else if tpe.is_false() {
                    Verdict::False
                } else if tpe.is_error() {
                    Verdict::Error
                } else {
                    panic!("case {name}: TPE did not fully fold a concrete receiver: {tpe:?}")
                };

                // --- Phase-3 concrete evaluator ---
                let ctx = Context::from_json_value(ctx_json).unwrap();
                let req = Request::new_unchecked(
                    EntityUIDEntry::known(r#"E::"p""#.parse().unwrap(), None),
                    EntityUIDEntry::known(r#"Action::"a""#.parse().unwrap(), None),
                    EntityUIDEntry::known(r#"E::"r""#.parse().unwrap(), None),
                    Some(ctx),
                );
                let entities = crate::entities::Entities::new();
                let ev = crate::evaluator::Evaluator::new(req, &entities, &exts);
                let expr: ast::Expr = src.parse().unwrap();
                let concrete_verdict = match ev.interpret(&expr, &SlotEnv::new()) {
                    Ok(v) => match v.get_as_bool() {
                        Ok(true) => Verdict::True,
                        Ok(false) => Verdict::False,
                        Err(_) => Verdict::Error,
                    },
                    Err(_) => Verdict::Error,
                };

                assert_eq!(
                    tpe_verdict, concrete_verdict,
                    "case {name}: TPE fold and Phase-3 evaluator disagree"
                );
            }
        }
    }
}
