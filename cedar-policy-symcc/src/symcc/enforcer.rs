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

//! This file defines the algorithm for emitting well-formedness assumptions about
//! Cedar hierarchies. A valid Cedar hierarchy graph is the irreflexive transitive
//! closure of a DAG.  (Note that the `in` operator is reflexive, but this is
//! enforced separately by the semantics, and it's not needed in the hierarchy.)
//!
//! The generated assumptions are expressed as Terms that enforce the acyclicty and
//! transitivity constraints on the symbolic store, for a given Cedar expression.
//!
//! For more technical details, see comments in SymCC/Enforcer.lean.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use cedar_policy_core::ast::{Expr, ExprKind};

use super::{
    compiler::compile,
    env::{SymEntities, SymEnv},
    factory::{and, app, implies, is_some, not, option_get, set_is_empty, set_member, set_subset},
    function::UnaryFunction,
    term::Term,
    term_type::TermType,
    type_abbrevs::EntityType,
};

/// Helper function used by `footprint()`.
/// In Lean this is defined inline inside `footprint()`, but in Rust, you can't
/// define closures that take generic arguments (like `impl IntoIterator`), so
/// we define it as a standalone function.
fn of_branch<'a>(
    x1: &Expr,
    ft1: impl IntoIterator<Item = Term> + 'a,
    ft2: impl IntoIterator<Item = Term> + 'a,
    ft3: impl IntoIterator<Item = Term> + 'a,
    env: &SymEnv,
) -> Box<dyn Iterator<Item = Term> + 'a> {
    match compile(x1, env) {
        Ok(Term::Some(t)) if *t == true.into() => Box::new(ft2.into_iter()),
        Ok(Term::Some(t)) if *t == false.into() => Box::new(ft3.into_iter()),
        Ok(_) => Box::new(ft1.into_iter().chain(ft2).chain(ft3)),
        Err(_) => Box::new(std::iter::empty()),
    }
}

/// Returns the terms corresponding to subexpressions of `x` of the following form:
///
///   * A variable term with an entity type
///   * An entity reference literal
///   * An attribute access expression with an entity type
///   * A binary (`getTag`) expression with an entity type
///
/// These are the only basic expressions in Cedar that may evaluate to an entity.
/// All other expressions that evaluate to an entity are build up from the above
/// basic expressions.
///
/// All returned terms are of type `TermType.option .entity`.
//
// NOTE: this essentially calls `compile()` on every subexpression, which is
// redundant by itself, and also potentially redundant with other `compile()`
// calls in verifier.rs. For a much more optimized version, use functionality
// in the `symccopt` module.
#[expect(
    unreachable_patterns,
    reason = "pattern may or may not be unreachable depending on whether `tolerant-ast` feature of cedar-policy-core is enabled"
)]
pub(crate) fn footprint<'a>(x: &'a Expr, env: &'a SymEnv) -> Box<dyn Iterator<Item = Term> + 'a> {
    let of_entity = |x: &Expr| -> Box<dyn Iterator<Item = Term>> {
        match compile(x, env) {
            Ok(t) => {
                if t.type_of().is_option_entity_type() {
                    Box::new(std::iter::once(t))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Err(_) => Box::new(std::iter::empty()),
        }
    };
    #[expect(clippy::unimplemented, reason = "Should fail at an earlier stage")]
    match x.expr_kind() {
        ExprKind::Lit(_) | ExprKind::Var(_) => of_entity(x),
        ExprKind::If {
            test_expr,
            then_expr,
            else_expr,
        } => of_branch(
            test_expr,
            footprint(test_expr, env),
            footprint(then_expr, env),
            footprint(else_expr, env),
            env,
        ),
        ExprKind::And { left, right } => of_branch(
            left,
            footprint(left, env),
            footprint(right, env),
            std::iter::empty(),
            env,
        ),
        ExprKind::Or { left, right } => of_branch(
            left,
            footprint(left, env),
            std::iter::empty(),
            footprint(right, env),
            env,
        ),
        ExprKind::BinaryApp { arg1, arg2, .. } => Box::new(
            of_entity(x)
                .chain(footprint(arg1, env))
                .chain(footprint(arg2, env)),
        ),
        ExprKind::GetAttr { expr, .. } => Box::new(of_entity(x).chain(footprint(expr, env))),
        ExprKind::HasAttr { expr, .. }
        | ExprKind::ExtHasAttr { expr, .. }
        | ExprKind::UnaryApp { arg: expr, .. }
        | ExprKind::Like { expr, .. }
        | ExprKind::Is { expr, .. } => footprint(expr, env),
        ExprKind::ExtensionFunctionApp { args: exprs, .. } | ExprKind::Set(exprs) => {
            Box::new(exprs.iter().flat_map(|x| footprint(x, env)))
        }
        ExprKind::Record(axs) => Box::new(axs.values().flat_map(|x| footprint(x, env))),
        // `.all` (feature `anyall`, D-70 A / D-71): the receiver's own footprint,
        // plus the `it`-free entity subterms of the predicate compiled against
        // the reserved element variable. Mirrors Lean `footprint`'s `.all` arm
        // (`footprint x₁ ∪ footprintAllPred p x₁`).
        #[cfg(feature = "anyall")]
        ExprKind::All { expr: x1, pred: p } => {
            Box::new(footprint(x1, env).chain(footprint_all_pred(p, x1, env)))
        }
        ExprKind::Slot(_) => unimplemented!("analyzing templates is not currently supported"),
        ExprKind::Unknown(_) => {
            unimplemented!("analyzing partial expressions is not currently supported")
        }
        _ => unimplemented!("analyzing `{}` is not currently supported", x),
    }
}

/// Footprint of a quantifier predicate `p` compiled against the element term
/// `it` (feature `anyall`, D-70 A): the entity-typed compiled terms of the
/// `it`-FREE subexpressions of `p`, in the same `of_entity`/`of_branch` shape as
/// `footprint` for `Expr` (over `compile_pred`). An `it`-dependent subexpression
/// contributes only its `it`-free children. Mirrors Lean `footprintPred`.
#[cfg(feature = "anyall")]
fn footprint_pred(
    p: &cedar_policy_core::ast::PredExpr,
    it: &Term,
    env: &SymEnv,
) -> Vec<Term> {
    use cedar_policy_core::ast::PredExprKind;
    // `ofEntity q`: if `q` mentions `it` it cannot be grounded; else compile it
    // and keep it iff it is an option-entity term.
    let of_entity = |q: &cedar_policy_core::ast::PredExpr| -> Vec<Term> {
        if super::compiler::pred_mentions_item(q) {
            return Vec::new();
        }
        match super::compiler::compile_pred(q, it, env) {
            Ok(t) if t.type_of().is_option_entity_type() => vec![t],
            _ => Vec::new(),
        }
    };
    // `ofBranch c ft1 ft2 ft3`: mirror compile's branch laziness.
    let of_branch =
        |c: &cedar_policy_core::ast::PredExpr, ft1: Vec<Term>, ft2: Vec<Term>, ft3: Vec<Term>| -> Vec<Term> {
            match super::compiler::compile_pred(c, it, env) {
                Ok(Term::Some(t)) if *t == true.into() => ft2,
                Ok(Term::Some(t)) if *t == false.into() => ft3,
                Ok(_) => ft1.into_iter().chain(ft2).chain(ft3).collect(),
                Err(_) => Vec::new(),
            }
        };
    match p.expr_kind() {
        PredExprKind::Item => Vec::new(),
        PredExprKind::Lit(_) | PredExprKind::Var(_) => of_entity(p),
        PredExprKind::If { test_expr, then_expr, else_expr } => of_branch(
            test_expr,
            footprint_pred(test_expr, it, env),
            footprint_pred(then_expr, it, env),
            footprint_pred(else_expr, it, env),
        ),
        PredExprKind::And { left, right } => of_branch(
            left,
            footprint_pred(left, it, env),
            footprint_pred(right, it, env),
            Vec::new(),
        ),
        PredExprKind::Or { left, right } => of_branch(
            left,
            footprint_pred(left, it, env),
            Vec::new(),
            footprint_pred(right, it, env),
        ),
        PredExprKind::BinaryApp { arg1, arg2, .. } => of_entity(p)
            .into_iter()
            .chain(footprint_pred(arg1, it, env))
            .chain(footprint_pred(arg2, it, env))
            .collect(),
        PredExprKind::GetAttr { expr, .. } => of_entity(p)
            .into_iter()
            .chain(footprint_pred(expr, it, env))
            .collect(),
        PredExprKind::HasAttr { expr, .. }
        | PredExprKind::ExtHasAttr { expr, .. }
        | PredExprKind::Like { expr, .. }
        | PredExprKind::Is { expr, .. }
        | PredExprKind::UnaryApp { arg: expr, .. } => footprint_pred(expr, it, env),
        PredExprKind::ExtensionFunctionApp { args, .. } => {
            args.iter().flat_map(|x| footprint_pred(x, it, env)).collect()
        }
        PredExprKind::Record(m) => m.values().flat_map(|x| footprint_pred(x, it, env)).collect(),
    }
}

/// Footprint of `expr.all(p)`'s predicate (feature `anyall`, D-71): compile the
/// receiver; if it is a set term, collect `footprint_pred` of `p` against the
/// element binding, mirroring compile's literal-fold / symbolic split. Kept
/// separate from `footprint` so the receiver match does not complicate it.
/// Mirrors Lean `footprintAllPred`.
#[cfg(feature = "anyall")]
fn footprint_all_pred(
    p: &cedar_policy_core::ast::PredExpr,
    x1: &Expr,
    env: &SymEnv,
) -> Vec<Term> {
    use super::factory::{anyall_it_var, some_of};
    use super::term_type::TermType;
    let t = match compile(x1, env) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    if matches!(t, Term::None(_)) {
        return Vec::new(); // D-69: a `.none` receiver compiles no predicate.
    }
    let got = option_get(t);
    let TermType::Set { ty: elem_ty } = got.type_of() else {
        return Vec::new();
    };
    let elem_ty = std::sync::Arc::unwrap_or_clone(elem_ty);
    match &got {
        // D-71: literal set of literal elements ⇒ predicate per element.
        Term::Set { elts, .. } if elts.iter().all(Term::is_literal) => elts
            .iter()
            .flat_map(|vi| footprint_pred(p, &some_of(vi.clone()), env))
            .collect(),
        // symbolic ⇒ bind the reserved element variable.
        _ => footprint_pred(p, &some_of(Term::Var(anyall_it_var(elem_ty))), env),
    }
}

/// Returns the set of Terms corresponding to the footprints of `exprs`.
/// Returning a `BTreeSet` ensures there are no duplicates.
pub(crate) fn footprints<'a>(
    exprs: impl IntoIterator<Item = &'a Expr>,
    env: &SymEnv,
) -> BTreeSet<Term> {
    exprs.into_iter().flat_map(|x| footprint(x, env)).collect()
}

/// Returns the acyclicity constraint for the given term
pub(crate) fn acyclicity(t: &Term, es: &SymEntities) -> Term {
    match t.type_of() {
        TermType::Option { ty } if matches!(*ty, TermType::Entity { .. }) => {
            match Arc::unwrap_or_clone(ty) {
                TermType::Entity { ety } => match es.ancestors_of_type(&ety, &ety) {
                    Some(f) => {
                        let t_unwrapped = option_get(t.clone());
                        implies(
                            is_some(t.clone()),
                            not(set_member(t_unwrapped.clone(), app(f.clone(), t_unwrapped))),
                        )
                    }
                    None => true.into(),
                },
                #[expect(
                    clippy::unreachable,
                    reason = "Code already checks that matches entity_type"
                )]
                _ => unreachable!("already checked it matches TermType::Entity above"),
            }
        }
        _ => true.into(),
    }
}

/// Returns the transitivity constraint for the given term
pub(crate) fn transitivity(t1: &Term, t2: &Term, es: &SymEntities) -> Term {
    let is_ancestor = |t2_unwrapped: Term, t1_unwrapped: Term, f12: UnaryFunction| -> Term {
        and(
            and(is_some(t1.clone()), is_some(t2.clone())),
            set_member(t2_unwrapped, app(f12, t1_unwrapped)),
        )
    };
    let are_ancestors = |t2_unwrapped: &Term,
                         anc2: &BTreeMap<EntityType, UnaryFunction>,
                         t1_unwrapped: &Term,
                         ety1|
     -> Term {
        anc2.iter().fold(true.into(), |acc, (ety3, f23)| {
            // inlining Lean's `areAncestorsOfType` here
            let are_ancestors_of_type = match es.ancestors_of_type(ety1, ety3) {
                Some(f13) => set_subset(
                    app(f23.clone(), t2_unwrapped.clone()),
                    app(f13.clone(), t1_unwrapped.clone()),
                ),
                None => set_is_empty(app(f23.clone(), t2_unwrapped.clone())),
            };
            and(acc, are_ancestors_of_type)
        })
    };

    if t1 == t2 {
        true.into()
    } else {
        match (t1.type_of(), t2.type_of()) {
            (TermType::Option { ty: ty1 }, TermType::Option { ty: ty2 }) => {
                match (Arc::unwrap_or_clone(ty1), Arc::unwrap_or_clone(ty2)) {
                    (TermType::Entity { ety: ety1 }, TermType::Entity { ety: ety2 }) => {
                        match (es.ancestors_of_type(&ety1, &ety2), es.ancestors(&ety2)) {
                            (Some(f12), Some(anc2)) => {
                                let t1_unwrapped = option_get(t1.clone());
                                let t2_unwrapped = option_get(t2.clone());
                                implies(
                                    is_ancestor(
                                        t2_unwrapped.clone(),
                                        t1_unwrapped.clone(),
                                        f12.clone(),
                                    ),
                                    are_ancestors(&t2_unwrapped, anc2, &t1_unwrapped, &ety1),
                                )
                            }
                            (_, _) => true.into(),
                        }
                    }
                    (_, _) => true.into(),
                }
            }
            (_, _) => true.into(),
        }
    }
}

/// Returns the ground acyclicity and transitivity assumptions for xs and env
pub fn enforce<'a>(xs: impl IntoIterator<Item = &'a Expr>, env: &SymEnv) -> BTreeSet<Term> {
    let ts = footprints(xs, env);
    let ac = ts.iter().map(|t| acyclicity(t, &env.entities));
    let tr = ts
        .iter()
        .flat_map(|t| ts.iter().map(|t2| transitivity(t, t2, &env.entities)));
    ac.chain(tr).collect()
}
