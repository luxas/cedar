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

use crate::ast::{EntityType, EntityUID, Expr, ExprKind, Literal, Name, Pattern, Template};
use crate::parser::Loc;

/// Predicates of the `.all`/`.any` set quantifiers among `expr`'s
/// subexpressions. `subexpressions()` yields only `Expr` nodes, so walks that
/// must also see predicate leaves chain these in.
#[cfg(feature = "anyall")]
fn quantifier_preds(expr: &Expr) -> impl Iterator<Item = &crate::ast::PredExpr> {
    expr.subexpressions()
        .filter_map(|e| match e.expr_kind() {
            ExprKind::All { pred, .. } => Some(pred.as_ref()),
            _ => None,
        })
        .flat_map(|p| p.subexpressions())
}

/// Returns an iterator over all literal entity uids in the expression.
pub(super) fn expr_entity_uids(expr: &Expr) -> impl Iterator<Item = &EntityUID> {
    let uids = expr.subexpressions().filter_map(|e| match e.expr_kind() {
        ExprKind::Lit(Literal::EntityUID(uid)) => Some(uid.as_ref()),
        _ => None,
    });
    #[cfg(feature = "anyall")]
    let uids = uids.chain(quantifier_preds(expr).filter_map(|p| match p.expr_kind() {
        crate::ast::PredExprKind::Lit(Literal::EntityUID(uid)) => Some(uid.as_ref()),
        _ => None,
    }));
    uids
}

/// Returns an iterator over all entity type names in the expression.
pub(super) fn expr_entity_type_names(expr: &Expr) -> impl Iterator<Item = &EntityType> {
    let tys = expr.subexpressions().filter_map(|e| match e.expr_kind() {
        ExprKind::Lit(Literal::EntityUID(uid)) => Some(uid.entity_type()),
        ExprKind::Is { entity_type, .. } => Some(entity_type),
        _ => None,
    });
    #[cfg(feature = "anyall")]
    let tys = tys.chain(quantifier_preds(expr).filter_map(|p| match p.expr_kind() {
        crate::ast::PredExprKind::Lit(Literal::EntityUID(uid)) => Some(uid.entity_type()),
        crate::ast::PredExprKind::Is { entity_type, .. } => Some(entity_type),
        _ => None,
    }));
    tys
}

/// Returns an iterator over all literal entity uids in a policy. This iterates
/// over any entities in the policy scope condition in addition to any entities
/// in the body.
pub(super) fn policy_entity_uids(template: &Template) -> impl Iterator<Item = &EntityUID> {
    template
        .principal_constraint()
        .as_inner()
        .get_euid()
        .into_iter()
        .map(|euid| euid.as_ref())
        .chain(template.action_constraint().iter_euids())
        .chain(
            template
                .resource_constraint()
                .as_inner()
                .get_euid()
                .into_iter()
                .map(|euid| euid.as_ref()),
        )
        .chain(
            template
                .non_scope_constraints()
                .into_iter()
                .flat_map(expr_entity_uids),
        )
}

/// Returns an iterator over all entity type names in the policy. This iterates
/// over the policy scope condition in addition to the body.
pub(super) fn policy_entity_type_names(template: &Template) -> impl Iterator<Item = &EntityType> {
    template
        .principal_constraint()
        .as_inner()
        .iter_entity_type_names()
        .chain(template.action_constraint().iter_entity_type_names())
        .chain(
            template
                .resource_constraint()
                .as_inner()
                .iter_entity_type_names(),
        )
        .chain(
            template
                .non_scope_constraints()
                .into_iter()
                .flat_map(expr_entity_type_names),
        )
}

/// The 3 different "classes" of text in an expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TextKind<'a> {
    /// String Literals
    String(Option<&'a Loc>, &'a str),
    /// Identifiers
    Identifier(Option<&'a Loc>, &'a str),
    /// Pattern Strings
    Pattern(Option<&'a Loc>, &'a Pattern),
}

/// Returns an iterator over all text (strings and identifiers) in the expression.
pub(super) fn expr_text(e: &'_ Expr) -> impl Iterator<Item = TextKind<'_>> {
    let text = e.subexpressions().flat_map(text_in_expr);
    #[cfg(feature = "anyall")]
    let text = text.chain(quantifier_preds(e).flat_map(text_in_pred));
    text
}

/// `text_in_expr` for a node of a `.all`/`.any` predicate
#[cfg(feature = "anyall")]
fn text_in_pred(p: &crate::ast::PredExpr) -> Vec<TextKind<'_>> {
    use crate::ast::PredExprKind;
    let loc = p.source_loc();
    match p.expr_kind() {
        PredExprKind::Lit(lit) => text_in_lit(loc, lit).into_iter().collect(),
        PredExprKind::ExtensionFunctionApp { fn_name, .. } => text_in_name(loc, fn_name).collect(),
        PredExprKind::GetAttr { attr, .. } | PredExprKind::HasAttr { attr, .. } => {
            vec![TextKind::Identifier(loc, attr)]
        }
        PredExprKind::ExtHasAttr { attrs, .. } => attrs
            .iter()
            .map(|attr| TextKind::Identifier(loc, attr))
            .collect(),
        PredExprKind::Like { pattern, .. } => vec![TextKind::Pattern(loc, pattern)],
        PredExprKind::Record(map) => map
            .keys()
            .map(|attr| TextKind::Identifier(loc, attr))
            .collect(),
        _ => vec![],
    }
}

// Returns a vector containing the text in the top level expression
fn text_in_expr(e: &Expr) -> impl IntoIterator<Item = TextKind<'_>> {
    match e.expr_kind() {
        ExprKind::Lit(lit) => text_in_lit(e.source_loc(), lit).into_iter().collect(),
        ExprKind::ExtensionFunctionApp { fn_name, .. } => {
            text_in_name(e.source_loc(), fn_name).collect()
        }
        ExprKind::GetAttr { attr, .. } => vec![TextKind::Identifier(e.source_loc(), attr)],
        ExprKind::HasAttr { attr, .. } => vec![TextKind::Identifier(e.source_loc(), attr)],
        ExprKind::ExtHasAttr { attrs, .. } => attrs
            .iter()
            .map(|attr| TextKind::Identifier(e.source_loc(), attr))
            .collect(),
        ExprKind::Like { pattern, .. } => {
            vec![TextKind::Pattern(e.source_loc(), pattern)]
        }
        ExprKind::Record(map) => map
            .keys()
            .map(|attr| TextKind::Identifier(e.source_loc(), attr))
            .collect(),
        _ => vec![],
    }
}

fn text_in_lit<'a>(
    loc: Option<&'a Loc>,
    lit: &'a Literal,
) -> impl IntoIterator<Item = TextKind<'a>> {
    match lit {
        Literal::Bool(_) => vec![],
        Literal::Long(_) => vec![],
        Literal::String(s) => vec![TextKind::String(loc, s)],
        Literal::EntityUID(euid) => text_in_euid(loc, euid).collect(),
    }
}

fn text_in_euid<'a>(
    loc: Option<&'a Loc>,
    euid: &'a EntityUID,
) -> impl Iterator<Item = TextKind<'a>> {
    text_in_entity_type(loc, euid.entity_type())
        .into_iter()
        .chain(std::iter::once(TextKind::Identifier(
            loc,
            euid.eid().as_ref(),
        )))
}

fn text_in_entity_type<'a>(
    loc: Option<&'a Loc>,
    ty: &'a EntityType,
) -> impl IntoIterator<Item = TextKind<'a>> {
    text_in_name(loc, ty.name()).collect::<Vec<_>>()
}

fn text_in_name<'a>(loc: Option<&'a Loc>, name: &'a Name) -> impl Iterator<Item = TextKind<'a>> {
    name.as_ref()
        .namespace_components()
        .map(move |id| TextKind::Identifier(loc, id.as_ref()))
        .chain(std::iter::once(TextKind::Identifier(
            loc,
            name.basename_as_ref().as_ref(),
        )))
}

#[cfg(test)]
mod tests {
    use super::{expr_entity_uids, expr_text};
    use crate::ast::{EntityUID, Expr, Literal, Pattern, PatternElem, Var};
    use crate::validator::expr_iterator::TextKind;
    use std::{collections::HashSet, str::FromStr};

    #[test]
    fn no_entities() {
        let no_entities = Expr::val(1);
        let entities: Vec<EntityUID> = expr_entity_uids(&no_entities).cloned().collect();
        assert_eq!(Vec::<EntityUID>::new(), entities);
    }

    #[test]
    fn entity_literal() {
        let euid =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_lit = Expr::val(euid.clone());

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_lit).cloned().collect();
        assert_eq!(vec![euid], entities);
    }

    #[test]
    fn entity_eq() {
        let euid =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_eq = Expr::is_eq(Expr::var(Var::Principal), Expr::val(euid.clone()));

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_eq).cloned().collect();
        assert_eq!(vec![euid], entities);
    }

    #[test]
    fn entity_in() {
        let euid =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_eq = Expr::is_in(Expr::var(Var::Principal), Expr::val(euid.clone()));

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_eq).cloned().collect();
        assert_eq!(vec![euid], entities);
    }

    #[test]
    fn entity_and() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let euid_bar =
            EntityUID::with_eid_and_type("test_entity_type", "bar").expect("valid identifier");
        let entity_and = Expr::and(
            Expr::is_eq(Expr::var(Var::Principal), Expr::val(euid_foo.clone())),
            Expr::is_in(Expr::var(Var::Resource), Expr::val(euid_bar.clone())),
        );

        let entities: HashSet<EntityUID> = expr_entity_uids(&entity_and).cloned().collect();
        assert_eq!(HashSet::from([euid_foo, euid_bar]), entities);
    }

    #[test]
    fn entity_in_set() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let euid_bar =
            EntityUID::with_eid_and_type("test_entity_type", "bar").expect("valid identifier");
        let euid_baz =
            EntityUID::with_eid_and_type("test_entity_type", "baz").expect("valid identifier");
        let entity_set = Expr::is_in(
            Expr::var(Var::Action),
            Expr::set(vec![
                Expr::val(euid_foo.clone()),
                Expr::val(euid_bar.clone()),
                Expr::val(euid_baz.clone()),
            ]),
        );

        let entities: HashSet<EntityUID> = expr_entity_uids(&entity_set).cloned().collect();
        assert_eq!(HashSet::from([euid_foo, euid_bar, euid_baz]), entities);
    }

    #[test]
    fn entity_if() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let euid_bar =
            EntityUID::with_eid_and_type("test_entity_type", "bar").expect("valid identifier");
        let euid_baz =
            EntityUID::with_eid_and_type("test_entity_type", "baz").expect("valid identifier");
        let entity_if = Expr::ite(
            Expr::is_in(Expr::var(Var::Principal), Expr::val(euid_foo.clone())),
            Expr::val(euid_bar.clone()),
            Expr::val(euid_baz.clone()),
        );

        let entities: HashSet<EntityUID> = expr_entity_uids(&entity_if).cloned().collect();
        assert_eq!(HashSet::from([euid_foo, euid_bar, euid_baz]), entities);
    }

    #[test]
    fn entity_has_attr() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_has_attr = Expr::has_attr(Expr::val(euid_foo.clone()), "bar".into());

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_has_attr).cloned().collect();
        assert_eq!(vec![euid_foo], entities);
    }

    #[test]
    fn entity_get_attr() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_get_attr = Expr::get_attr(Expr::val(euid_foo.clone()), "bar".into());

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_get_attr).cloned().collect();
        assert_eq!(vec![euid_foo], entities);
    }

    #[test]
    fn entity_record_get_attr() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_get_elem = Expr::get_attr(
            Expr::record(vec![("bar".into(), Expr::val(euid_foo.clone()))]).unwrap(),
            "bar".into(),
        );

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_get_elem).cloned().collect();
        assert_eq!(vec![euid_foo], entities);
    }

    #[test]
    fn entity_record() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let entity_record =
            Expr::record(vec![("bar".into(), Expr::val(euid_foo.clone()))]).unwrap();

        let entities: Vec<EntityUID> = expr_entity_uids(&entity_record).cloned().collect();
        assert_eq!(vec![euid_foo], entities);
    }

    #[test]
    fn entity_full_scope() {
        let euid_foo =
            EntityUID::with_eid_and_type("test_entity_type", "foo").expect("valid identifier");
        let euid_bar =
            EntityUID::with_eid_and_type("test_entity_type", "bar").expect("valid identifier");
        let euid_baz =
            EntityUID::with_eid_and_type("test_entity_type", "baz").expect("valid identifier");
        let euid_buz =
            EntityUID::with_eid_and_type("test_entity_type", "buz").expect("valid identifier");
        let scope = Expr::and(
            Expr::is_eq(Expr::var(Var::Principal), Expr::val(euid_foo.clone())),
            Expr::and(
                Expr::is_in(
                    Expr::var(Var::Action),
                    Expr::set(vec![
                        Expr::val(euid_bar.clone()),
                        Expr::val(euid_baz.clone()),
                    ]),
                ),
                Expr::is_in(Expr::var(Var::Action), Expr::val(euid_buz.clone())),
            ),
        );

        let entities: HashSet<EntityUID> = expr_entity_uids(&scope).cloned().collect();
        assert_eq!(
            HashSet::from([euid_foo, euid_bar, euid_baz, euid_buz]),
            entities
        );
    }

    #[test]
    fn test_strs() {
        let p = Expr::and(
            Expr::get_attr(Expr::var(Var::Principal), "test".into()),
            Expr::val(EntityUID::from_str("a::b::\"c\"").unwrap()),
        );
        let strs: HashSet<_> = expr_text(&p).collect();
        assert_eq!(
            HashSet::from([
                TextKind::Identifier(None, "test"),
                TextKind::Identifier(None, "a"),
                TextKind::Identifier(None, "b"),
                TextKind::Identifier(None, "c")
            ]),
            strs
        );
    }

    #[test]
    fn test_strs_lit() {
        let e = Expr::and(
            Expr::val(Literal::Bool(true)),
            Expr::and(
                Expr::val(Literal::Bool(false)),
                Expr::and(
                    Expr::val(EntityUID::from_str("a::b::\"c\"").unwrap()),
                    Expr::and(Expr::val(Literal::Long(123)), Expr::val("this is a test")),
                ),
            ),
        );
        let strs: HashSet<_> = expr_text(&e).collect();
        assert_eq!(
            HashSet::from([
                TextKind::Identifier(None, "a"),
                TextKind::Identifier(None, "b"),
                TextKind::Identifier(None, "c"),
                TextKind::String(None, "this is a test"),
            ]),
            strs
        );
    }

    #[test]
    fn test_strs_atrs() {
        let r = Expr::record([
            ("a1".into(), Expr::val(true)),
            ("a2".into(), Expr::val(false)),
        ])
        .unwrap();
        let e = Expr::ite(
            Expr::get_attr(
                Expr::val(EntityUID::from_str("another::\"euid\"").unwrap()),
                "myattr".into(),
            ),
            Expr::has_attr(r, "myattr2".into()),
            Expr::val(false),
        );

        let strs: HashSet<_> = expr_text(&e).collect();

        assert_eq!(
            HashSet::from([
                TextKind::Identifier(None, "a1"),
                TextKind::Identifier(None, "a2"),
                TextKind::Identifier(None, "another"),
                TextKind::Identifier(None, "euid"),
                TextKind::Identifier(None, "myattr"),
                TextKind::Identifier(None, "myattr2"),
            ]),
            strs
        );
    }

    #[test]
    fn test_strs_ext() {
        let e = Expr::call_extension_fn("test".parse().unwrap(), vec![Expr::val("arg")]);
        let strs: HashSet<_> = expr_text(&e).collect();
        assert_eq!(
            HashSet::from([
                TextKind::Identifier(None, "test"),
                TextKind::String(None, "arg"),
            ]),
            strs
        );
    }

    #[test]
    fn test_strs_like() {
        let p = Pattern::from(vec![PatternElem::Wildcard, PatternElem::Char('a')]);
        let e = Expr::like(Expr::val("test"), p.clone());
        let strs: HashSet<_> = expr_text(&e).collect();

        assert_eq!(
            HashSet::from([TextKind::Pattern(None, &p), TextKind::String(None, "test")]),
            strs
        );
    }

    #[cfg(feature = "anyall")]
    #[test]
    fn entities_and_text_inside_quantifier_predicate() {
        use crate::ast::{BinaryOp, PredExprBuilder};
        let pb = PredExprBuilder::new;
        let uid = EntityUID::with_eid_and_type("Photo", "p").unwrap();
        // `[1].all(it == 1 && Photo::"p".owner == "a")`
        let p = pb().and(
            pb().binary_app(BinaryOp::Eq, pb().item(), pb().val(1)).unwrap(),
            pb().binary_app(
                BinaryOp::Eq,
                pb().get_attr(pb().val(uid.clone()), "owner".into()),
                pb().val("a"),
            )
            .unwrap(),
        );
        let e = Expr::all(Expr::set([Expr::val(1)]), p);
        let uids: Vec<EntityUID> = expr_entity_uids(&e).cloned().collect();
        assert_eq!(uids, vec![uid]);
        let text: HashSet<TextKind<'_>> = expr_text(&e).collect();
        assert!(text.contains(&TextKind::Identifier(None, "owner")));
        assert!(text.contains(&TextKind::String(None, "a")));
    }
}
