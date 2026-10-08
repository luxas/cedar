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

//! Validator tests for the `.all` / `.any` set quantifier (requirements 1.5,
//! 6.1-6.3).

use super::test_utils::{
    assert_typecheck_fails, assert_typecheck_fails_for_mode, assert_typechecks_empty_schema,
    assert_typechecks_for_mode, empty_schema_file,
};
use crate::ast::{BinaryOp, Expr, PredExpr, PredExprBuilder};
use crate::validator::{types::Type, ValidationError, ValidationMode};
use std::collections::HashSet;

fn pb() -> PredExprBuilder {
    PredExprBuilder::new()
}

/// `0 < it`
fn it_pos() -> PredExpr {
    pb().binary_app(BinaryOp::Less, pb().val(0), pb().item()).unwrap()
}

fn ints(is: &[i64]) -> Expr {
    Expr::set(is.iter().map(|i| Expr::val(*i)))
}

fn rec(attr: &str, v: &str) -> Expr {
    Expr::record([(attr.into(), Expr::val(v))]).unwrap()
}

fn dept_is_eng() -> PredExpr {
    pb().binary_app(
        BinaryOp::Eq,
        pb().get_attr(pb().item(), "dept".into()),
        pb().val("eng"),
    )
    .unwrap()
}

fn has_set_term_error(errs: &HashSet<ValidationError>) -> bool {
    errs.iter()
        .any(|e| matches!(e, ValidationError::SetTermInQuantifier(_)))
}

#[test]
fn all_over_longs_is_bool() {
    assert_typechecks_empty_schema(&Expr::all(ints(&[1, 2]), it_pos()), &Type::primitive_boolean());
}

#[test]
fn any_over_longs_is_bool() {
    assert_typechecks_empty_schema(&Expr::any(ints(&[1, 2]), it_pos()), &Type::primitive_boolean());
}

#[test]
fn all_over_records_types_it_as_the_element() {
    let e = Expr::all(Expr::set([rec("dept", "eng"), rec("dept", "ops")]), dept_is_eng());
    assert_typechecks_empty_schema(&e, &Type::primitive_boolean());
}

#[test]
fn missing_attr_on_element_is_rejected() {
    let e = Expr::all(Expr::set([rec("team", "eng")]), dept_is_eng());
    let errs = assert_typecheck_fails(empty_schema_file(), &e, Some(&Type::primitive_boolean()));
    assert!(!errs.is_empty());
}

#[test]
fn has_guard_inside_predicate_typechecks() {
    let p = pb().and(pb().has_attr(pb().item(), "dept".into()), dept_is_eng());
    let e = Expr::all(Expr::set([rec("dept", "eng")]), p);
    assert_typechecks_empty_schema(&e, &Type::primitive_boolean());
}

#[test]
fn non_set_receiver_is_rejected() {
    let errs = assert_typecheck_fails(
        empty_schema_file(),
        &Expr::all(Expr::val(1), it_pos()),
        Some(&Type::primitive_boolean()),
    );
    assert!(!errs.is_empty());
}

#[test]
fn non_bool_predicate_is_rejected() {
    let errs = assert_typecheck_fails(
        empty_schema_file(),
        &Expr::all(ints(&[1]), pb().item()),
        Some(&Type::primitive_boolean()),
    );
    assert!(!errs.is_empty());
    assert!(!has_set_term_error(&errs));
}

#[test]
fn ill_typed_predicate_is_rejected() {
    // `"a" < it` over longs
    let p = pb()
        .binary_app(BinaryOp::Less, pb().val("a"), pb().item())
        .unwrap();
    let errs = assert_typecheck_fails(
        empty_schema_file(),
        &Expr::all(ints(&[1]), p),
        Some(&Type::primitive_boolean()),
    );
    assert!(!errs.is_empty());
}

#[test]
fn set_valued_it_is_a_set_term() {
    // requirement 1.5: `it` is itself a set
    let p = pb()
        .binary_app(BinaryOp::Eq, pb().item(), pb().item())
        .unwrap();
    let e = Expr::all(Expr::set([ints(&[1])]), p);
    let errs = assert_typecheck_fails(empty_schema_file(), &e, Some(&Type::primitive_boolean()));
    assert!(has_set_term_error(&errs), "{errs:?}");
}

#[test]
fn set_valued_attribute_is_a_set_term() {
    // requirement 1.5: a sub-expression of set type (`it.tags`)
    let p = pb()
        .binary_app(
            BinaryOp::Eq,
            pb().get_attr(pb().item(), "tags".into()),
            pb().get_attr(pb().item(), "tags".into()),
        )
        .unwrap();
    let elem = Expr::record([("tags".into(), ints(&[1]))]).unwrap();
    let e = Expr::all(Expr::set([elem]), p);
    let errs = assert_typecheck_fails(empty_schema_file(), &e, Some(&Type::primitive_boolean()));
    assert!(has_set_term_error(&errs), "{errs:?}");
}

#[test]
fn empty_set_receiver_in_permissive_mode() {
    // requirement 6.3: an unknown element type is handled like other set ops
    assert_typechecks_for_mode(
        empty_schema_file(),
        &Expr::all(Expr::set([]), it_pos()),
        &Type::primitive_boolean(),
        ValidationMode::Permissive,
    );
}

#[test]
fn empty_set_receiver_in_strict_mode_is_rejected() {
    let errs = assert_typecheck_fails_for_mode(
        empty_schema_file(),
        &Expr::all(Expr::set([]), it_pos()),
        Some(&Type::primitive_boolean()),
        ValidationMode::Strict,
    );
    assert!(errs
        .iter()
        .any(|e| matches!(e, ValidationError::EmptySetForbidden(_))));
}
