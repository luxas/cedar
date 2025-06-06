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

use super::{
    EntityUID, Expr, ExprKind, ExpressionConstructionError, Literal, Name, PartialValue, Type,
    Unknown, Value, ValueKind,
};
use crate::entities::json::err::JsonSerializationError;
use crate::extensions::Extensions;
use crate::parser::err::ParseErrors;
use crate::parser::{self, MaybeLoc};
use miette::Diagnostic;
use smol_str::{SmolStr, ToSmolStr};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;
use thiserror::Error;

/// A few places in Core use these "restricted expressions" (for lack of a
/// better term) which are in some sense the minimal subset of `Expr` required
/// to express all possible `Value`s.
///
/// Specifically, "restricted" expressions are
/// defined as expressions containing only the following:
///   - bool, int, and string literals
///   - literal EntityUIDs such as User::"alice"
///   - extension function calls, where the arguments must be other things
///     on this list
///   - set and record literals, where the values must be other things on
///     this list
///
/// That means the following are not allowed in "restricted" expressions:
///   - `principal`, `action`, `resource`, `context`
///   - builtin operators and functions, including `.`, `in`, `has`, `like`,
///     `.contains()`
///   - if-then-else expressions
///
/// These restrictions represent the expressions that are allowed to appear as
/// attribute values in `Slice` and `Context`.
#[derive(Hash, Debug, Clone, PartialEq, Eq)]
pub struct RestrictedExpr(Expr);

impl RestrictedExpr {
    /// Create a new `RestrictedExpr` from an `Expr`.
    ///
    /// This function is "safe" in the sense that it will verify that the
    /// provided `expr` does indeed qualify as a "restricted" expression,
    /// returning an error if not.
    ///
    /// Note this check requires recursively walking the AST. For a version of
    /// this function that doesn't perform this check, see `new_unchecked()`
    /// below.
    pub fn new(expr: Expr) -> Result<Self, RestrictedExpressionError> {
        is_restricted(&expr)?;
        Ok(Self(expr))
    }

    /// Create a new `RestrictedExpr` from an `Expr`, where the caller is
    /// responsible for ensuring that the `Expr` is a valid "restricted
    /// expression". If it is not, internal invariants will be violated, which
    /// may lead to other errors later, panics, or even incorrect results.
    ///
    /// For a "safer" version of this function that returns an error for invalid
    /// inputs, see `new()` above.
    pub fn new_unchecked(expr: Expr) -> Self {
        // in debug builds, this does the check anyway, panicking if it fails
        if cfg!(debug_assertions) {
            // PANIC SAFETY: We're in debug mode and panicking intentionally
            #[allow(clippy::unwrap_used)]
            Self::new(expr).unwrap()
        } else {
            Self(expr)
        }
    }

    /// Return the `RestrictedExpr`, but with the new `source_loc` (or `None`).
    pub fn with_maybe_source_loc(self, source_loc: MaybeLoc) -> Self {
        Self(self.0.with_maybe_source_loc(source_loc))
    }

    /// Create a `RestrictedExpr` that's just a single `Literal`.
    ///
    /// Note that you can pass this a `Literal`, an `Integer`, a `String`, etc.
    pub fn val(v: impl Into<Literal>) -> Self {
        // All literals are valid restricted-exprs
        Self::new_unchecked(Expr::val(v))
    }

    /// Create a `RestrictedExpr` that's just a single `Unknown`.
    pub fn unknown(u: Unknown) -> Self {
        // All unknowns are valid restricted-exprs
        Self::new_unchecked(Expr::unknown(u))
    }

    /// Create a `RestrictedExpr` which evaluates to a Set of the given `RestrictedExpr`s
    pub fn set(exprs: impl IntoIterator<Item = RestrictedExpr>) -> Self {
        // Set expressions are valid restricted-exprs if their elements are; and
        // we know the elements are because we require `RestrictedExpr`s in the
        // parameter
        Self::new_unchecked(Expr::set(exprs.into_iter().map(Into::into)))
    }

    /// Create a `RestrictedExpr` which evaluates to a Record with the given
    /// (key, value) pairs.
    ///
    /// Throws an error if any key occurs two or more times.
    pub fn record(
        pairs: impl IntoIterator<Item = (SmolStr, RestrictedExpr)>,
    ) -> Result<Self, ExpressionConstructionError> {
        // Record expressions are valid restricted-exprs if their elements are;
        // and we know the elements are because we require `RestrictedExpr`s in
        // the parameter
        Ok(Self::new_unchecked(Expr::record(
            pairs.into_iter().map(|(k, v)| (k, v.into())),
        )?))
    }

    /// Create a `RestrictedExpr` which calls the given extension function
    pub fn call_extension_fn(
        function_name: Name,
        args: impl IntoIterator<Item = RestrictedExpr>,
    ) -> Self {
        // Extension-function calls are valid restricted-exprs if their
        // arguments are; and we know the arguments are because we require
        // `RestrictedExpr`s in the parameter
        Self::new_unchecked(Expr::call_extension_fn(
            function_name,
            args.into_iter().map(Into::into).collect(),
        ))
    }

    /// Write a RestrictedExpr in "natural JSON" format.
    ///
    /// Used to output the context as a map from Strings to JSON Values
    pub fn to_natural_json(&self) -> Result<serde_json::Value, JsonSerializationError> {
        self.as_borrowed().to_natural_json()
    }

    /// Get the `bool` value of this `RestrictedExpr` if it's a boolean, or
    /// `None` if it is not a boolean
    pub fn as_bool(&self) -> Option<bool> {
        // the only way a `RestrictedExpr` can be a boolean is if it's a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Get the `i64` value of this `RestrictedExpr` if it's a long, or `None`
    /// if it is not a long
    pub fn as_long(&self) -> Option<i64> {
        // the only way a `RestrictedExpr` can be a long is if it's a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::Long(i)) => Some(*i),
            _ => None,
        }
    }

    /// Get the `SmolStr` value of this `RestrictedExpr` if it's a string, or
    /// `None` if it is not a string
    pub fn as_string(&self) -> Option<&SmolStr> {
        // the only way a `RestrictedExpr` can be a string is if it's a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get the `EntityUID` value of this `RestrictedExpr` if it's an entity
    /// reference, or `None` if it is not an entity reference
    pub fn as_euid(&self) -> Option<&EntityUID> {
        // the only way a `RestrictedExpr` can be an entity reference is if it's
        // a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::EntityUID(e)) => Some(e),
            _ => None,
        }
    }

    /// Get `Unknown` value of this `RestrictedExpr` if it's an `Unknown`, or
    /// `None` if it is not an `Unknown`
    pub fn as_unknown(&self) -> Option<&Unknown> {
        match self.expr_kind() {
            ExprKind::Unknown(u) => Some(u),
            _ => None,
        }
    }

    /// Iterate over the elements of the set if this `RestrictedExpr` is a set,
    /// or `None` if it is not a set
    pub fn as_set_elements(&self) -> Option<impl Iterator<Item = BorrowedRestrictedExpr<'_>>> {
        match self.expr_kind() {
            ExprKind::Set(set) => Some(set.iter().map(BorrowedRestrictedExpr::new_unchecked)), // since the RestrictedExpr invariant holds for the input set, it will hold for each element as well
            _ => None,
        }
    }

    /// Iterate over the (key, value) pairs of the record if this
    /// `RestrictedExpr` is a record, or `None` if it is not a record
    pub fn as_record_pairs(
        &self,
    ) -> Option<impl Iterator<Item = (&SmolStr, BorrowedRestrictedExpr<'_>)>> {
        match self.expr_kind() {
            ExprKind::Record(map) => Some(
                map.iter()
                    .map(|(k, v)| (k, BorrowedRestrictedExpr::new_unchecked(v))),
            ), // since the RestrictedExpr invariant holds for the input record, it will hold for each attr value as well
            _ => None,
        }
    }

    /// Get the name and args of the called extension function if this
    /// `RestrictedExpr` is an extension function call, or `None` if it is not
    /// an extension function call
    pub fn as_extn_fn_call(
        &self,
    ) -> Option<(&Name, impl Iterator<Item = BorrowedRestrictedExpr<'_>>)> {
        match self.expr_kind() {
            ExprKind::ExtensionFunctionApp { fn_name, args } => Some((
                fn_name,
                args.iter().map(BorrowedRestrictedExpr::new_unchecked),
            )), // since the RestrictedExpr invariant holds for the input call, it will hold for each argument as well
            _ => None,
        }
    }
}

impl From<Value> for RestrictedExpr {
    fn from(value: Value) -> RestrictedExpr {
        RestrictedExpr::from(value.value).with_maybe_source_loc(value.loc)
    }
}

impl From<ValueKind> for RestrictedExpr {
    fn from(value: ValueKind) -> RestrictedExpr {
        match value {
            ValueKind::Lit(lit) => RestrictedExpr::val(lit),
            ValueKind::Set(set) => {
                RestrictedExpr::set(set.iter().map(|val| RestrictedExpr::from(val.clone())))
            }
            // PANIC SAFETY: cannot have duplicate key because the input was already a BTreeMap
            #[allow(clippy::expect_used)]
            ValueKind::Record(record) => RestrictedExpr::record(
                Arc::unwrap_or_clone(record)
                    .into_iter()
                    .map(|(k, v)| (k, RestrictedExpr::from(v))),
            )
            .expect("can't have duplicate keys, because the input `map` was already a BTreeMap"),
            ValueKind::ExtensionValue(ev) => {
                let ev = Arc::unwrap_or_clone(ev);
                ev.into()
            }
        }
    }
}

impl TryFrom<PartialValue> for RestrictedExpr {
    type Error = PartialValueToRestrictedExprError;
    fn try_from(pvalue: PartialValue) -> Result<RestrictedExpr, PartialValueToRestrictedExprError> {
        match pvalue {
            PartialValue::Value(v) => Ok(RestrictedExpr::from(v)),
            PartialValue::Residual(expr) => match RestrictedExpr::new(expr) {
                Ok(e) => Ok(e),
                Err(RestrictedExpressionError::InvalidRestrictedExpression(
                    restricted_expr_errors::InvalidRestrictedExpressionError { expr, .. },
                )) => Err(PartialValueToRestrictedExprError::NontrivialResidual {
                    residual: Box::new(expr),
                }),
            },
        }
    }
}

/// Errors when converting `PartialValue` to `RestrictedExpr`
#[derive(Debug, PartialEq, Eq, Diagnostic, Error)]
pub enum PartialValueToRestrictedExprError {
    /// The `PartialValue` contains a nontrivial residual that isn't a valid `RestrictedExpr`
    #[error("residual is not a valid restricted expression: `{residual}`")]
    NontrivialResidual {
        /// Residual that isn't a valid `RestrictedExpr`
        residual: Box<Expr>,
    },
}

impl std::str::FromStr for RestrictedExpr {
    type Err = RestrictedExpressionParseError;

    fn from_str(s: &str) -> Result<RestrictedExpr, Self::Err> {
        parser::parse_restrictedexpr(s)
    }
}

/// While `RestrictedExpr` wraps an _owned_ `Expr`, `BorrowedRestrictedExpr`
/// wraps a _borrowed_ `Expr`, with the same invariants.
///
/// We derive `Copy` for this type because it's just a single reference, and
/// `&T` is `Copy` for all `T`.
#[derive(Hash, Debug, Clone, PartialEq, Eq, Copy)]
pub struct BorrowedRestrictedExpr<'a>(&'a Expr);

impl<'a> BorrowedRestrictedExpr<'a> {
    /// Create a new `BorrowedRestrictedExpr` from an `&Expr`.
    ///
    /// This function is "safe" in the sense that it will verify that the
    /// provided `expr` does indeed qualify as a "restricted" expression,
    /// returning an error if not.
    ///
    /// Note this check requires recursively walking the AST. For a version of
    /// this function that doesn't perform this check, see `new_unchecked()`
    /// below.
    pub fn new(expr: &'a Expr) -> Result<Self, RestrictedExpressionError> {
        is_restricted(expr)?;
        Ok(Self(expr))
    }

    /// Create a new `BorrowedRestrictedExpr` from an `&Expr`, where the caller
    /// is responsible for ensuring that the `Expr` is a valid "restricted
    /// expression". If it is not, internal invariants will be violated, which
    /// may lead to other errors later, panics, or even incorrect results.
    ///
    /// For a "safer" version of this function that returns an error for invalid
    /// inputs, see `new()` above.
    pub fn new_unchecked(expr: &'a Expr) -> Self {
        // in debug builds, this does the check anyway, panicking if it fails
        if cfg!(debug_assertions) {
            // PANIC SAFETY: We're in debug mode and panicking intentionally
            #[allow(clippy::unwrap_used)]
            Self::new(expr).unwrap()
        } else {
            Self(expr)
        }
    }

    /// Write a BorrowedRestrictedExpr in "natural JSON" format.
    ///
    /// Used to output the context as a map from Strings to JSON Values
    pub fn to_natural_json(self) -> Result<serde_json::Value, JsonSerializationError> {
        Ok(serde_json::to_value(
            crate::entities::json::CedarValueJson::from_expr(self)?,
        )?)
    }

    /// Convert `BorrowedRestrictedExpr` to `RestrictedExpr`.
    /// This has approximately the cost of cloning the `Expr`.
    pub fn to_owned(self) -> RestrictedExpr {
        RestrictedExpr::new_unchecked(self.0.clone())
    }

    /// Get the `bool` value of this `RestrictedExpr` if it's a boolean, or
    /// `None` if it is not a boolean
    pub fn as_bool(&self) -> Option<bool> {
        // the only way a `RestrictedExpr` can be a boolean is if it's a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Get the `i64` value of this `RestrictedExpr` if it's a long, or `None`
    /// if it is not a long
    pub fn as_long(&self) -> Option<i64> {
        // the only way a `RestrictedExpr` can be a long is if it's a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::Long(i)) => Some(*i),
            _ => None,
        }
    }

    /// Get the `SmolStr` value of this `RestrictedExpr` if it's a string, or
    /// `None` if it is not a string
    pub fn as_string(&self) -> Option<&SmolStr> {
        // the only way a `RestrictedExpr` can be a string is if it's a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get the `EntityUID` value of this `RestrictedExpr` if it's an entity
    /// reference, or `None` if it is not an entity reference
    pub fn as_euid(&self) -> Option<&EntityUID> {
        // the only way a `RestrictedExpr` can be an entity reference is if it's
        // a literal
        match self.expr_kind() {
            ExprKind::Lit(Literal::EntityUID(e)) => Some(e),
            _ => None,
        }
    }

    /// Get `Unknown` value of this `RestrictedExpr` if it's an `Unknown`, or
    /// `None` if it is not an `Unknown`
    pub fn as_unknown(&self) -> Option<&Unknown> {
        match self.expr_kind() {
            ExprKind::Unknown(u) => Some(u),
            _ => None,
        }
    }

    /// Iterate over the elements of the set if this `RestrictedExpr` is a set,
    /// or `None` if it is not a set
    pub fn as_set_elements(&self) -> Option<impl Iterator<Item = BorrowedRestrictedExpr<'_>>> {
        match self.expr_kind() {
            ExprKind::Set(set) => Some(set.iter().map(BorrowedRestrictedExpr::new_unchecked)), // since the RestrictedExpr invariant holds for the input set, it will hold for each element as well
            _ => None,
        }
    }

    /// Iterate over the (key, value) pairs of the record if this
    /// `RestrictedExpr` is a record, or `None` if it is not a record
    pub fn as_record_pairs(
        &self,
    ) -> Option<impl Iterator<Item = (&'_ SmolStr, BorrowedRestrictedExpr<'_>)>> {
        match self.expr_kind() {
            ExprKind::Record(map) => Some(
                map.iter()
                    .map(|(k, v)| (k, BorrowedRestrictedExpr::new_unchecked(v))),
            ), // since the RestrictedExpr invariant holds for the input record, it will hold for each attr value as well
            _ => None,
        }
    }

    /// Get the name and args of the called extension function if this
    /// `RestrictedExpr` is an extension function call, or `None` if it is not
    /// an extension function call
    pub fn as_extn_fn_call(
        &self,
    ) -> Option<(&Name, impl Iterator<Item = BorrowedRestrictedExpr<'_>>)> {
        match self.expr_kind() {
            ExprKind::ExtensionFunctionApp { fn_name, args } => Some((
                fn_name,
                args.iter().map(BorrowedRestrictedExpr::new_unchecked),
            )), // since the RestrictedExpr invariant holds for the input call, it will hold for each argument as well
            _ => None,
        }
    }

    /// Try to compute the runtime type of this expression. See
    /// [`Expr::try_type_of`] for exactly what this computes.
    ///
    /// On a restricted expression, there are fewer cases where we might fail to
    /// compute the type, but there are still `unknown`s and extension function
    /// calls which may cause this function to return `None` .
    pub fn try_type_of(&self, extensions: &Extensions<'_>) -> Option<Type> {
        self.0.try_type_of(extensions)
    }
}

/// Helper function: does the given `Expr` qualify as a "restricted" expression.
///
/// Returns `Ok(())` if yes, or a `RestrictedExpressionError` if no.
fn is_restricted(expr: &Expr) -> Result<(), RestrictedExpressionError> {
    match expr.expr_kind() {
        ExprKind::Lit(_) => Ok(()),
        ExprKind::Unknown(_) => Ok(()),
        ExprKind::Var(_) => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "variables".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::Slot(_) => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "template slots".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::If { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "if-then-else".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::And { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "&&".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::Or { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "||".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::UnaryApp { op, .. } => {
            Err(restricted_expr_errors::InvalidRestrictedExpressionError {
                feature: op.to_smolstr(),
                expr: expr.clone(),
            }
            .into())
        }
        ExprKind::BinaryApp { op, .. } => {
            Err(restricted_expr_errors::InvalidRestrictedExpressionError {
                feature: op.to_smolstr(),
                expr: expr.clone(),
            }
            .into())
        }
        ExprKind::GetAttr { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "attribute accesses".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::HasAttr { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "'has'".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::Like { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "'like'".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::Is { .. } => Err(restricted_expr_errors::InvalidRestrictedExpressionError {
            feature: "'is'".into(),
            expr: expr.clone(),
        }
        .into()),
        ExprKind::ExtensionFunctionApp { args, .. } => args.iter().try_for_each(is_restricted),
        ExprKind::Set(exprs) => exprs.iter().try_for_each(is_restricted),
        ExprKind::Record(map) => map.values().try_for_each(is_restricted),
        #[cfg(feature = "tolerant-ast")]
        ExprKind::Error { .. } => Ok(()),
    }
}

// converting into Expr is always safe; restricted exprs are always valid Exprs
impl From<RestrictedExpr> for Expr {
    fn from(r: RestrictedExpr) -> Expr {
        r.0
    }
}

impl AsRef<Expr> for RestrictedExpr {
    fn as_ref(&self) -> &Expr {
        &self.0
    }
}

impl Deref for RestrictedExpr {
    type Target = Expr;
    fn deref(&self) -> &Expr {
        self.as_ref()
    }
}

impl std::fmt::Display for RestrictedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

// converting into Expr is always safe; restricted exprs are always valid Exprs
impl<'a> From<BorrowedRestrictedExpr<'a>> for &'a Expr {
    fn from(r: BorrowedRestrictedExpr<'a>) -> &'a Expr {
        r.0
    }
}

impl<'a> AsRef<Expr> for BorrowedRestrictedExpr<'a> {
    fn as_ref(&self) -> &'a Expr {
        self.0
    }
}

impl RestrictedExpr {
    /// Turn an `&RestrictedExpr` into a `BorrowedRestrictedExpr`
    pub fn as_borrowed(&self) -> BorrowedRestrictedExpr<'_> {
        BorrowedRestrictedExpr::new_unchecked(self.as_ref())
    }
}

impl<'a> Deref for BorrowedRestrictedExpr<'a> {
    type Target = Expr;
    fn deref(&self) -> &'a Expr {
        self.0
    }
}

impl std::fmt::Display for BorrowedRestrictedExpr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

/// Like `ExprShapeOnly`, but for restricted expressions.
///
/// A newtype wrapper around (borrowed) restricted expressions that provides
/// `Eq` and `Hash` implementations that ignore any source information or other
/// generic data used to annotate the expression.
#[derive(Eq, Debug, Clone)]
pub struct RestrictedExprShapeOnly<'a>(BorrowedRestrictedExpr<'a>);

impl<'a> RestrictedExprShapeOnly<'a> {
    /// Construct a `RestrictedExprShapeOnly` from a `BorrowedRestrictedExpr`.
    /// The `BorrowedRestrictedExpr` is not modified, but any comparisons on the
    /// resulting `RestrictedExprShapeOnly` will ignore source information and
    /// generic data.
    pub fn new(e: BorrowedRestrictedExpr<'a>) -> RestrictedExprShapeOnly<'a> {
        RestrictedExprShapeOnly(e)
    }
}

impl PartialEq for RestrictedExprShapeOnly<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_shape(&other.0)
    }
}

impl Hash for RestrictedExprShapeOnly<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash_shape(state);
    }
}

/// Error when constructing a restricted expression from unrestricted
/// expression
//
// CAUTION: this type is publicly exported in `cedar-policy`.
// Don't make fields `pub`, don't make breaking changes, and use caution
// when adding public methods.
#[derive(Debug, Clone, PartialEq, Eq, Error, Diagnostic)]
pub enum RestrictedExpressionError {
    /// An expression was expected to be a "restricted" expression, but contained
    /// a feature that is not allowed in restricted expressions.
    #[error(transparent)]
    #[diagnostic(transparent)]
    InvalidRestrictedExpression(#[from] restricted_expr_errors::InvalidRestrictedExpressionError),
}

/// Error subtypes for [`RestrictedExpressionError`]
pub mod restricted_expr_errors {
    use super::Expr;
    use crate::impl_diagnostic_from_method_on_field;
    use miette::Diagnostic;
    use smol_str::SmolStr;
    use thiserror::Error;

    /// An expression was expected to be a "restricted" expression, but contained
    /// a feature that is not allowed in restricted expressions.
    //
    // CAUTION: this type is publicly exported in `cedar-policy`.
    // Don't make fields `pub`, don't make breaking changes, and use caution
    // when adding public methods.
    #[derive(Debug, Clone, PartialEq, Eq, Error)]
    #[error("not allowed to use {feature} in a restricted expression: `{expr}`")]
    pub struct InvalidRestrictedExpressionError {
        /// String description of what disallowed feature appeared in the expression
        pub(crate) feature: SmolStr,
        /// the (sub-)expression that uses the disallowed feature. This may be a
        /// sub-expression of a larger expression.
        pub(crate) expr: Expr,
    }

    // custom impl of `Diagnostic`: take source location from the `expr` field's `.source_loc()` method
    impl Diagnostic for InvalidRestrictedExpressionError {
        impl_diagnostic_from_method_on_field!(expr, source_loc);
    }
}

/// Errors possible from `RestrictedExpr::from_str()`
//
// This is NOT a publicly exported error type.
#[derive(Debug, Clone, PartialEq, Eq, Diagnostic, Error)]
pub enum RestrictedExpressionParseError {
    /// Failed to parse the expression
    #[error(transparent)]
    #[diagnostic(transparent)]
    Parse(#[from] ParseErrors),
    /// Parsed successfully as an expression, but failed to construct a
    /// restricted expression, for the reason indicated in the underlying error
    #[error(transparent)]
    #[diagnostic(transparent)]
    InvalidRestrictedExpression(#[from] RestrictedExpressionError),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ast::expression_construction_errors;
    use crate::parser::err::{ParseError, ToASTError, ToASTErrorKind};
    use crate::parser::{IntoMaybeLoc, Loc};
    use std::str::FromStr;
    use std::sync::Arc;

    #[test]
    fn duplicate_key() {
        // duplicate key is an error when mapped to values of different types
        assert_eq!(
            RestrictedExpr::record([
                ("foo".into(), RestrictedExpr::val(37),),
                ("foo".into(), RestrictedExpr::val("hello"),),
            ]),
            Err(expression_construction_errors::DuplicateKeyError {
                key: "foo".into(),
                context: "in record literal",
            }
            .into())
        );

        // duplicate key is an error when mapped to different values of same type
        assert_eq!(
            RestrictedExpr::record([
                ("foo".into(), RestrictedExpr::val(37),),
                ("foo".into(), RestrictedExpr::val(101),),
            ]),
            Err(expression_construction_errors::DuplicateKeyError {
                key: "foo".into(),
                context: "in record literal",
            }
            .into())
        );

        // duplicate key is an error when mapped to the same value multiple times
        assert_eq!(
            RestrictedExpr::record([
                ("foo".into(), RestrictedExpr::val(37),),
                ("foo".into(), RestrictedExpr::val(37),),
            ]),
            Err(expression_construction_errors::DuplicateKeyError {
                key: "foo".into(),
                context: "in record literal",
            }
            .into())
        );

        // duplicate key is an error even when other keys appear in between
        assert_eq!(
            RestrictedExpr::record([
                ("bar".into(), RestrictedExpr::val(-3),),
                ("foo".into(), RestrictedExpr::val(37),),
                ("spam".into(), RestrictedExpr::val("eggs"),),
                ("foo".into(), RestrictedExpr::val(37),),
                ("eggs".into(), RestrictedExpr::val("spam"),),
            ]),
            Err(expression_construction_errors::DuplicateKeyError {
                key: "foo".into(),
                context: "in record literal",
            }
            .into())
        );

        // duplicate key is also an error when parsing from string
        let str = r#"{ foo: 37, bar: "hi", foo: 101 }"#;
        assert_eq!(
            RestrictedExpr::from_str(str),
            Err(RestrictedExpressionParseError::Parse(
                ParseErrors::singleton(ParseError::ToAST(ToASTError::new(
                    ToASTErrorKind::ExpressionConstructionError(
                        expression_construction_errors::DuplicateKeyError {
                            key: "foo".into(),
                            context: "in record literal",
                        }
                        .into()
                    ),
                    Loc::new(0..32, Arc::from(str)).into_maybe_loc()
                )))
            )),
        )
    }
}
