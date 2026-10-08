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

#![cfg(feature = "anyall")]

//! Restricted predicate AST for the non-nested set quantifier `expr.all(pred)`.
//!
//! A [`PredExpr`] is a predicate over the current set element (the surface
//! keyword `it`, represented by [`PredExprKind::Item`]). It is a *restricted*
//! expression: by construction it can contain **no set terms** — no set
//! literals, no `contains`/`containsAll`/`containsAny`/`isEmpty`, and no nested
//! quantifier (`All`). That restriction is what makes `.all`/`.any` policies
//! analyzable (decidable) per Mohamed et al. (FMCAD 2025), Condition 1.
//!
//! The restrictions are enforced two ways:
//! - *Structurally* (unrepresentable): there is no `Set` variant, no nested
//!   `All`, and no `Slot`/`Unknown`.
//! - *By smart constructor*: [`PredExprBuilder`] is the only public way to
//!   build a [`PredExpr`], and it rejects the set-valued operators
//!   (`IsEmpty`, `Contains`, `ContainsAll`, `ContainsAny`).
//!
//! Everything in this module is `#[cfg(feature = "anyall")]`.

use crate::ast::{BinaryOp, EntityType, Literal, Name, Pattern, UnaryOp, Var};
use crate::parser::Loc;
use educe::Educe;
use miette::Diagnostic;
use nonempty::NonEmpty;
use smol_str::SmolStr;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;

/// A restricted predicate expression over the current set element `it`.
///
/// Mirrors [`crate::ast::Expr`]'s shape (an inner [`PredExprKind`], an optional
/// source location that is ignored for equality/hashing, and generic node
/// data), but can only represent the set-free, non-nested fragment.
#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct PredExpr<T = ()> {
    expr_kind: PredExprKind<T>,
    #[educe(PartialEq(ignore))]
    #[educe(Hash(ignore))]
    source_loc: Option<Loc>,
    data: T,
}

/// The predicate variants. Deliberately excludes `Set`, nested `All`, `Slot`,
/// and `Unknown` so set-freeness and non-nesting are structural.
#[derive(Hash, Debug, Clone, PartialEq, Eq)]
pub enum PredExprKind<T = ()> {
    /// The current set element — surface keyword `it`. The only new leaf.
    Item,
    /// A literal. [`Literal`] is already set-free (set literals are always
    /// `ExprKind::Set`, never a `Literal`), so it is reused directly with no
    /// wrapper type.
    Lit(Literal),
    /// A variable (`principal`, `action`, `resource`, `context`).
    Var(Var),
    /// Ternary expression.
    If {
        /// Condition. Must evaluate to Bool.
        test_expr: Arc<PredExpr<T>>,
        /// Value if true.
        then_expr: Arc<PredExpr<T>>,
        /// Value if false.
        else_expr: Arc<PredExpr<T>>,
    },
    /// Boolean AND (short-circuiting left-to-right).
    And {
        /// Left operand.
        left: Arc<PredExpr<T>>,
        /// Right operand.
        right: Arc<PredExpr<T>>,
    },
    /// Boolean OR (short-circuiting left-to-right).
    Or {
        /// Left operand.
        left: Arc<PredExpr<T>>,
        /// Right operand.
        right: Arc<PredExpr<T>>,
    },
    /// Built-in unary operator. `op` is never [`UnaryOp::IsEmpty`] (rejected by
    /// the smart constructor — `isEmpty` is a set term).
    UnaryApp {
        /// Unary operator. Never `IsEmpty`.
        op: UnaryOp,
        /// Argument.
        arg: Arc<PredExpr<T>>,
    },
    /// Built-in binary operator. `op` is never `Contains`/`ContainsAll`/
    /// `ContainsAny` (rejected by the smart constructor — those are set terms).
    BinaryApp {
        /// Binary operator. Never a `Contains*` op.
        op: BinaryOp,
        /// First argument.
        arg1: Arc<PredExpr<T>>,
        /// Second argument.
        arg2: Arc<PredExpr<T>>,
    },
    /// Extension function application.
    ExtensionFunctionApp {
        /// Function name.
        fn_name: Name,
        /// Arguments.
        args: Arc<Vec<PredExpr<T>>>,
    },
    /// Get an attribute/field.
    GetAttr {
        /// Receiver. Must evaluate to Entity or Record.
        expr: Arc<PredExpr<T>>,
        /// Attribute/field name.
        attr: SmolStr,
    },
    /// `has` an attribute/field.
    HasAttr {
        /// Receiver. Must evaluate to Entity or Record.
        expr: Arc<PredExpr<T>>,
        /// Attribute/field name.
        attr: SmolStr,
    },
    /// `has` a nested sequence of attributes.
    ExtHasAttr {
        /// Receiver. Must evaluate to Entity or Record.
        expr: Arc<PredExpr<T>>,
        /// Attribute sequence.
        attrs: NonEmpty<SmolStr>,
    },
    /// Regex-like string matching.
    Like {
        /// Receiver. Must evaluate to String.
        expr: Arc<PredExpr<T>>,
        /// Pattern.
        pattern: Pattern,
    },
    /// Entity-type test.
    Is {
        /// Receiver. Must evaluate to an Entity.
        expr: Arc<PredExpr<T>>,
        /// Entity type tested against.
        entity_type: EntityType,
    },
    /// Anonymous record. Records are allowed in a predicate; sets are not.
    Record(Arc<BTreeMap<SmolStr, PredExpr<T>>>),
}

/// Error returned by a [`PredExprBuilder`] smart constructor when the requested
/// operator is a set term and therefore not allowed inside an analyzable
/// predicate.
#[derive(Debug, Clone, PartialEq, Eq, Error, Diagnostic)]
pub enum PredConstructionError {
    /// A set-valued operator was used in a predicate.
    #[error("the operator `{op}` is a set operator and is not allowed in a `.all`/`.any` predicate")]
    SetOperator {
        /// The rejected operator, rendered via its `Display`.
        op: SmolStr,
    },
}

impl<T> PredExprKind<T> {
    /// Variant order, mirroring the derived `Ord` on a fieldless enum. Used by
    /// [`PredExpr::cmp_shape`]. The numbering is independent of
    /// [`crate::ast::ExprKind::variant_order`] — this is a different type.
    fn variant_order(&self) -> u8 {
        match self {
            PredExprKind::Item => 0,
            PredExprKind::Lit(_) => 1,
            PredExprKind::Var(_) => 2,
            PredExprKind::If { .. } => 3,
            PredExprKind::And { .. } => 4,
            PredExprKind::Or { .. } => 5,
            PredExprKind::UnaryApp { .. } => 6,
            PredExprKind::BinaryApp { .. } => 7,
            PredExprKind::ExtensionFunctionApp { .. } => 8,
            PredExprKind::GetAttr { .. } => 9,
            PredExprKind::HasAttr { .. } => 10,
            PredExprKind::ExtHasAttr { .. } => 11,
            PredExprKind::Like { .. } => 12,
            PredExprKind::Is { .. } => 13,
            PredExprKind::Record(_) => 14,
        }
    }
}

impl<T> PredExpr<T> {
    /// The inner [`PredExprKind`].
    pub fn expr_kind(&self) -> &PredExprKind<T> {
        &self.expr_kind
    }

    /// Consume and return the inner [`PredExprKind`].
    pub fn into_expr_kind(self) -> PredExprKind<T> {
        self.expr_kind
    }

    /// The source location, if any.
    pub fn source_loc(&self) -> Option<&Loc> {
        self.source_loc.as_ref()
    }

    /// The node data.
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Logically negate this predicate: wrap it in `!(..)`.
    ///
    /// This is a plain structural wrapper (it does *not* simplify
    /// double-negation), matching [`crate::expr_builder::ExprBuilder::not`].
    /// It is used by the `any` lowering `expr.any(p) == !expr.all(!p)`.
    /// `Not` is always an allowed predicate operator, so no check is needed.
    pub fn negate(self) -> PredExpr<T>
    where
        T: Default,
    {
        PredExpr {
            expr_kind: PredExprKind::UnaryApp {
                op: UnaryOp::Not,
                arg: Arc::new(self),
            },
            source_loc: None,
            data: T::default(),
        }
    }

    /// Structural equality ignoring source locations and node data — the
    /// predicate analogue of [`crate::ast::Expr::eq_shape`].
    pub fn eq_shape<U>(&self, other: &PredExpr<U>) -> bool {
        use PredExprKind::*;
        match (self.expr_kind(), other.expr_kind()) {
            (Item, Item) => true,
            (Lit(l), Lit(l1)) => l == l1,
            (Var(v), Var(v1)) => v == v1,
            (
                If {
                    test_expr: t,
                    then_expr: th,
                    else_expr: e,
                },
                If {
                    test_expr: t1,
                    then_expr: th1,
                    else_expr: e1,
                },
            ) => t.eq_shape(t1) && th.eq_shape(th1) && e.eq_shape(e1),
            (And { left: l, right: r }, And { left: l1, right: r1 })
            | (Or { left: l, right: r }, Or { left: l1, right: r1 }) => {
                l.eq_shape(l1) && r.eq_shape(r1)
            }
            (UnaryApp { op, arg }, UnaryApp { op: op1, arg: arg1 }) => {
                op == op1 && arg.eq_shape(arg1)
            }
            (
                BinaryApp { op, arg1, arg2 },
                BinaryApp {
                    op: op1,
                    arg1: arg11,
                    arg2: arg21,
                },
            ) => op == op1 && arg1.eq_shape(arg11) && arg2.eq_shape(arg21),
            (
                ExtensionFunctionApp { fn_name, args },
                ExtensionFunctionApp {
                    fn_name: fn_name1,
                    args: args1,
                },
            ) => {
                fn_name == fn_name1
                    && args.len() == args1.len()
                    && args.iter().zip(args1.iter()).all(|(a, a1)| a.eq_shape(a1))
            }
            (GetAttr { expr, attr }, GetAttr { expr: expr1, attr: attr1 })
            | (HasAttr { expr, attr }, HasAttr { expr: expr1, attr: attr1 }) => {
                attr == attr1 && expr.eq_shape(expr1)
            }
            (
                ExtHasAttr { expr, attrs },
                ExtHasAttr {
                    expr: expr1,
                    attrs: attrs1,
                },
            ) => attrs == attrs1 && expr.eq_shape(expr1),
            (Like { expr, pattern }, Like { expr: expr1, pattern: pattern1 }) => {
                pattern == pattern1 && expr.eq_shape(expr1)
            }
            (
                Is { expr, entity_type },
                Is {
                    expr: expr1,
                    entity_type: entity_type1,
                },
            ) => entity_type == entity_type1 && expr.eq_shape(expr1),
            (Record(m), Record(m1)) => {
                m.len() == m1.len()
                    && m.iter().zip(m1.iter()).all(|((k, v), (k1, v1))| {
                        k == k1 && v.eq_shape(v1)
                    })
            }
            _ => false,
        }
    }

    /// Hash the shape, ignoring source locations and node data — the predicate
    /// analogue of [`crate::ast::Expr::hash_shape`].
    pub fn hash_shape<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.expr_kind().variant_order().hash(state);
        use PredExprKind::*;
        match self.expr_kind() {
            Item => {}
            Lit(l) => l.hash(state),
            Var(v) => v.hash(state),
            If {
                test_expr,
                then_expr,
                else_expr,
            } => {
                test_expr.hash_shape(state);
                then_expr.hash_shape(state);
                else_expr.hash_shape(state);
            }
            And { left, right } | Or { left, right } => {
                left.hash_shape(state);
                right.hash_shape(state);
            }
            UnaryApp { op, arg } => {
                op.hash(state);
                arg.hash_shape(state);
            }
            BinaryApp { op, arg1, arg2 } => {
                op.hash(state);
                arg1.hash_shape(state);
                arg2.hash_shape(state);
            }
            ExtensionFunctionApp { fn_name, args } => {
                fn_name.hash(state);
                state.write_usize(args.len());
                args.iter().for_each(|a| a.hash_shape(state));
            }
            GetAttr { expr, attr } | HasAttr { expr, attr } => {
                attr.hash(state);
                expr.hash_shape(state);
            }
            ExtHasAttr { expr, attrs } => {
                attrs.hash(state);
                expr.hash_shape(state);
            }
            Like { expr, pattern } => {
                pattern.hash(state);
                expr.hash_shape(state);
            }
            Is { expr, entity_type } => {
                entity_type.hash(state);
                expr.hash_shape(state);
            }
            Record(m) => {
                state.write_usize(m.len());
                m.iter().for_each(|(k, v)| {
                    k.hash(state);
                    v.hash_shape(state);
                });
            }
        }
    }

    /// Compare the shape, ignoring source locations and node data — the
    /// predicate analogue of [`crate::ast::Expr::cmp_shape`].
    pub fn cmp_shape(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use PredExprKind::*;
        match (self.expr_kind(), other.expr_kind()) {
            (Item, Item) => Ordering::Equal,
            (Lit(l), Lit(l1)) => l.cmp(l1),
            (Var(v), Var(v1)) => v.cmp(v1),
            (
                If {
                    test_expr: t,
                    then_expr: th,
                    else_expr: e,
                },
                If {
                    test_expr: t1,
                    then_expr: th1,
                    else_expr: e1,
                },
            ) => t
                .cmp_shape(t1)
                .then_with(|| th.cmp_shape(th1))
                .then_with(|| e.cmp_shape(e1)),
            (And { left: l, right: r }, And { left: l1, right: r1 })
            | (Or { left: l, right: r }, Or { left: l1, right: r1 }) => {
                l.cmp_shape(l1).then_with(|| r.cmp_shape(r1))
            }
            (UnaryApp { op, arg }, UnaryApp { op: op1, arg: arg1 }) => {
                op.cmp(op1).then_with(|| arg.cmp_shape(arg1))
            }
            (
                BinaryApp { op, arg1, arg2 },
                BinaryApp {
                    op: op1,
                    arg1: arg11,
                    arg2: arg21,
                },
            ) => op
                .cmp(op1)
                .then_with(|| arg1.cmp_shape(arg11))
                .then_with(|| arg2.cmp_shape(arg21)),
            (
                ExtensionFunctionApp { fn_name, args },
                ExtensionFunctionApp {
                    fn_name: fn_name1,
                    args: args1,
                },
            ) => fn_name.cmp(fn_name1).then_with(|| {
                args.len().cmp(&args1.len()).then_with(|| {
                    args.iter()
                        .zip(args1.iter())
                        .map(|(a, a1)| a.cmp_shape(a1))
                        .find(|o| *o != Ordering::Equal)
                        .unwrap_or(Ordering::Equal)
                })
            }),
            (GetAttr { expr, attr }, GetAttr { expr: expr1, attr: attr1 })
            | (HasAttr { expr, attr }, HasAttr { expr: expr1, attr: attr1 }) => {
                attr.cmp(attr1).then_with(|| expr.cmp_shape(expr1))
            }
            (
                ExtHasAttr { expr, attrs },
                ExtHasAttr {
                    expr: expr1,
                    attrs: attrs1,
                },
            ) => attrs
                .iter()
                .cmp(attrs1.iter())
                .then_with(|| expr.cmp_shape(expr1)),
            (Like { expr, pattern }, Like { expr: expr1, pattern: pattern1 }) => {
                pattern.cmp(pattern1).then_with(|| expr.cmp_shape(expr1))
            }
            (
                Is { expr, entity_type },
                Is {
                    expr: expr1,
                    entity_type: entity_type1,
                },
            ) => entity_type
                .cmp(entity_type1)
                .then_with(|| expr.cmp_shape(expr1)),
            (Record(m), Record(m1)) => m
                .len()
                .cmp(&m1.len())
                .then_with(|| {
                    m.keys()
                        .cmp(m1.keys())
                })
                .then_with(|| {
                    m.values()
                        .zip(m1.values())
                        .map(|(v, v1)| v.cmp_shape(v1))
                        .find(|o| *o != Ordering::Equal)
                        .unwrap_or(Ordering::Equal)
                }),
            // Different variants: order by variant_order.
            (l, r) => l.variant_order().cmp(&r.variant_order()),
        }
    }
}

impl<T: Clone> PredExpr<T> {
    /// Rebuild this predicate with fresh default node data of type `U`,
    /// discarding the current data `T`. Source locations are preserved.
    ///
    /// Used by [`crate::expr_builder::ExprBuilder::pred_from_ast`] so an `All`
    /// node's predicate can cross a `try_into_expr::<B>` conversion that
    /// changes the node-data type (every other arm likewise rebuilds with the
    /// target builder's data).
    pub fn with_default_data<U: Default + Clone>(&self) -> PredExpr<U> {
        use PredExprKind::*;
        let rebuild = |k: PredExprKind<U>| PredExpr {
            expr_kind: k,
            source_loc: self.source_loc.clone(),
            data: U::default(),
        };
        match self.expr_kind() {
            Item => rebuild(Item),
            Lit(l) => rebuild(Lit(l.clone())),
            Var(v) => rebuild(Var(*v)),
            If {
                test_expr,
                then_expr,
                else_expr,
            } => rebuild(If {
                test_expr: Arc::new(test_expr.with_default_data()),
                then_expr: Arc::new(then_expr.with_default_data()),
                else_expr: Arc::new(else_expr.with_default_data()),
            }),
            And { left, right } => rebuild(And {
                left: Arc::new(left.with_default_data()),
                right: Arc::new(right.with_default_data()),
            }),
            Or { left, right } => rebuild(Or {
                left: Arc::new(left.with_default_data()),
                right: Arc::new(right.with_default_data()),
            }),
            UnaryApp { op, arg } => rebuild(UnaryApp {
                op: *op,
                arg: Arc::new(arg.with_default_data()),
            }),
            BinaryApp { op, arg1, arg2 } => rebuild(BinaryApp {
                op: *op,
                arg1: Arc::new(arg1.with_default_data()),
                arg2: Arc::new(arg2.with_default_data()),
            }),
            ExtensionFunctionApp { fn_name, args } => rebuild(ExtensionFunctionApp {
                fn_name: fn_name.clone(),
                args: Arc::new(args.iter().map(|a| a.with_default_data()).collect()),
            }),
            GetAttr { expr, attr } => rebuild(GetAttr {
                expr: Arc::new(expr.with_default_data()),
                attr: attr.clone(),
            }),
            HasAttr { expr, attr } => rebuild(HasAttr {
                expr: Arc::new(expr.with_default_data()),
                attr: attr.clone(),
            }),
            ExtHasAttr { expr, attrs } => rebuild(ExtHasAttr {
                expr: Arc::new(expr.with_default_data()),
                attrs: attrs.clone(),
            }),
            Like { expr, pattern } => rebuild(Like {
                expr: Arc::new(expr.with_default_data()),
                pattern: pattern.clone(),
            }),
            Is { expr, entity_type } => rebuild(Is {
                expr: Arc::new(expr.with_default_data()),
                entity_type: entity_type.clone(),
            }),
            Record(m) => rebuild(Record(Arc::new(
                m.iter()
                    .map(|(k, v)| (k.clone(), v.with_default_data()))
                    .collect(),
            ))),
        }
    }

    /// Substitute unknowns by name, recursing into the predicate structure.
    ///
    /// Phase 1 only needs this so [`crate::ast::Expr::substitute_general`] can
    /// recurse through an `All` node's predicate without dropping structure.
    /// A `PredExpr` has no `Unknown` leaf, so substitution never replaces a
    /// node here; it only rebuilds the tree. Kept for structural completeness
    /// and to mirror `Expr::substitute_general`.
    pub fn substitute_general<F>(&self, _mapping: &F) -> PredExpr<T>
    where
        F: Fn(&SmolStr) -> Option<crate::ast::Value>,
    {
        // No `Unknown`/`Item`-rebinding substitution happens in a predicate; it
        // is a structural clone. (Element binding happens at evaluation time in
        // Phase 2/3, not via name substitution.)
        self.clone()
    }
}

impl PredExpr {
    /// Instantiate this predicate for one set element: every `it`
    /// ([`PredExprKind::Item`]) becomes the element, embedded as an [`Expr`]
    /// via `From<Value> for Expr`, and every other node maps to its `ExprKind`
    /// counterpart. Evaluating the result is evaluating the predicate with
    /// `it` bound to `elem`.
    pub fn instantiate(&self, elem: &crate::ast::Value) -> crate::ast::Expr {
        use crate::ast::Expr;
        use PredExprKind::*;
        match self.expr_kind() {
            Item => Expr::from(elem.clone()),
            Lit(l) => Expr::val(l.clone()),
            Var(v) => Expr::var(*v),
            If {
                test_expr,
                then_expr,
                else_expr,
            } => Expr::ite(
                test_expr.instantiate(elem),
                then_expr.instantiate(elem),
                else_expr.instantiate(elem),
            ),
            And { left, right } => Expr::and(left.instantiate(elem), right.instantiate(elem)),
            Or { left, right } => Expr::or(left.instantiate(elem), right.instantiate(elem)),
            UnaryApp { op, arg } => Expr::unary_app(*op, arg.instantiate(elem)),
            BinaryApp { op, arg1, arg2 } => {
                Expr::binary_app(*op, arg1.instantiate(elem), arg2.instantiate(elem))
            }
            ExtensionFunctionApp { fn_name, args } => Expr::call_extension_fn(
                fn_name.clone(),
                args.iter().map(|a| a.instantiate(elem)).collect(),
            ),
            GetAttr { expr, attr } => Expr::get_attr(expr.instantiate(elem), attr.clone()),
            HasAttr { expr, attr } => Expr::has_attr(expr.instantiate(elem), attr.clone()),
            ExtHasAttr { expr, attrs } => {
                Expr::extended_has_attr(expr.instantiate(elem), attrs.clone())
            }
            Like { expr, pattern } => Expr::like(expr.instantiate(elem), pattern.clone()),
            Is { expr, entity_type } => {
                Expr::is_entity_type(expr.instantiate(elem), entity_type.clone())
            }
            #[expect(
                clippy::expect_used,
                reason = "the keys come from a BTreeMap, so there are no duplicates"
            )]
            Record(map) => Expr::record(map.iter().map(|(k, v)| (k.clone(), v.instantiate(elem))))
                .expect("the keys come from a BTreeMap, so there are no duplicates"),
        }
    }
}

/// The only public way to build a [`PredExpr`]. Smart constructors reject the
/// set-valued operators so an analyzable predicate cannot be built with one.
#[derive(Clone, Debug)]
pub struct PredExprBuilder<T = ()> {
    source_loc: Option<Loc>,
    data: T,
}

impl<T: Default> Default for PredExprBuilder<T> {
    fn default() -> Self {
        Self {
            source_loc: None,
            data: T::default(),
        }
    }
}

impl<T: Default + Clone> PredExprBuilder<T> {
    /// A new builder carrying no data.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a source location.
    pub fn with_maybe_source_loc(mut self, l: Option<&Loc>) -> Self {
        self.source_loc = l.cloned();
        self
    }

    fn build(&self, expr_kind: PredExprKind<T>) -> PredExpr<T> {
        PredExpr {
            expr_kind,
            source_loc: self.source_loc.clone(),
            data: self.data.clone(),
        }
    }

    /// The current element `it`.
    pub fn item(self) -> PredExpr<T> {
        self.build(PredExprKind::Item)
    }

    /// A literal.
    pub fn val(self, v: impl Into<Literal>) -> PredExpr<T> {
        self.build(PredExprKind::Lit(v.into()))
    }

    /// A variable.
    pub fn var(self, v: Var) -> PredExpr<T> {
        self.build(PredExprKind::Var(v))
    }

    /// Ternary.
    pub fn ite(self, test_expr: PredExpr<T>, then_expr: PredExpr<T>, else_expr: PredExpr<T>) -> PredExpr<T> {
        self.build(PredExprKind::If {
            test_expr: Arc::new(test_expr),
            then_expr: Arc::new(then_expr),
            else_expr: Arc::new(else_expr),
        })
    }

    /// Boolean AND.
    pub fn and(self, left: PredExpr<T>, right: PredExpr<T>) -> PredExpr<T> {
        self.build(PredExprKind::And {
            left: Arc::new(left),
            right: Arc::new(right),
        })
    }

    /// Boolean OR.
    pub fn or(self, left: PredExpr<T>, right: PredExpr<T>) -> PredExpr<T> {
        self.build(PredExprKind::Or {
            left: Arc::new(left),
            right: Arc::new(right),
        })
    }

    /// Logical negation (always allowed).
    pub fn not(self, arg: PredExpr<T>) -> PredExpr<T> {
        self.build(PredExprKind::UnaryApp {
            op: UnaryOp::Not,
            arg: Arc::new(arg),
        })
    }

    /// Apply a unary operator. Rejects [`UnaryOp::IsEmpty`] (a set term).
    pub fn unary_app(self, op: UnaryOp, arg: PredExpr<T>) -> Result<PredExpr<T>, PredConstructionError> {
        if matches!(op, UnaryOp::IsEmpty) {
            return Err(PredConstructionError::SetOperator {
                op: op.to_string().into(),
            });
        }
        Ok(self.build(PredExprKind::UnaryApp {
            op,
            arg: Arc::new(arg),
        }))
    }

    /// Apply a binary operator. Rejects `Contains`/`ContainsAll`/`ContainsAny`
    /// (set terms). `In` is allowed structurally here — whether a particular
    /// `In` whose RHS is a set counts as a "set term" is a validator/typecheck
    /// concern handled in Phase 3, not a construction-time structural reject.
    pub fn binary_app(
        self,
        op: BinaryOp,
        arg1: PredExpr<T>,
        arg2: PredExpr<T>,
    ) -> Result<PredExpr<T>, PredConstructionError> {
        if matches!(
            op,
            BinaryOp::Contains | BinaryOp::ContainsAll | BinaryOp::ContainsAny
        ) {
            return Err(PredConstructionError::SetOperator {
                op: op.to_string().into(),
            });
        }
        Ok(self.build(PredExprKind::BinaryApp {
            op,
            arg1: Arc::new(arg1),
            arg2: Arc::new(arg2),
        }))
    }

    /// Get an attribute/field.
    pub fn get_attr(self, expr: PredExpr<T>, attr: SmolStr) -> PredExpr<T> {
        self.build(PredExprKind::GetAttr {
            expr: Arc::new(expr),
            attr,
        })
    }

    /// `has` an attribute/field.
    pub fn has_attr(self, expr: PredExpr<T>, attr: SmolStr) -> PredExpr<T> {
        self.build(PredExprKind::HasAttr {
            expr: Arc::new(expr),
            attr,
        })
    }

    /// `like` pattern match.
    pub fn like(self, expr: PredExpr<T>, pattern: Pattern) -> PredExpr<T> {
        self.build(PredExprKind::Like {
            expr: Arc::new(expr),
            pattern,
        })
    }

    /// `is` entity-type test.
    pub fn is_entity_type(self, expr: PredExpr<T>, entity_type: EntityType) -> PredExpr<T> {
        self.build(PredExprKind::Is {
            expr: Arc::new(expr),
            entity_type,
        })
    }

    /// A record.
    pub fn record(self, pairs: impl IntoIterator<Item = (SmolStr, PredExpr<T>)>) -> PredExpr<T> {
        self.build(PredExprKind::Record(Arc::new(pairs.into_iter().collect())))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn pb() -> PredExprBuilder {
        PredExprBuilder::new()
    }

    fn hash_of(p: &PredExpr) -> u64 {
        let mut h = DefaultHasher::new();
        p.hash_shape(&mut h);
        h.finish()
    }

    #[test]
    fn item_and_lit_build() {
        let it = pb().item();
        assert!(matches!(it.expr_kind(), PredExprKind::Item));
        let lit = pb().val(1i64);
        assert!(matches!(lit.expr_kind(), PredExprKind::Lit(Literal::Long(1))));
    }

    #[test]
    fn pred_negate_roundtrip() {
        // negate wraps in Not, and does NOT simplify double negation
        let it = pb().item();
        let neg = it.clone().negate();
        match neg.expr_kind() {
            PredExprKind::UnaryApp { op: UnaryOp::Not, arg } => {
                assert!(arg.eq_shape(&it));
            }
            other => panic!("expected Not(it), got {other:?}"),
        }
        let neg2 = neg.clone().negate();
        // Not(Not(it)) — two layers, unsimplified
        match neg2.expr_kind() {
            PredExprKind::UnaryApp { op: UnaryOp::Not, arg } => {
                assert!(arg.eq_shape(&neg));
            }
            other => panic!("expected Not(Not(it)), got {other:?}"),
        }
    }

    #[test]
    fn smart_ctor_rejects_is_empty() {
        let err = pb().unary_app(UnaryOp::IsEmpty, pb().item());
        assert!(matches!(err, Err(PredConstructionError::SetOperator { .. })));
        // but Not and Neg are fine
        assert!(pb().unary_app(UnaryOp::Not, pb().item()).is_ok());
        assert!(pb().unary_app(UnaryOp::Neg, pb().item()).is_ok());
    }

    #[test]
    fn smart_ctor_rejects_contains_family() {
        for op in [BinaryOp::Contains, BinaryOp::ContainsAll, BinaryOp::ContainsAny] {
            let err = pb().binary_app(op, pb().item(), pb().val(1i64));
            assert!(
                matches!(err, Err(PredConstructionError::SetOperator { .. })),
                "{op} should be rejected"
            );
        }
        // positive sanity: Eq, Less, In are allowed
        for op in [BinaryOp::Eq, BinaryOp::Less, BinaryOp::In] {
            assert!(
                pb().binary_app(op, pb().item(), pb().val(1i64)).is_ok(),
                "{op} should be allowed"
            );
        }
    }

    #[test]
    fn shape_eq_hash_cmp() {
        // it == 1
        let p1 = pb()
            .binary_app(BinaryOp::Eq, pb().item(), pb().val(1i64))
            .unwrap();
        let p2 = pb()
            .binary_app(BinaryOp::Eq, pb().item(), pb().val(1i64))
            .unwrap();
        // it == 2
        let p3 = pb()
            .binary_app(BinaryOp::Eq, pb().item(), pb().val(2i64))
            .unwrap();
        assert!(p1.eq_shape(&p2));
        assert_eq!(hash_of(&p1), hash_of(&p2));
        assert!(!p1.eq_shape(&p3));
        assert_eq!(p1.cmp_shape(&p2), std::cmp::Ordering::Equal);
        assert_ne!(p1.cmp_shape(&p3), std::cmp::Ordering::Equal);
        // cmp is a total order consistent with itself
        assert_eq!(p1.cmp_shape(&p3), p1.cmp_shape(&p3));
        assert_eq!(p3.cmp_shape(&p1), p1.cmp_shape(&p3).reverse());
    }

    #[test]
    fn record_and_getattr_build() {
        // it.department
        let dept = pb().get_attr(pb().item(), "department".into());
        assert!(matches!(dept.expr_kind(), PredExprKind::GetAttr { .. }));
        // { "k": it }
        let rec = pb().record([("k".into(), pb().item())]);
        match rec.expr_kind() {
            PredExprKind::Record(m) => assert_eq!(m.len(), 1),
            other => panic!("expected record, got {other:?}"),
        }
    }
}
