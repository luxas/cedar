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

//! F-4: cvc5-backed proof that the SymCC solve path works end-to-end on a
//! SYMBOLIC `.all` over a set-typed context attribute. These tests require the
//! `CVC5` env var to point at a cvc5 binary (as every other integration test
//! here does). They exercise the production solve path, which now emits
//! `(set-logic HO_ALL)` (matching Lean `Encoder.lean`) so cvc5 accepts the
//! two-`set.filter`-lambda encoding rather than rejecting the HO construct.
//!
//! We call `check_equivalent` DIRECTLY (the un-optimized path): the SymCCOpt
//! path still rejects `.all` under the D-77 interim, so the combined
//! `assert_equivalent` helper (which runs both) is not usable here.
#![cfg(feature = "anyall")]

use cedar_policy::{PolicySet, Schema, Validator};
use cedar_policy_symcc::{solver::LocalSolver, CedarSymCompiler, SymEnv, WellTypedPolicies};

mod utils;
use utils::{pset_from_text, schema_from_cedarstr, Environments};

fn schema() -> Schema {
    schema_from_cedarstr(
        r#"
        entity User;
        entity Thing;
        action view appliesTo {
          principal: [User],
          resource: [Thing],
          context: { ports: Set<Long> }
        };
    "#,
    )
}

/// Compile two policy sets and run the UN-OPTIMIZED `check_equivalent` against
/// cvc5; returns whether they are equivalent. Proves cvc5 does not error on the
/// `HO_ALL` logic + `set.filter` encoding of a symbolic `.all`.
async fn equivalent(pset1_text: &str, pset2_text: &str) -> bool {
    let validator = Validator::new(schema());
    let pset1: PolicySet = pset_from_text(pset1_text, &validator);
    let pset2: PolicySet = pset_from_text(pset2_text, &validator);
    let envs = Environments::new(validator.schema(), "User", "Action::\"view\"", "Thing");
    let symenv: &SymEnv = &envs.symenv;
    let typed1 = WellTypedPolicies::from_policies(&pset1, &envs.req_env, envs.schema).unwrap();
    let typed2 = WellTypedPolicies::from_policies(&pset2, &envs.req_env, envs.schema).unwrap();
    let mut compiler = CedarSymCompiler::new(LocalSolver::cvc5().unwrap()).unwrap();
    compiler
        .check_equivalent(&typed1, &typed2, symenv)
        .await
        .expect("cvc5 should solve the symbolic `.all` query under HO_ALL, not error on the logic")
}

// Two policies with the SAME symbolic `.all` guard are equivalent (cvc5: UNSAT
// of the dis-equivalence). This proves cvc5 accepts the HO_ALL `set.filter`
// encoding and decides it. (The guards are identical but written against
// independently-compiled policy sets, so the equivalence is solved, not
// trivially syntactic.)
#[tokio::test]
async fn symbolic_all_identical_guard_is_equivalent() {
    let eq = equivalent(
        r#"permit(principal, action, resource) when { context.ports.all(it > 0) };"#,
        r#"permit(principal, action, resource) when { context.ports.all(it > 0) };"#,
    )
    .await;
    assert!(eq, "the same `.all` guard must be equivalent (UNSAT)");
}

// Two policies whose `.all` guards DIFFER (`it > 0` vs `it >= 0`) are NOT
// equivalent (cvc5 finds a counterexample: a set containing `0`). This proves
// the solver actually DECIDES the symbolic `set.all`, not just parses it.
#[tokio::test]
async fn symbolic_all_different_guard_is_not_equivalent() {
    let eq = equivalent(
        r#"permit(principal, action, resource) when { context.ports.all(it > 0) };"#,
        r#"permit(principal, action, resource) when { context.ports.all(it >= 0) };"#,
    )
    .await;
    assert!(!eq, "`it > 0` and `it >= 0` quantifier guards must NOT be equivalent (SAT)");
}

// F-6: an entity-set `.all` whose predicate has a NON-EMPTY entity footprint
// (`it == principal`), driven end-to-end through cvc5. Unlike the `Set<Long>`
// cases above, this exercises the receiver + predicate entity-footprint union
// (enforcer `footprint`'s `.all` arm) through the SMT solve path, so a footprint
// that produced malformed SMT or a crash would surface here. The two guards
// (`it == principal` vs `it == resource.owner`) differ, so cvc5 must DECIDE a
// counterexample ⇒ NOT equivalent.
//
// NOTE on "a missing footprint would flip the verdict": that is NOT feasible to
// construct for any `.all` the compiler accepts. The only footprint-sensitive op
// is `in` (the ancestors UF), and D-70-A (`pred_no_it_dependent_in`) REJECTS an
// `it`-dependent left operand of `in` inside a predicate — the one shape whose
// element UIDs the footprint would have to cover. An `==`/attribute predicate
// (the supported entity shape) never consumes the ancestor UF, so its `it`-free
// entity term's acyclicity/transitivity assumption cannot change a sat/unsat
// verdict. Hence the footprint correctness guard is the DETERMINISTIC
// `enforcer.rs` unit test (F-6), and this cvc5 case only confirms the entity
// footprint path solves cleanly, not that a missing term flips the answer.
#[tokio::test]
async fn entity_set_all_different_guard_is_not_equivalent() {
    let validator = Validator::new(entity_schema());
    let pset1 = pset_from_text(
        r#"permit(principal, action, resource) when { resource.owners.all(it == principal) };"#,
        &validator,
    );
    let pset2 = pset_from_text(
        r#"permit(principal, action, resource) when { resource.owners.all(it == resource.owner) };"#,
        &validator,
    );
    let envs = Environments::new(validator.schema(), "User", "Action::\"view\"", "Doc");
    let typed1 =
        WellTypedPolicies::from_policies(&pset1, &envs.req_env, envs.schema).unwrap();
    let typed2 =
        WellTypedPolicies::from_policies(&pset2, &envs.req_env, envs.schema).unwrap();
    let mut compiler = CedarSymCompiler::new(LocalSolver::cvc5().unwrap()).unwrap();
    let eq = compiler
        .check_equivalent(&typed1, &typed2, &envs.symenv)
        .await
        .expect("cvc5 should solve the entity-set `.all` query under HO_ALL");
    assert!(
        !eq,
        "`it == principal` and `it == resource.owner` guards must NOT be equivalent (SAT)"
    );
}

fn entity_schema() -> Schema {
    schema_from_cedarstr(
        r#"
        entity User;
        entity Doc { owner: User, owners: Set<User> };
        action view appliesTo {
          principal: [User],
          resource: [Doc],
          context: {}
        };
    "#,
    )
}
