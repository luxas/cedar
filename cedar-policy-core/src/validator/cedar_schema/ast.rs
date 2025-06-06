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

use std::{collections::BTreeMap, iter::once};

use crate::{
    ast::{Annotation, Annotations, AnyId, Id, InternalName},
    parser::{AsLocRef, Loc, MaybeLoc, Node},
};
use itertools::{Either, Itertools};
use nonempty::NonEmpty;
use smol_str::SmolStr;
// We don't need this import on macOS but CI fails without it
#[allow(unused_imports)]
use smol_str::ToSmolStr;

use crate::validator::json_schema;

use super::err::UserError;

pub const BUILTIN_TYPES: [&str; 3] = ["Long", "String", "Bool"];

pub(super) const CEDAR_NAMESPACE: &str = "__cedar";

/// A struct that can be annotated, e.g., entity types.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Annotated<T> {
    /// The struct that's optionally annotated
    pub data: T,
    /// Annotations
    pub annotations: Annotations,
}

pub type Schema = Vec<Annotated<Namespace>>;

#[allow(clippy::type_complexity)]
pub fn deduplicate_annotations<T>(
    data: T,
    annotations: Vec<Node<(Node<AnyId>, Option<Node<SmolStr>>)>>,
) -> Result<Annotated<T>, UserError> {
    let mut unique_annotations: BTreeMap<Node<AnyId>, Option<Node<SmolStr>>> = BTreeMap::new();
    for annotation in annotations {
        let (key, value) = annotation.node;
        if let Some((old_key, _)) = unique_annotations.get_key_value(&key) {
            return Err(UserError::DuplicateAnnotations(
                key.node,
                Node::with_maybe_source_loc((), old_key.loc.clone()),
                Node::with_maybe_source_loc((), key.loc),
            ));
        } else {
            unique_annotations.insert(key, value);
        }
    }
    Ok(Annotated {
        data,
        annotations: unique_annotations
            .into_iter()
            .map(|(key, value)| {
                let (val, loc) = match value {
                    Some(n) => (Some(n.node), n.loc),
                    None => (None, None),
                };
                (key.node, Annotation::with_optional_value(val, loc))
            })
            .collect(),
    })
}

/// A path is a non empty list of identifiers that forms a namespace + type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path(Node<PathInternal>);
impl Path {
    /// Create a [`Path`] with a single entry
    pub fn single(basename: Id, loc: MaybeLoc) -> Self {
        Self(Node::with_maybe_source_loc(
            PathInternal {
                basename,
                namespace: vec![],
            },
            loc,
        ))
    }

    /// Create [`Path`] with a head and an iterator. Most significant name first.
    pub fn new(basename: Id, namespace: impl IntoIterator<Item = Id>, loc: MaybeLoc) -> Self {
        let namespace = namespace.into_iter().collect();
        Self(Node::with_maybe_source_loc(
            PathInternal {
                basename,
                namespace,
            },
            loc,
        ))
    }

    /// Borrowed iteration of the [`Path`]'s elements. Most significant name first
    pub fn iter(&self) -> impl Iterator<Item = &Id> {
        self.0.node.iter()
    }

    /// Source [`Loc`] of this [`Path`]
    pub fn loc(&self) -> Option<&Loc> {
        self.0.loc.as_loc_ref()
    }

    /// Consume the [`Path`] and get an owned iterator over the elements. Most significant name first
    #[allow(clippy::should_implement_trait)] // difficult to write the `IntoIter` type for this implementation
    pub fn into_iter(self) -> impl Iterator<Item = Node<Id>> {
        let loc = self.0.loc;
        self.0
            .node
            .into_iter()
            .map(move |x| Node::with_maybe_source_loc(x, loc.clone()))
    }

    /// Get the base type name as well as the (potentially empty) prefix
    pub fn split_last(self) -> (Vec<Id>, Id) {
        (self.0.node.namespace, self.0.node.basename)
    }

    /// Is this referring to a name in the `__cedar` namespace (eg: `__cedar::Bool`)
    pub fn is_in_cedar(&self) -> bool {
        self.0.node.is_in_cedar()
    }
}

impl From<Path> for InternalName {
    fn from(value: Path) -> Self {
        InternalName::new(value.0.node.basename, value.0.node.namespace, value.0.loc)
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PathInternal {
    basename: Id,
    namespace: Vec<Id>,
}

impl PathInternal {
    fn iter(&self) -> impl Iterator<Item = &Id> {
        self.namespace.iter().chain(once(&self.basename))
    }

    /// Is this referring to a name _in_ the `__cedar` namespace (ex: `__cedar::Bool`)
    fn is_in_cedar(&self) -> bool {
        match self.namespace.as_slice() {
            [id] => id.as_ref() == CEDAR_NAMESPACE,
            _ => false,
        }
    }
}

impl IntoIterator for PathInternal {
    type Item = Id;
    type IntoIter = std::iter::Chain<<Vec<Id> as IntoIterator>::IntoIter, std::iter::Once<Id>>;

    fn into_iter(self) -> Self::IntoIter {
        self.namespace.into_iter().chain(once(self.basename))
    }
}

impl std::fmt::Display for PathInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.namespace.is_empty() {
            write!(f, "{}", self.basename)
        } else {
            let namespace = self.namespace.iter().map(|id| id.as_ref()).join("::");
            write!(f, "{namespace}::{}", self.basename)
        }
    }
}

/// This struct represents Entity Uids in the Schema Syntax
#[derive(Debug, Clone)]
pub struct QualName {
    pub path: Option<Path>,
    pub eid: SmolStr,
}

impl QualName {
    pub fn unqualified(eid: SmolStr) -> Self {
        Self { path: None, eid }
    }

    pub fn qualified(path: Path, eid: SmolStr) -> Self {
        Self {
            path: Some(path),
            eid,
        }
    }
}

/// A [`Namespace`] has a name and a collection declaration
/// A schema is made up of a series of fragments
/// A fragment is a series of namespaces
#[derive(Debug, Clone)]
pub struct Namespace {
    /// The name of this namespace. If [`None`], then this is the unqualified namespace
    pub name: Option<Path>,
    /// The [`Declaration`]s contained in this namespace
    pub decls: Vec<Annotated<Node<Declaration>>>,
    pub loc: MaybeLoc,
}

impl Namespace {
    /// Is this [`Namespace`] unqualfied?
    pub fn is_unqualified(&self) -> bool {
        self.name.is_none()
    }
}

pub trait Decl {
    fn names(&self) -> Vec<Node<SmolStr>>;
}

/// Schema Declarations,
/// Defines either entity types, action types, or common types
#[derive(Debug, Clone)]
pub enum Declaration {
    Entity(EntityDecl),
    Action(ActionDecl),
    Type(TypeDecl),
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: Node<Id>,
    pub def: Node<Type>,
}

impl Decl for TypeDecl {
    fn names(&self) -> Vec<Node<SmolStr>> {
        vec![self.name.clone().map(|id| id.to_smolstr())]
    }
}

#[derive(Debug, Clone)]
pub enum EntityDecl {
    Standard(StandardEntityDecl),
    Enum(EnumEntityDecl),
}

impl EntityDecl {
    pub fn names(&self) -> impl Iterator<Item = &Node<Id>> + '_ {
        match self {
            Self::Enum(d) => d.names.iter(),
            Self::Standard(d) => d.names.iter(),
        }
    }
}

/// Declaration of an entity type
#[derive(Debug, Clone)]
pub struct StandardEntityDecl {
    /// Entity Type Names bound by this declaration.
    /// More than one name can be bound if they have the same definition, for convenience
    pub names: NonEmpty<Node<Id>>,
    /// Entity Types this type is allowed to be related to via the `in` relation
    pub member_of_types: Vec<Path>,
    /// Attributes this entity has
    pub attrs: Node<Vec<Node<Annotated<AttrDecl>>>>,
    /// Tag type for this entity (`None` means no tags on this entity)
    pub tags: Option<Node<Type>>,
}

/// Declaration of an entity type
#[derive(Debug, Clone)]
pub struct EnumEntityDecl {
    pub names: NonEmpty<Node<Id>>,
    pub choices: NonEmpty<Node<SmolStr>>,
}

/// Type definitions
#[derive(Debug, Clone)]
pub enum Type {
    /// A set of types
    Set(Box<Node<Type>>),
    /// A [`Path`] that could either refer to a Common Type or an Entity Type
    Ident(Path),
    /// A Record
    Record(Vec<Node<Annotated<AttrDecl>>>),
}

/// Primitive Type Definitions
#[derive(Debug, Clone)]
pub enum PrimitiveType {
    /// Cedar Longs
    Long,
    /// Cedar Strings
    String,
    /// Cedar booleans
    Bool,
}

impl<N> From<PrimitiveType> for json_schema::TypeVariant<N> {
    fn from(value: PrimitiveType) -> Self {
        match value {
            PrimitiveType::Long => json_schema::TypeVariant::Long,
            PrimitiveType::String => json_schema::TypeVariant::String,
            PrimitiveType::Bool => json_schema::TypeVariant::Boolean,
        }
    }
}

/// Attribute declarations, used in records and entity types.
/// One [`AttrDecl`] is one key-value pair.
#[derive(Debug, Clone)]
pub struct AttrDecl {
    /// Name of this attribute
    pub name: Node<SmolStr>,
    /// Whether or not it is a required attribute (default `true`)
    pub required: bool,
    /// The type of this attribute
    pub ty: Node<Type>,
}

/// The target of a [`PRAppDecl`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PR {
    /// Applies to the `principal` variable
    Principal,
    /// Applies to the `resource` variable
    Resource,
}

impl std::fmt::Display for PR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PR::Principal => write!(f, "principal"),
            PR::Resource => write!(f, "resource"),
        }
    }
}

/// A declaration that defines what kind of entities this action can be applied against
#[derive(Debug, Clone)]
pub struct PRAppDecl {
    /// Is this constraining the `principal` or the `resource`
    pub kind: Node<PR>,
    /// What entity types are allowed? `None` means none
    pub entity_tys: Option<NonEmpty<Path>>,
}

/// A declaration of constraints on an action type
#[derive(Debug, Clone)]
pub enum AppDecl {
    /// Constraints on the `principal` or `resource`
    PR(PRAppDecl),
    /// Constraints on the `context`
    Context(Either<Path, Node<Vec<Node<Annotated<AttrDecl>>>>>),
}

/// An action declaration
#[derive(Debug, Clone)]
pub struct ActionDecl {
    /// The names this declaration is binding.
    /// More than one name can be bound if they have the same definition, for convenience.
    pub names: NonEmpty<Node<SmolStr>>,
    /// The parents of this action
    pub parents: Option<NonEmpty<Node<QualName>>>,
    /// The constraining clauses in this declarations
    pub app_decls: Option<Node<NonEmpty<Node<AppDecl>>>>,
}

impl Decl for ActionDecl {
    fn names(&self) -> Vec<Node<SmolStr>> {
        self.names.iter().cloned().collect()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::parser::IntoMaybeLoc;

    use super::*;

    fn loc() -> Loc {
        Loc::new((1, 1), Arc::from("foo"))
    }

    // Ensure the iterators over [`Path`]s return most significant names first
    #[test]
    fn path_iter() {
        let p = Path::new(
            "baz".parse().unwrap(),
            ["foo".parse().unwrap(), "bar".parse().unwrap()],
            loc().into_maybe_loc(),
        );

        let expected: Vec<Id> = vec![
            "foo".parse().unwrap(),
            "bar".parse().unwrap(),
            "baz".parse().unwrap(),
        ];

        let expected_borrowed = expected.iter().collect::<Vec<_>>();

        let borrowed = p.iter().collect::<Vec<_>>();
        assert_eq!(borrowed, expected_borrowed);
        let moved = p.into_iter().map(|n| n.node).collect::<Vec<_>>();
        assert_eq!(moved, expected);
    }
}
