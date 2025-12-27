use cgmath::{vec2, vec3, ElementWise, Vector2};
use perovskite_core::far_sheet::{SheetControl, TessellationMode};

use crate::game_state::mapgen::far_mesh::tri_quad::{EntryCore, TilePosture};

// sqrt is not const sadly
const SQRT_3: f64 = 1.7320508075688772;
// The basis vectors for the finest lattice. Every grid we will render is a (likely-power-of-two) multiple of this.
// We may not necessarily render this lattice, we may only render a coarser lattice.
const FUNDAMENTAL_BASIS_U: cgmath::Vector2<f64> = cgmath::Vector2::new(2.0 / SQRT_3, 0.0);
const FUNDAMENTAL_BASIS_V: cgmath::Vector2<f64> = cgmath::Vector2::new(1.0 / SQRT_3, 1.0);

const LATTICE2WORLD: cgmath::Matrix2<f64> = cgmath::Matrix2::new(
    FUNDAMENTAL_BASIS_U.x,
    FUNDAMENTAL_BASIS_U.y,
    FUNDAMENTAL_BASIS_V.x,
    FUNDAMENTAL_BASIS_V.y,
);
const WORLD2LATTICE: cgmath::Matrix2<f64> =
    cgmath::Matrix2::new(SQRT_3 / 2.0, 0.0, -1.0 / 2.0, 1.0);

// Every grid we will render will have this shape:
const GRID_M: usize = 64;
const GRID_N: usize = 64;
const GRID_K: isize = -1;

// World space: (each triangle is one sheet)
//       ^------------x
//      / \         /
//     /   \       /  64*basis_v
//    /     \     /
//   /       \   /
//  /         \ /
// O-----------v
//  64*basis_u
//
// Lattice space in tri_quad (each triangle is one sheet)
// (a 1x1 square cannot distinguish between upper and lower half, a deficiency
// in the tri_quad implementation)
//
// X----X
// |\   |
// | \  | 2
// |  \ |
// |   \|
// O----X
//   2

const TRI_QUAD_TOTAL_SIZE: u32 = 1 << 30;
const TRI_QUAD_FINEST_VALID: u32 = 2;
// This is appoximately a 64 * 2 * 2 / sqrt(3), i.e. 147-block-long tile,
// with 2*2/sqrt(3) = 2.3 block stride.
pub(crate) const FINEST_RENDERED_SIZE: u32 = 8;
// This is 16x larger, with 37-block stride, roughly.
pub(crate) const COARSEST_RENDERED_SIZE: u32 = 128;

pub(crate) fn to_sheet_control(coord: &EntryCore) -> SheetControl {
    let basis_mult = match coord.posture() {
        TilePosture::LowerHalf => vec2(1.0, 1.0),
        TilePosture::UpperHalf => vec2(1.0, -1.0),
    };
    let basis_u =
        FUNDAMENTAL_BASIS_U.mul_element_wise(basis_mult) * 0.5 * (coord.side_length() as f64);
    let basis_v =
        FUNDAMENTAL_BASIS_V.mul_element_wise(basis_mult) * 0.5 * (coord.side_length() as f64);

    let coarse_basis_u = FUNDAMENTAL_BASIS_U * 32.0;
    let coarse_basis_v = FUNDAMENTAL_BASIS_V * 32.0;
    let y_correction = match coord.posture() {
        TilePosture::LowerHalf => 0.0,
        TilePosture::UpperHalf => coord.side_length() as f64,
    };
    let origin = coarse_basis_u * (coord.x_range().start as f64 + (i32::MIN >> 1) as f64)
        + coarse_basis_v * (coord.y_range().start as f64 + y_correction + (i32::MIN >> 1) as f64);

    SheetControl::new(
        vec3(origin.x, 0.0, origin.y),
        basis_u,
        basis_v,
        GRID_M,
        GRID_N,
        GRID_K,
        TessellationMode::NegDiagonal,
    )
    .unwrap()
}

pub(crate) fn world_pos_to_map_pos(pos: Vector2<f64>) -> (u32, u32) {
    // Each sheet is 64 elements long, but the finest sheet corresponds to a 2x2 triangle
    // in coarse lattice space because of tri_quad limitations.
    let lattice =
        (WORLD2LATTICE * pos / 32.0) - vec2((i32::MIN >> 1) as f64, (i32::MIN >> 1) as f64);
    (lattice.x as u32, lattice.y as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::SquareMatrix;
    use cgmath::{assert_relative_eq, vec2};

    #[test]
    fn test_lattice2world() {
        assert_relative_eq!(
            LATTICE2WORLD * vec2(0.0, 1.0),
            vec2(1.0 / SQRT_3, 1.0),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            LATTICE2WORLD * vec2(1.0, 1.0),
            vec2(SQRT_3, 1.0),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            LATTICE2WORLD * WORLD2LATTICE,
            cgmath::Matrix2::identity(),
            epsilon = 1e-10
        );
    }
}

/// A triangular quadtree. This is specific to the game server, and is not a general-purpose data structure.
/// In particular: weaker ability to remove nodes, filling-in procedure specific to the game's needs,
/// change tracking used for network updates, etc, all tightly coupled with the data structure.
///
/// It has a generic type parameter for two reasons: Testing, and flexibility for the network implementation/caching/etc.
pub(crate) mod tri_quad {
    use std::ops::Range;

    slotmap::new_key_type! {
        struct NodeKey;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) enum TilePosture {
        LowerHalf,
        UpperHalf,
    }

    /// Callback that is called when a node is inserted or deleted. These callbacks should
    /// both provide a value, and also have a side effect, e.g. sending a network update to
    /// the client.
    pub(crate) trait ChangeCallbacks<T: Send + Sync + 'static> {
        /// Called when a node is inserted, with an EntryCore describing the node. Returns the value to store in the node.
        fn insert(&mut self, entry: &EntryCore) -> T;
        /// Called when a node is deleted, with the value that was stored in the node.
        fn delete(&mut self, entry: T);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) enum PolicyDecision {
        /// Insert the node at the current level.
        InsertAtLevel,
        /// Subdivide the node into four smaller nodes, then recursively fill.
        Subdivide,
        /// Do not insert the node, and do not fill in any of its children. Leave it as-is.
        DoNotInsert,
        /// If the node is subdivided, bring it back to the current level.
        Retract,
        /// Regardless of whether the node is subdivided, delete it.
        Delete,
    }
    pub(crate) trait InsertionPolicy {
        fn decide(&self, entry: &EntryCore) -> PolicyDecision;
    }

    /// A node of a triangular quadtree. Note that this can be a server implementation detail and
    /// does not need a stable network serialization or protocol - the server just sends the client
    /// sheets.
    enum TriQuadNode<T> {
        EmptyLeaf,
        Leaf(T),
        Internal {
            rect: TriQuadPair,
            tris: [NodeKey; 2],
        },
    }

    #[derive(Clone, Copy)]
    struct TriQuadPair {
        upper: NodeKey,
        lower: NodeKey,
    }

    pub(crate) struct TriQuadTree<T> {
        root: TriQuadPair,
        nodes: slotmap::SlotMap<NodeKey, TriQuadNode<T>>,
        // Precondition: grid_size != u32::MAX
        // Precondition: grid_size & (grid_size - 1) == 0
        grid_size: u32,
    }

    impl<T> TriQuadTree<T> {
        /// Creates a new TriQuadTree.
        ///
        /// Args:
        ///     * grid_size: Side length of the root level. Must be a power of two.
        ///
        /// Returns:
        ///     * TriQuadTree<T>: The new TriQuadTree.
        pub(crate) fn new(grid_size: u32) -> Self {
            if !grid_size.is_power_of_two() {
                panic!("grid_size must be a power of 2");
            }
            let mut nodes = slotmap::SlotMap::with_key();
            let upper = nodes.insert(TriQuadNode::EmptyLeaf);
            let lower = nodes.insert(TriQuadNode::EmptyLeaf);
            Self {
                root: TriQuadPair { upper, lower },
                nodes,
                grid_size,
            }
        }

        fn core_entry(&self, x: u32, y: u32) -> EntryCore {
            self.traverse_rects(&self.root, x, y, self.grid_size - 1, self.grid_size >> 1)
        }

        pub(crate) fn entry(&self, x: u32, y: u32) -> TriQuadEntry<'_, T> {
            TriQuadEntry {
                core: self.core_entry(x, y),
                map: &self.nodes,
            }
        }

        pub(crate) fn entry_mut(&mut self, x: u32, y: u32) -> TriQuadEntryMut<'_, T> {
            TriQuadEntryMut {
                core: self.core_entry(x, y),
                map: &mut self.nodes,
            }
        }

        fn traverse_nodes(
            &self,
            slot: NodeKey,
            x: u32,
            y: u32,
            dense_mask: u32,
            leading_mask: u32,
        ) -> EntryCore {
            match &self.nodes[slot] {
                TriQuadNode::Leaf(_) | TriQuadNode::EmptyLeaf => {
                    EntryCore::new(slot, x, y, dense_mask)
                }
                TriQuadNode::Internal { rect, tris } => {
                    let cx = (x & leading_mask) != 0;
                    let cy = (y & leading_mask) != 0;

                    match (cx, cy) {
                        (true, false) => {
                            self.traverse_nodes(tris[0], x, y, dense_mask >> 1, leading_mask >> 1)
                        }
                        (false, true) => {
                            self.traverse_nodes(tris[1], x, y, dense_mask >> 1, leading_mask >> 1)
                        }
                        (false, false) | (true, true) => {
                            self.traverse_rects(rect, x, y, dense_mask >> 1, leading_mask >> 1)
                        }
                    }
                }
            }
        }

        fn traverse_rects(
            &self,
            rect: &TriQuadPair,
            x: u32,
            y: u32,
            dense_mask: u32,
            leading_mask: u32,
        ) -> EntryCore {
            let sum = (x & dense_mask).saturating_add(y & dense_mask);
            let node = if sum < dense_mask {
                rect.lower
            } else {
                rect.upper
            };
            self.traverse_nodes(node, x, y, dense_mask, leading_mask)
        }
    }

    impl<T: Send + Sync + 'static> TriQuadTree<T> {
        /// Remove a node and all of its children. The node itself must have already been
        /// removed or replaced either before this call or right after.
        /// (Removed == no longer referenced by a parent node, AND removed from the slotmap)
        /// (Replaced == a new node was put into the slotmap, and we're now recursively cleaning
        ///   up the old node and its children)
        fn remove_detached_tree(
            &mut self,
            node: TriQuadNode<T>,
            callbacks: &mut impl ChangeCallbacks<T>,
        ) {
            match node {
                TriQuadNode::Leaf(data) => callbacks.delete(data),
                TriQuadNode::Internal { rect, tris } => {
                    let node = self.nodes.remove(rect.lower).unwrap();
                    self.remove_detached_tree(node, callbacks);
                    let node = self.nodes.remove(rect.upper).unwrap();
                    self.remove_detached_tree(node, callbacks);

                    for &tri in tris.iter() {
                        let node = self.nodes.remove(tri).unwrap();
                        self.remove_detached_tree(node, callbacks);
                    }
                }
                TriQuadNode::EmptyLeaf => {}
            }
        }

        /// Insert a node at the given position, with the given side length. If there is already
        /// finer geometry at this location, do nothing (the finer geometry takes precedence).
        ///
        /// Otherwise, insert the node at this level, call the callbacks, then recursively
        /// insert neighbors to fill nodes on the way back up, with the most coarse representation
        /// that will fit. This will insert at every intermediate level
        ///
        /// It is the responsibility of the caller to disregard levels that are too coarse to be
        /// interesting, and provide a suitable empty value from the callback (that empty value
        /// being application-dependent)
        ///
        /// Args:
        ///     * x: x coordinate.
        ///     * y: y coordinate.
        ///     * side_length: Side length of the target level. Must be a power of two.
        ///     * callbacks: Callbacks to be called when nodes are inserted or removed.
        pub(crate) fn insert_at(
            &mut self,
            x: u32,
            y: u32,
            side_length: u32,
            callbacks: &mut impl ChangeCallbacks<T>,
        ) {
            struct PolicyImpl {
                x: u32,
                y: u32,
                stop_len: u32,
            }
            impl InsertionPolicy for PolicyImpl {
                fn decide(&self, entry: &EntryCore) -> PolicyDecision {
                    if entry.contains(self.x, self.y) && entry.side_length() > self.stop_len {
                        PolicyDecision::Subdivide
                    } else {
                        PolicyDecision::InsertAtLevel
                    }
                }
            }
            self.fill_with_policy(
                callbacks,
                &mut PolicyImpl {
                    x: x,
                    y: y,
                    stop_len: side_length,
                },
            )
        }

        fn fill_with_policy_impl(
            &mut self,
            entry: EntryCore,
            callbacks: &mut impl ChangeCallbacks<T>,
            policy: &impl InsertionPolicy,
        ) {
            let node = &self.nodes[entry.node];
            let decision = policy.decide(&entry);
            let xmin = entry.x_range().start;
            let ymin = entry.y_range().start;
            let xmax = entry.x_range().end - 1;
            let ymax = entry.y_range().end - 1;
            let rect_coord = match entry.posture {
                TilePosture::LowerHalf => (xmin, ymin),
                TilePosture::UpperHalf => (xmax, ymax),
            };
            match decision {
                PolicyDecision::Subdivide => {
                    let next_rect;
                    let next_tris;
                    match node {
                        TriQuadNode::Leaf(_) | TriQuadNode::EmptyLeaf => {
                            // The node is a leaf. Subdivide it once.
                            next_rect = TriQuadPair {
                                lower: self.nodes.insert(TriQuadNode::EmptyLeaf),
                                upper: self.nodes.insert(TriQuadNode::EmptyLeaf),
                            };
                            next_tris = [
                                self.nodes.insert(TriQuadNode::EmptyLeaf),
                                self.nodes.insert(TriQuadNode::EmptyLeaf),
                            ];
                            let new_node = TriQuadNode::Internal {
                                rect: next_rect,
                                tris: next_tris,
                            };
                            let old_node = std::mem::replace(&mut self.nodes[entry.node], new_node);
                            self.remove_detached_tree(old_node, callbacks);
                        }
                        TriQuadNode::Internal { rect, tris } => {
                            next_rect = *rect;
                            next_tris = *tris;
                        }
                    }

                    for (node, coord, posture) in [
                        (next_tris[0], (xmax, ymin), entry.posture),
                        (next_tris[1], (xmin, ymax), entry.posture),
                        (next_rect.lower, rect_coord, TilePosture::LowerHalf),
                        (next_rect.upper, rect_coord, TilePosture::UpperHalf),
                    ] {
                        self.fill_with_policy_impl(
                            EntryCore::new_from_posture(
                                node,
                                coord.0,
                                coord.1,
                                entry.dense_mask >> 1,
                                posture,
                            ),
                            callbacks,
                            policy,
                        );
                    }
                }
                PolicyDecision::InsertAtLevel => {
                    match node {
                        TriQuadNode::Leaf(_) => {
                            // Do nothing, data already present at same or denser level
                        }
                        TriQuadNode::Internal { rect, tris } => {
                            // May need some retractions, but not at this level
                            for (node, coord, posture) in [
                                (tris[0], (xmax, ymin), entry.posture),
                                (tris[1], (xmin, ymax), entry.posture),
                                (rect.lower, rect_coord, TilePosture::LowerHalf),
                                (rect.upper, rect_coord, TilePosture::UpperHalf),
                            ] {
                                self.fill_with_policy_impl(
                                    EntryCore::new_from_posture(
                                        node,
                                        coord.0,
                                        coord.1,
                                        entry.dense_mask >> 1,
                                        posture,
                                    ),
                                    callbacks,
                                    policy,
                                );
                            }
                        }
                        TriQuadNode::EmptyLeaf => {
                            let data = callbacks.insert(&entry);
                            self.nodes[entry.node] = TriQuadNode::Leaf(data);
                        }
                    }
                }
                PolicyDecision::DoNotInsert => {
                    return;
                }
                PolicyDecision::Retract => match node {
                    TriQuadNode::Internal { .. } => {
                        let data = callbacks.insert(&entry);
                        let old_node =
                            std::mem::replace(&mut self.nodes[entry.node], TriQuadNode::Leaf(data));
                        self.remove_detached_tree(old_node, callbacks);
                    }
                    TriQuadNode::EmptyLeaf => {
                        let data = callbacks.insert(&entry);
                        self.nodes[entry.node] = TriQuadNode::Leaf(data);
                    }
                    TriQuadNode::Leaf(_) => {
                        // Do nothing, data already present at intended level
                    }
                },
                PolicyDecision::Delete => {
                    let old_node =
                        std::mem::replace(&mut self.nodes[entry.node], TriQuadNode::EmptyLeaf);
                    self.remove_detached_tree(old_node, callbacks);
                }
            }
        }

        pub(crate) fn fill_with_policy(
            &mut self,
            callbacks: &mut impl ChangeCallbacks<T>,
            policy: &impl InsertionPolicy,
        ) {
            let root_lower = EntryCore::new_from_posture(
                self.root.lower,
                0,
                0,
                self.grid_size - 1,
                TilePosture::LowerHalf,
            );
            self.fill_with_policy_impl(root_lower, callbacks, policy);

            let root_upper = EntryCore::new_from_posture(
                self.root.upper,
                self.grid_size - 1,
                self.grid_size - 1,
                self.grid_size - 1,
                TilePosture::UpperHalf,
            );
            self.fill_with_policy_impl(root_upper, callbacks, policy);
        }

        fn assert_key_filled(&self, key: NodeKey) {
            match &self.nodes[key] {
                TriQuadNode::Leaf(_) => {}
                TriQuadNode::Internal { rect, tris } => {
                    self.assert_key_filled(tris[0]);
                    self.assert_key_filled(tris[1]);
                    self.assert_key_filled(rect.lower);
                    self.assert_key_filled(rect.upper);
                }
                TriQuadNode::EmptyLeaf => {
                    panic!("node {:?} is an empty leaf", key);
                }
            }
        }

        fn assert_filled(&self) {
            let lower = &self.nodes[self.root.lower];
            let upper = &self.nodes[self.root.upper];
            if let TriQuadNode::Internal { .. } = lower {
                self.assert_key_filled(self.root.lower);
            }
            if let TriQuadNode::Internal { .. } = upper {
                self.assert_key_filled(self.root.upper);
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct EntryCore {
        x: u32,
        y: u32,
        posture: TilePosture,
        dense_mask: u32,
        node: NodeKey,
    }
    impl EntryCore {
        fn new(slot: NodeKey, x: u32, y: u32, dense_mask: u32) -> EntryCore {
            Self::new_from_posture(
                slot,
                x,
                y,
                dense_mask,
                if (x & dense_mask).saturating_add(y & dense_mask) < dense_mask {
                    TilePosture::LowerHalf
                } else {
                    TilePosture::UpperHalf
                },
            )
        }
        fn new_from_posture(
            slot: NodeKey,
            x: u32,
            y: u32,
            dense_mask: u32,
            posture: TilePosture,
        ) -> EntryCore {
            EntryCore {
                x,
                y,
                posture,
                dense_mask,
                node: slot,
            }
        }

        pub(crate) fn contains(&self, x: u32, y: u32) -> bool {
            if self.x & !self.dense_mask != x & !self.dense_mask {
                return false;
            }
            if self.y & !self.dense_mask != y & !self.dense_mask {
                return false;
            }
            let expected_posture =
                if (x & self.dense_mask).saturating_add(y & self.dense_mask) < self.dense_mask {
                    TilePosture::LowerHalf
                } else {
                    TilePosture::UpperHalf
                };
            if self.posture != expected_posture {
                return false;
            }
            true
        }
    }

    pub(crate) struct TriQuadEntry<'a, T> {
        core: EntryCore,
        map: &'a slotmap::SlotMap<NodeKey, TriQuadNode<T>>,
    }
    impl<'a, T> TriQuadEntry<'a, T> {
        pub(crate) fn core(&self) -> &EntryCore {
            &self.core
        }
        fn node(&self) -> &TriQuadNode<T> {
            &self.map[self.core.node]
        }
        pub(crate) fn value(&self) -> Option<&T> {
            match self.node() {
                TriQuadNode::Leaf(value) => Some(value),
                _ => None,
            }
        }
    }

    impl EntryCore {
        pub(crate) fn posture(&self) -> TilePosture {
            self.posture
        }
        pub(crate) fn x_range(&self) -> Range<u32> {
            // Lower is inclusive, add 1 since dense mask gives an inclusive range
            let lower = self.x & !self.dense_mask;
            let upper = (self.x | self.dense_mask) + 1;
            lower..upper
        }
        pub(crate) fn y_range(&self) -> Range<u32> {
            // Upper is exclusive, add 1 since dense mask gives an inclusive range
            let lower = self.y & !self.dense_mask;
            let upper = (self.y | self.dense_mask) + 1;
            lower..upper
        }
        pub(crate) fn debug_describe(&self) -> String {
            format!(
                "{} half of [{},{}) x [{},{}) ([{},{}) x [{},{}))",
                match self.posture {
                    TilePosture::LowerHalf => "Lower",
                    TilePosture::UpperHalf => "Upper",
                },
                (self.x_range().start as i32).wrapping_add(i32::MIN >> 1),
                (self.x_range().end as i32).wrapping_add(i32::MIN >> 1),
                (self.y_range().start as i32).wrapping_add(i32::MIN >> 1),
                (self.y_range().end as i32).wrapping_add(i32::MIN >> 1),
                self.x_range().start,
                self.x_range().end,
                self.y_range().start,
                self.y_range().end,
            )
        }
        pub(crate) fn side_length(&self) -> u32 {
            self.dense_mask + 1
        }
    }

    pub(crate) struct TriQuadEntryMut<'a, T> {
        core: EntryCore,
        map: &'a mut slotmap::SlotMap<NodeKey, TriQuadNode<T>>,
    }
    impl<'a, T> TriQuadEntryMut<'a, T> {
        fn core(&self) -> &EntryCore {
            &self.core
        }
        fn node(&self) -> &TriQuadNode<T> {
            &self.map[self.core.node]
        }
        pub(crate) fn value(&self) -> Option<&T> {
            match self.node() {
                TriQuadNode::Leaf(value) => Some(value),
                _ => None,
            }
        }
        fn node_mut(&mut self) -> &mut TriQuadNode<T> {
            &mut self.map[self.core.node]
        }
        pub(crate) fn value_mut(&mut self) -> Option<&mut T> {
            match self.node_mut() {
                TriQuadNode::Leaf(value) => Some(value),
                _ => None,
            }
        }
    }
    pub(crate) struct EntryMut<'a, T> {
        core: EntryCore,
        map: &'a mut slotmap::SlotMap<NodeKey, TriQuadNode<T>>,
    }

    #[cfg(test)]
    mod tests {
        use std::{cell::RefCell, collections::HashSet};

        use super::*;

        fn make_pair<T>(
            nodes: &RefCell<slotmap::SlotMap<NodeKey, TriQuadNode<T>>>,
            lower: TriQuadNode<T>,
            upper: TriQuadNode<T>,
        ) -> TriQuadPair {
            let mut nodes = nodes.borrow_mut();
            TriQuadPair {
                lower: nodes.insert(lower),
                upper: nodes.insert(upper),
            }
        }

        fn make_tris<T>(
            nodes: &RefCell<slotmap::SlotMap<NodeKey, TriQuadNode<T>>>,
            rect: TriQuadPair,
            node0: TriQuadNode<T>,
            node1: TriQuadNode<T>,
        ) -> TriQuadNode<T> {
            let mut nodes = nodes.borrow_mut();
            TriQuadNode::Internal {
                rect,
                tris: [nodes.insert(node0), nodes.insert(node1)],
            }
        }

        #[track_caller]
        fn check_entry<T: PartialEq + std::fmt::Debug>(
            entry: TriQuadEntry<T>,
            expected: Option<&T>,
        ) {
            // This is for a unit test, println is ok
            println!("{}", entry.core().debug_describe());
            assert_eq!(entry.value(), expected);
        }

        #[test]
        fn test_entry_core_contains() {
            let lower_entry = EntryCore {
                x: 32,
                y: 0,
                dense_mask: 15,
                node: NodeKey::default(),
                posture: TilePosture::LowerHalf,
            };
            assert!(lower_entry.contains(46, 0));
            assert!(!lower_entry.contains(47, 0));
            assert!(lower_entry.contains(35, 1));
            assert!(!lower_entry.contains(31, 1));
            assert!(!lower_entry.contains(15, 15));
            assert!(!lower_entry.contains(16, 16));
            assert!(lower_entry.contains(32, 3));

            let upper_entry = EntryCore {
                posture: TilePosture::UpperHalf,
                ..lower_entry
            };
            assert!(!upper_entry.contains(46, 0));
            assert!(upper_entry.contains(47, 0));
            assert!(upper_entry.contains(46, 5));
            assert!(!upper_entry.contains(35, 1));
            assert!(!upper_entry.contains(31, 1));
            assert!(!upper_entry.contains(15, 15));
            assert!(!upper_entry.contains(16, 16));
            assert!(!upper_entry.contains(33, 3));
            assert!(upper_entry.contains(33, 15));
        }

        #[test]
        fn test_traversal_simple_pair() {
            let nodes = RefCell::new(slotmap::SlotMap::with_key());
            let root = make_pair(&nodes, TriQuadNode::Leaf('a'), TriQuadNode::Leaf('b'));
            let tree = TriQuadTree {
                root,
                nodes: nodes.into_inner(),
                grid_size: 16,
            };
            check_entry(tree.entry(12, 12), Some(&'b'));
            check_entry(tree.entry(3, 10), Some(&'a'));
        }

        #[test]
        fn test_traversal_one_deep() {
            // refcell is ugly but allows quick and dirty testing where we build
            // the tree by hand.
            let nodes = RefCell::new(slotmap::SlotMap::with_key());

            let ej = make_pair(&nodes, TriQuadNode::Leaf('e'), TriQuadNode::Leaf('j'));
            let root = make_pair(
                &nodes,
                TriQuadNode::Leaf('a'),
                make_tris(&nodes, ej, TriQuadNode::Leaf('b'), TriQuadNode::Leaf('c')),
            );
            let tree = TriQuadTree {
                root,
                nodes: nodes.into_inner(),
                grid_size: 16,
            };
            println!("j");
            check_entry(tree.entry(15, 15), Some(&'j'));
            println!("e");
            check_entry(tree.entry(11, 11), Some(&'e'));
            println!("c");
            check_entry(tree.entry(1, 15), Some(&'c'));
            println!("b");
            check_entry(tree.entry(1, 1), Some(&'a'));
            println!("b");
            check_entry(tree.entry(15, 1), Some(&'b'));
            check_entry(tree.entry(15, 7), Some(&'b'));
        }

        #[test]
        fn test_traversal_two_deep() {
            let nodes = RefCell::new(slotmap::SlotMap::with_key());

            let root = make_pair(
                &nodes,
                TriQuadNode::Leaf('a'),
                make_tris(
                    &nodes,
                    make_pair(
                        &nodes,
                        make_tris(
                            &nodes,
                            make_pair(&nodes, TriQuadNode::Leaf('e'), TriQuadNode::Leaf('f')),
                            TriQuadNode::Leaf('g'),
                            TriQuadNode::Leaf('h'),
                        ),
                        TriQuadNode::Leaf('j'),
                    ),
                    TriQuadNode::Leaf('b'),
                    TriQuadNode::Leaf('c'),
                ),
            );

            let tree = TriQuadTree {
                root,
                nodes: nodes.into_inner(),
                grid_size: 16,
            };
            println!("j");
            check_entry(tree.entry(15, 15), Some(&'j'));
            println!("g");
            check_entry(tree.entry(12, 8), Some(&'g'));
            println!("h");
            check_entry(tree.entry(8, 12), Some(&'h'));
            println!("e");
            check_entry(tree.entry(9, 9), Some(&'e'));
            println!("f");
            check_entry(tree.entry(11, 10), Some(&'f'));
            println!("c");
            check_entry(tree.entry(1, 15), Some(&'c'));
            println!("b");
            check_entry(tree.entry(1, 1), Some(&'a'));
            println!("b");
            check_entry(tree.entry(15, 1), Some(&'b'));
            check_entry(tree.entry(15, 7), Some(&'b'));
        }

        #[test]
        fn test_simple_insert() {
            struct TestCallbacks {
                fills: HashSet<(Range<u32>, Range<u32>, TilePosture)>,
            }
            impl ChangeCallbacks<String> for TestCallbacks {
                fn insert(&mut self, entry: &EntryCore) -> String {
                    self.fills.insert((
                        entry.x_range().clone(),
                        entry.y_range().clone(),
                        entry.posture(),
                    ));
                    println!(
                        "inserting {} {:?}",
                        entry.debug_describe(),
                        crate::game_state::mapgen::far_mesh::to_sheet_control(entry)
                    );
                    entry.debug_describe()
                }
                fn delete(&mut self, value: String) {
                    println!("deleting {}", value);
                }
            }
            let mut tree = TriQuadTree::new(16);
            let mut callbacks = TestCallbacks {
                fills: HashSet::new(),
            };
            tree.insert_at(9, 9, 4, &mut callbacks);

            let expected_fills = [
                // zeroth level lower half
                (0..16, 0..16, TilePosture::LowerHalf),
                // First level, flanking triangles
                (0..8, 8..16, TilePosture::UpperHalf),
                (8..16, 0..8, TilePosture::UpperHalf),
                // First level, rectangle pair
                (8..16, 8..16, TilePosture::UpperHalf),
                // Second level, flanking triangles
                (8..12, 12..16, TilePosture::LowerHalf),
                (12..16, 8..12, TilePosture::LowerHalf),
                // Second level, rectangle pair
                (8..12, 8..12, TilePosture::LowerHalf),
                (8..12, 8..12, TilePosture::UpperHalf),
            ];
            assert_eq!(callbacks.fills, HashSet::from_iter(expected_fills));

            tree.assert_filled();
        }

        #[rustfmt::skip]
        fn empirical_insert_expected() -> HashSet<(Range<u32>, Range<u32>, TilePosture)> {
            HashSet::from_iter([
                (0..2147483648, 0..2147483648, TilePosture::LowerHalf),
                (1073741824..2147483648, 0..1073741824, TilePosture::UpperHalf),  
                (1040187392..1073741824, 1073741824..1107296256, TilePosture::UpperHalf),
                (1006632960..1040187392, 1107296256..1140850688, TilePosture::UpperHalf),
                (1040187392..1073741824, 1107296256..1140850688, TilePosture::LowerHalf),
                (1065353216..1073741824, 1107296256..1115684864, TilePosture::UpperHalf),
                (1056964608..1065353216, 1115684864..1124073472, TilePosture::UpperHalf),
                (1065353216..1073741824, 1115684864..1124073472, TilePosture::LowerHalf),
                (1069547520..1073741824, 1115684864..1119879168, TilePosture::UpperHalf),
                (1067450368..1069547520, 1119879168..1121976320, TilePosture::UpperHalf),
                (1066401792..1067450368, 1121976320..1123024896, TilePosture::UpperHalf),
                (1065877504..1066401792, 1123024896..1123549184, TilePosture::UpperHalf),
                (1065353216..1065877504, 1123549184..1124073472, TilePosture::UpperHalf),
                (1065877504..1066401792, 1123549184..1124073472, TilePosture::LowerHalf),
                (1066139648..1066401792, 1123549184..1123811328, TilePosture::UpperHalf),
                (1065877504..1066139648, 1123811328..1124073472, TilePosture::UpperHalf),
                (1066139648..1066401792, 1123811328..1124073472, TilePosture::LowerHalf),
                (1066270720..1066401792, 1123811328..1123942400, TilePosture::UpperHalf),
                (1066237952..1066270720, 1123942400..1123975168, TilePosture::UpperHalf),
                (1066205184..1066237952, 1123975168..1124007936, TilePosture::UpperHalf),
                (1066254336..1066270720, 1123975168..1123991552, TilePosture::LowerHalf),
                (1066246144..1066254336, 1123991552..1123999744, TilePosture::LowerHalf),
                (1066237952..1066246144, 1123999744..1124007936, TilePosture::LowerHalf),
                (1066242048..1066246144, 1123991552..1123995648, TilePosture::LowerHalf),
                (1066237952..1066242048, 1123995648..1123999744, TilePosture::LowerHalf),
                (1066240000..1066242048, 1123991552..1123993600, TilePosture::LowerHalf),
                (1066237952..1066240000, 1123993600..1123995648, TilePosture::LowerHalf),
                (1066239744..1066240000, 1123991552..1123991808, TilePosture::LowerHalf),
                (1066239488..1066239744, 1123991808..1123992064, TilePosture::LowerHalf),
                (1066239616..1066239744, 1123991552..1123991680, TilePosture::LowerHalf),
                (1066239552..1066239616, 1123991680..1123991744, TilePosture::LowerHalf),
                (1066239488..1066239552, 1123991744..1123991808, TilePosture::LowerHalf),
                (1066239488..1066239552, 1123991680..1123991744, TilePosture::LowerHalf),
                (1066239520..1066239552, 1123991680..1123991712, TilePosture::UpperHalf),
                (1066239488..1066239520, 1123991712..1123991744, TilePosture::UpperHalf),
                (1066239548..1066239552, 1123991712..1123991716, TilePosture::LowerHalf),
                (1066239544..1066239548, 1123991716..1123991720, TilePosture::LowerHalf),
                (1066239544..1066239548, 1123991712..1123991716, TilePosture::LowerHalf),
                (1066239544..1066239548, 1123991712..1123991716, TilePosture::UpperHalf),
                (1066239536..1066239544, 1123991720..1123991728, TilePosture::LowerHalf),
                (1066239536..1066239544, 1123991712..1123991720, TilePosture::LowerHalf),
                (1066239536..1066239544, 1123991712..1123991720, TilePosture::UpperHalf),
                (1066239520..1066239536, 1123991728..1123991744, TilePosture::LowerHalf),
                (1066239520..1066239536, 1123991712..1123991728, TilePosture::LowerHalf),
                (1066239520..1066239536, 1123991712..1123991728, TilePosture::UpperHalf),
                (1066239520..1066239552, 1123991712..1123991744, TilePosture::UpperHalf),
                (1066239488..1066239616, 1123991552..1123991680, TilePosture::LowerHalf),
                (1066239488..1066239616, 1123991552..1123991680, TilePosture::UpperHalf),
                (1066239488..1066239744, 1123991552..1123991808, TilePosture::UpperHalf),
                (1066238976..1066239488, 1123992064..1123992576, TilePosture::LowerHalf),
                (1066238976..1066239488, 1123991552..1123992064, TilePosture::LowerHalf),
                (1066238976..1066239488, 1123991552..1123992064, TilePosture::UpperHalf),
                (1066237952..1066238976, 1123992576..1123993600, TilePosture::LowerHalf),
                (1066237952..1066238976, 1123991552..1123992576, TilePosture::LowerHalf),
                (1066237952..1066238976, 1123991552..1123992576, TilePosture::UpperHalf),
                (1066237952..1066240000, 1123991552..1123993600, TilePosture::UpperHalf),
                (1066237952..1066242048, 1123991552..1123995648, TilePosture::UpperHalf),
                (1066237952..1066246144, 1123991552..1123999744, TilePosture::UpperHalf),
                (1066237952..1066254336, 1123975168..1123991552, TilePosture::LowerHalf),
                (1066237952..1066254336, 1123975168..1123991552, TilePosture::UpperHalf),
                (1066237952..1066270720, 1123975168..1124007936, TilePosture::UpperHalf),
                (1066139648..1066205184, 1124007936..1124073472, TilePosture::UpperHalf),
                (1066205184..1066270720, 1124007936..1124073472, TilePosture::LowerHalf),
                (1066205184..1066270720, 1124007936..1124073472, TilePosture::UpperHalf),
                (1066270720..1066401792, 1123942400..1124073472, TilePosture::LowerHalf),
                (1066270720..1066401792, 1123942400..1124073472, TilePosture::UpperHalf),
                (1066401792..1067450368, 1123024896..1124073472, TilePosture::LowerHalf),
                (1066401792..1067450368, 1123024896..1124073472, TilePosture::UpperHalf),
                (1067450368..1069547520, 1121976320..1124073472, TilePosture::LowerHalf),
                (1067450368..1069547520, 1121976320..1124073472, TilePosture::UpperHalf),
                (1069547520..1073741824, 1119879168..1124073472, TilePosture::LowerHalf),
                (1069547520..1073741824, 1119879168..1124073472, TilePosture::UpperHalf),
                (1040187392..1056964608, 1124073472..1140850688, TilePosture::UpperHalf),
                (1056964608..1073741824, 1124073472..1140850688, TilePosture::LowerHalf),
                (1056964608..1073741824, 1124073472..1140850688, TilePosture::UpperHalf),
                (939524096..1006632960, 1140850688..1207959552, TilePosture::UpperHalf),
                (1006632960..1073741824, 1140850688..1207959552, TilePosture::LowerHalf),
                (1006632960..1073741824, 1140850688..1207959552, TilePosture::UpperHalf),
                (805306368..939524096, 1207959552..1342177280, TilePosture::UpperHalf),
                (939524096..1073741824, 1207959552..1342177280, TilePosture::LowerHalf),
                (939524096..1073741824, 1207959552..1342177280, TilePosture::UpperHalf),
                (536870912..805306368, 1342177280..1610612736, TilePosture::UpperHalf),
                (805306368..1073741824, 1342177280..1610612736, TilePosture::LowerHalf),
                (805306368..1073741824, 1342177280..1610612736, TilePosture::UpperHalf),
                (0..536870912, 1610612736..2147483648, TilePosture::UpperHalf),   
                (536870912..1073741824, 1610612736..2147483648, TilePosture::LowerHalf),
                (536870912..1073741824, 1610612736..2147483648, TilePosture::UpperHalf),
                (1073741824..2147483648, 1073741824..2147483648, TilePosture::LowerHalf),
                (1073741824..2147483648, 1073741824..2147483648, TilePosture::UpperHalf),
            ])
        }

        #[test]
        fn test_empirical_insert() {
            struct TestCallbacks {
                fills: HashSet<(Range<u32>, Range<u32>, TilePosture)>,
            }
            impl ChangeCallbacks<String> for TestCallbacks {
                fn insert(&mut self, entry: &EntryCore) -> String {
                    self.fills.insert((
                        entry.x_range().clone(),
                        entry.y_range().clone(),
                        entry.posture(),
                    ));
                    println!(
                        "inserting {} {:?}",
                        entry.debug_describe(),
                        crate::game_state::mapgen::far_mesh::to_sheet_control(entry)
                    );
                    entry.debug_describe()
                }
                fn delete(&mut self, value: String) {
                    println!("deleting {}", value);
                }
            }
            let mut tree = TriQuadTree::new(1 << 31);
            let mut callbacks = TestCallbacks {
                fills: HashSet::new(),
            };
            tree.insert_at(1066239547, 1123991713, 4, &mut callbacks);

            let expected = empirical_insert_expected();
            let actual: HashSet<_> = HashSet::from_iter(callbacks.fills);
            println!("Missing but expected: {:?}", expected.difference(&actual));
            println!("Extra but not expected: {:?}", actual.difference(&expected));
            assert_eq!(actual, expected);

            tree.assert_filled();
        }
    }
}
