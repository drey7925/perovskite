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
pub(crate) const FINEST_RENDERED_SIZE: u32 = 4;
// This is 16x larger, with 37-block stride, roughly.
pub(crate) const COARSEST_RENDERED_SIZE: u32 = 8; // Changed from 64 for ease of testing

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
    let origin = coarse_basis_u * (coord.x_range().start as f64 + i32::MIN as f64)
        + coarse_basis_v * (coord.y_range().start as f64 + y_correction + i32::MIN as f64);

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
    let lattice = (WORLD2LATTICE * pos / 32.0) - vec2(i32::MIN as f64, i32::MIN as f64);
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
            match self.nodes.get(slot).unwrap() {
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
            let sum = (x & dense_mask) + (y & dense_mask);
            let node = if sum < dense_mask {
                rect.lower
            } else {
                rect.upper
            };
            self.traverse_nodes(node, x, y, dense_mask, leading_mask)
        }
    }

    impl<T: Send + Sync + 'static> TriQuadTree<T> {
        fn traverse_and_fill_nodes(
            &mut self,
            slot: NodeKey,
            x: u32,
            y: u32,
            dense_mask: u32,
            leading_mask: u32,
            stop_side_length: u32,
            callbacks: &mut impl ChangeCallbacks<T>,
            current_posture: TilePosture,
        ) {
            use std::sync::atomic::AtomicUsize;
            use std::sync::atomic::Ordering;
            static DEPTH: AtomicUsize = AtomicUsize::new(0);
            let depth = DEPTH.fetch_add(1, Ordering::SeqCst);
            println!(
                "{} fillnode ({},{},{},{} -> {})",
                " ".repeat(depth),
                (x as i32).wrapping_sub(i32::MIN),
                (y as i32).wrapping_sub(i32::MIN),
                dense_mask,
                leading_mask,
                stop_side_length
            );
            // At the leaf we wanted to reach.
            if leading_mask <= stop_side_length {
                let entry = EntryCore {
                    x,
                    y,
                    posture: current_posture,
                    dense_mask,
                    node: slot,
                };
                let data = callbacks.insert(&entry);
                let old =
                    std::mem::replace(self.nodes.get_mut(slot).unwrap(), TriQuadNode::Leaf(data));
                self.remove_detached_tree(old, callbacks);
            } else {
                // Not at the leaf we wanted to reach. Inspect what we have,
                // then traverse down to the leaf we want to reach.
                let node = self.nodes.get(slot).unwrap();
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
                        let old_node =
                            std::mem::replace(self.nodes.get_mut(slot).unwrap(), new_node);
                        self.remove_detached_tree(old_node, callbacks);
                    }
                    TriQuadNode::Internal { rect, tris } => {
                        next_rect = *rect;
                        next_tris = *tris;
                    }
                }
                let xmin = x & !dense_mask;
                let ymin = y & !dense_mask;
                let xmax = x | dense_mask;
                let ymax = y | dense_mask;
                println!(
                    "{} Considering x [{},{}) y [{},{}) dense_mask {} leading_mask {} stop_side_length {}",
                    " ".repeat(depth),
                    (xmin as i32).wrapping_sub(i32::MIN),
                    (xmax as i32).wrapping_sub(i32::MIN),
                    (ymin as i32).wrapping_sub(i32::MIN),
                    (ymax as i32).wrapping_sub(i32::MIN),
                    dense_mask,
                    leading_mask,
                    stop_side_length
                );

                let cx = (x & leading_mask >> 1) != 0;
                let cy = (y & leading_mask >> 1) != 0;
                let dense_node = match (cx, cy) {
                    (true, false) => {
                        println!("{} dense nt0", " ".repeat(depth));
                        next_tris[0]
                    }
                    (false, true) => {
                        println!("{} dense nt1", " ".repeat(depth));
                        next_tris[1]
                    }
                    (false, false) | (true, true) => {
                        // The decision of triangle orientation needs to happen
                        // with a smaller mask. Compare to how traverse_nodes halves the mask
                        // when calling traverse_rects, then traverse_rects passes that same
                        // halved mask to traverse_nodes.
                        let half_mask = dense_mask >> 1;
                        let sum = (x & half_mask) + (y & half_mask);
                        if sum < half_mask {
                            println!("{} dense rl", " ".repeat(depth));
                            next_rect.lower
                        } else {
                            println!("{} dense ru", " ".repeat(depth));
                            next_rect.upper
                        }
                    }
                };

                println!(
                    "{} Current posture: {:?}",
                    " ".repeat(depth),
                    current_posture
                );
                let rect_coord = match current_posture {
                    TilePosture::LowerHalf => (xmin, ymin),
                    TilePosture::UpperHalf => (xmax, ymax),
                };
                for (node, coord, posture) in [
                    (next_tris[0], (xmax, ymin), current_posture),
                    (next_tris[1], (xmin, ymax), current_posture),
                    (next_rect.lower, rect_coord, TilePosture::LowerHalf),
                    (next_rect.upper, rect_coord, TilePosture::UpperHalf),
                ] {
                    let stop_mask = if node == dense_node {
                        println!("{} dense node", " ".repeat(depth));
                        stop_side_length
                    } else {
                        leading_mask >> 1
                    };
                    let coord = if node == dense_node { (x, y) } else { coord };
                    self.traverse_and_fill_nodes(
                        node,
                        coord.0,
                        coord.1,
                        dense_mask >> 1,
                        leading_mask >> 1,
                        stop_mask,
                        callbacks,
                        posture,
                    );
                }
            }
            DEPTH.fetch_sub(1, Ordering::SeqCst);
        }

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
        /// that will fit. This will insert at every intermediate level EXCEPT the uppermost
        /// diagonal split (i.e. if the position ends up in the lower half-plane, then the giant
        /// triangle for the upper half-plane will not be inserted).
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
            assert!(side_length.is_power_of_two());
            let starting_entry = self.core_entry(x, y);
            if starting_entry.side_length() < side_length {
                // Finer geometry already present, do nothing.
                return;
            } else if starting_entry.side_length() == side_length {
                // Edge case, exact match. As a precondition, it should already be present, because
                // the fill case should have filled in siblings on its way back up.
                self.assert_key_filled(starting_entry.node);
                return;
            } else {
                self.traverse_and_fill_nodes(
                    starting_entry.node,
                    x,
                    y,
                    starting_entry.dense_mask,
                    starting_entry.dense_mask + 1,
                    side_length,
                    callbacks,
                    starting_entry.posture,
                );
            }
        }

        fn assert_key_filled(&self, key: NodeKey) {
            match self.nodes.get(key).unwrap() {
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
            let lower = self.nodes.get(self.root.lower).unwrap();
            let upper = self.nodes.get(self.root.upper).unwrap();
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
            EntryCore {
                x,
                y,
                posture: if (x & dense_mask) + (y & dense_mask) < dense_mask {
                    TilePosture::LowerHalf
                } else {
                    TilePosture::UpperHalf
                },
                dense_mask,
                node: slot,
            }
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
            self.map.get(self.core.node).unwrap()
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
                "{} half of [{},{}) x [{},{})",
                match self.posture {
                    TilePosture::LowerHalf => "Lower",
                    TilePosture::UpperHalf => "Upper",
                },
                (self.x_range().start as i32).wrapping_add(i32::MIN),
                (self.x_range().end as i32).wrapping_add(i32::MIN),
                (self.y_range().start as i32).wrapping_add(i32::MIN),
                (self.y_range().end as i32).wrapping_add(i32::MIN)
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
            self.map.get(self.core.node).unwrap()
        }
        pub(crate) fn value(&self) -> Option<&T> {
            match self.node() {
                TriQuadNode::Leaf(value) => Some(value),
                _ => None,
            }
        }
        fn node_mut(&mut self) -> &mut TriQuadNode<T> {
            self.map.get_mut(self.core.node).unwrap()
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
            println!("{}", entry.core().debug_describe());
            assert_eq!(entry.value(), expected);
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
            assert_eq!(callbacks.fills.len(), 7);

            let expected_fills = [
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
            assert_eq!(callbacks.fills.len(), expected_fills.len());
            assert_eq!(callbacks.fills, HashSet::from_iter(expected_fills));

            tree.assert_filled();
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
            tree.insert_at(2147483663, 2147483649, 4, &mut callbacks);

            let expected_fills = [
                (
                    3221225472..0,
                    2147483648..3221225472,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..3221225472,
                    3221225472..0,
                    TilePosture::LowerHalf,
                ),
                (
                    2684354560..3221225472,
                    2147483648..2684354560,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2684354560,
                    2684354560..3221225472,
                    TilePosture::LowerHalf,
                ),
                (
                    2415919104..2684354560,
                    2147483648..2415919104,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2415919104,
                    2415919104..2684354560,
                    TilePosture::LowerHalf,
                ),
                (
                    2281701376..2415919104,
                    2147483648..2281701376,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2281701376,
                    2281701376..2415919104,
                    TilePosture::LowerHalf,
                ),
                (
                    2214592512..2281701376,
                    2147483648..2214592512,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2214592512,
                    2214592512..2281701376,
                    TilePosture::LowerHalf,
                ),
                (
                    2181038080..2214592512,
                    2147483648..2181038080,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2181038080,
                    2181038080..2214592512,
                    TilePosture::LowerHalf,
                ),
                (
                    2164260864..2181038080,
                    2147483648..2164260864,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2164260864,
                    2164260864..2181038080,
                    TilePosture::LowerHalf,
                ),
                (
                    2155872256..2164260864,
                    2147483648..2155872256,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2155872256,
                    2155872256..2164260864,
                    TilePosture::LowerHalf,
                ),
                (
                    2151677952..2155872256,
                    2147483648..2151677952,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2151677952,
                    2151677952..2155872256,
                    TilePosture::LowerHalf,
                ),
                (
                    2149580800..2151677952,
                    2147483648..2149580800,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2149580800,
                    2149580800..2151677952,
                    TilePosture::LowerHalf,
                ),
                (
                    2148532224..2149580800,
                    2147483648..2148532224,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2148532224,
                    2148532224..2149580800,
                    TilePosture::LowerHalf,
                ),
                (
                    2148007936..2148532224,
                    2147483648..2148007936,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2148007936,
                    2148007936..2148532224,
                    TilePosture::LowerHalf,
                ),
                (
                    2147745792..2148007936,
                    2147483648..2147745792,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147745792,
                    2147745792..2148007936,
                    TilePosture::LowerHalf,
                ),
                (
                    2147614720..2147745792,
                    2147483648..2147614720,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147614720,
                    2147614720..2147745792,
                    TilePosture::LowerHalf,
                ),
                (
                    2147549184..2147614720,
                    2147483648..2147549184,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147549184,
                    2147549184..2147614720,
                    TilePosture::LowerHalf,
                ),
                (
                    2147516416..2147549184,
                    2147483648..2147516416,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147516416,
                    2147516416..2147549184,
                    TilePosture::LowerHalf,
                ),
                (
                    2147500032..2147516416,
                    2147483648..2147500032,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147500032,
                    2147500032..2147516416,
                    TilePosture::LowerHalf,
                ),
                (
                    2147491840..2147500032,
                    2147483648..2147491840,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147491840,
                    2147491840..2147500032,
                    TilePosture::LowerHalf,
                ),
                (
                    2147487744..2147491840,
                    2147483648..2147487744,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147487744,
                    2147487744..2147491840,
                    TilePosture::LowerHalf,
                ),
                (
                    2147485696..2147487744,
                    2147483648..2147485696,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147485696,
                    2147485696..2147487744,
                    TilePosture::LowerHalf,
                ),
                (
                    2147484672..2147485696,
                    2147483648..2147484672,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147484672,
                    2147484672..2147485696,
                    TilePosture::LowerHalf,
                ),
                (
                    2147484160..2147484672,
                    2147483648..2147484160,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147484160,
                    2147484160..2147484672,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483904..2147484160,
                    2147483648..2147483904,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147483904,
                    2147483904..2147484160,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483776..2147483904,
                    2147483648..2147483776,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147483776,
                    2147483776..2147483904,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483712..2147483776,
                    2147483648..2147483712,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147483712,
                    2147483712..2147483776,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483680..2147483712,
                    2147483648..2147483680,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147483680,
                    2147483680..2147483712,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483664..2147483680,
                    2147483648..2147483664,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147483664,
                    2147483664..2147483680,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483648..2147483664,
                    2147483648..2147483664,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483660..2147483664,
                    2147483648..2147483652,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483656..2147483660,
                    2147483652..2147483656,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483660..2147483664,
                    2147483652..2147483656,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483660..2147483664,
                    2147483652..2147483656,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147483656,
                    2147483656..2147483664,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483656..2147483664,
                    2147483656..2147483664,
                    TilePosture::LowerHalf,
                ),
                (
                    2147483656..2147483664,
                    2147483656..2147483664,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147483680,
                    2147483648..2147483680,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147483712,
                    2147483648..2147483712,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147483776,
                    2147483648..2147483776,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147483904,
                    2147483648..2147483904,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147484160,
                    2147483648..2147484160,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147484672,
                    2147483648..2147484672,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147485696,
                    2147483648..2147485696,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147487744,
                    2147483648..2147487744,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147491840,
                    2147483648..2147491840,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147500032,
                    2147483648..2147500032,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147516416,
                    2147483648..2147516416,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147549184,
                    2147483648..2147549184,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147614720,
                    2147483648..2147614720,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2147745792,
                    2147483648..2147745792,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2148007936,
                    2147483648..2148007936,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2148532224,
                    2147483648..2148532224,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2149580800,
                    2147483648..2149580800,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2151677952,
                    2147483648..2151677952,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2155872256,
                    2147483648..2155872256,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2164260864,
                    2147483648..2164260864,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2181038080,
                    2147483648..2181038080,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2214592512,
                    2147483648..2214592512,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2281701376,
                    2147483648..2281701376,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2415919104,
                    2147483648..2415919104,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..2684354560,
                    2147483648..2684354560,
                    TilePosture::UpperHalf,
                ),
                (
                    2147483648..3221225472,
                    2147483648..3221225472,
                    TilePosture::UpperHalf,
                ),
            ];
            assert_eq!(callbacks.fills.len(), expected_fills.len());
            assert_eq!(callbacks.fills, HashSet::from_iter(expected_fills));

            tree.assert_filled();
        }
    }
}
