mod far_mesh {
    use cgmath::vec2;

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

    // We define a fundamental tile as a single (64,64,-1) grid of the fundamental lattice
    //
    // World space:
    //    ^-----/
    //   / \   /
    //  /   \ /
    // O-----v
    //
    // Lattice space:
    //
    // X----X
    // |\   |
    // | \  |
    // |  \ |
    // |   \|
    // O----X
    // Specific to the current implementation of the lattice using triangles

    // Something something right triangular quadtree using barycentric coordinates given as integers
    // with bit-twiddles to traverse

    #[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
    pub(crate) enum TilePosture {
        LowerHalf,
        UpperHalf,
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
    struct LatticeTileCoord {
        // Note that one unit of x and y is a whole fundamental tile. This means
        // that it covers well over (1,1), and is really at least 64x as large
        // as the world dimension, and we don't have to worry about overflow
        // from valid player positions.
        x: u32,
        y: u32,
        posture: TilePosture,
    }

    fn world_lattice_tile(coord: cgmath::Vector2<f64>) -> LatticeTileCoord {
        let lattice_coord = WORLD2LATTICE * vec2(coord.x as f64, coord.y as f64) / 64.0;
        // With 53-ish bits of precision, all of these operations are exact.
        let x = lattice_coord.x.floor() as u32;
        let y = lattice_coord.y.floor() as u32;

        // can't use .fract(), it returns negative values for negative inputs
        // We need a positive fraction
        let xfrac = lattice_coord.x - x as f64;
        let yfrac = lattice_coord.y - y as f64;

        let posture = if yfrac > xfrac {
            TilePosture::LowerHalf
        } else {
            TilePosture::UpperHalf
        };

        LatticeTileCoord { x, y, posture }
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

        use crate::game_state::mapgen::far_mesh::far_mesh::TilePosture;

        slotmap::new_key_type! {
            struct NodeKey;
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
                println!("node ({},{},{},{})", x, y, dense_mask, leading_mask);
                match self.nodes.get(slot).unwrap() {
                    TriQuadNode::Leaf(_) | TriQuadNode::EmptyLeaf => {
                        EntryCore::new(slot, x, y, dense_mask)
                    }
                    TriQuadNode::Internal { rect, tris } => {
                        let cx = (x & leading_mask) != 0;
                        let cy = (y & leading_mask) != 0;

                        println!("cx {} cy {}", cx, cy);
                        match (cx, cy) {
                            (true, false) => self.traverse_nodes(
                                tris[0],
                                x,
                                y,
                                dense_mask >> 1,
                                leading_mask >> 1,
                            ),
                            (false, true) => self.traverse_nodes(
                                tris[1],
                                x,
                                y,
                                dense_mask >> 1,
                                leading_mask >> 1,
                            ),
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
                println!("pair ({},{},{},{})", x, y, dense_mask, leading_mask);
                let sum = (x & dense_mask) + (y & dense_mask);
                println!("sum {} = {} + {}", sum, x & dense_mask, y & dense_mask);
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
                println!(
                    "{{ fillnode ({},{},{},{} -> {})",
                    x, y, dense_mask, leading_mask, stop_side_length
                );
                // At the leaf we wanted to reach.
                if leading_mask <= stop_side_length {
                    println!("base case");
                    let entry = EntryCore {
                        x,
                        y,
                        posture: current_posture,
                        dense_mask,
                        node: slot,
                    };
                    let data = callbacks.insert(&entry);
                    let old = std::mem::replace(
                        self.nodes.get_mut(slot).unwrap(),
                        TriQuadNode::Leaf(data),
                    );
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

                    let cx = (x & leading_mask) != 0;
                    let cy = (y & leading_mask) != 0;
                    let dense_node = match (cx, cy) {
                        (true, false) => next_tris[0],
                        (false, true) => next_tris[1],
                        (false, false) | (true, true) => {
                            // The decision of triangle orientation needs to happen
                            // with a smaller mask. Compare to how traverse_nodes halves the mask
                            // when calling traverse_rects, then traverse_rects passes that same
                            // halved mask to traverse_nodes.
                            let half_mask = dense_mask >> 1;
                            let sum = (x & half_mask) + (y & half_mask);
                            if sum < half_mask {
                                next_rect.lower
                            } else {
                                next_rect.upper
                            }
                        }
                    };
                    let xmin = x & !dense_mask;
                    let ymin = y & !dense_mask;
                    let xmax = x | dense_mask;
                    let ymax = y | dense_mask;
                    let rect_coord = match current_posture {
                        TilePosture::LowerHalf => (xmin, ymin),
                        TilePosture::UpperHalf => (xmax, ymax),
                    };
                    for (node, coord, posture, debug) in [
                        (next_tris[0], (xmax, ymin), current_posture, "t0"),
                        (next_tris[1], (xmin, ymax), current_posture, "t1"),
                        (next_rect.lower, rect_coord, TilePosture::LowerHalf, "lower"),
                        (next_rect.upper, rect_coord, TilePosture::UpperHalf, "upper"),
                    ] {
                        println!("traverse {}", debug);
                        let stop_mask = if node == dense_node {
                            println!("will recurse");
                            stop_side_length
                        } else {
                            println!("will not recurse");
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

                println!("}}");
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
                    self.x_range().start,
                    self.x_range().end,
                    self.y_range().start,
                    self.y_range().end
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
                        println!("inserting {}", entry.debug_describe());
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
        }
    }
}
