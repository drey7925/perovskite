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

        /// Callback that is called when a node is inserted or deleted.
        pub(crate) trait ChangeCallbacks<T: Default + Send + Sync + 'static> {
            fn insert(&mut self, entry: &EntryCore) -> T;
            fn delete(&mut self, entry: &TriQuadEntryMut<T>);
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
            pub(crate) fn new(grid_size: u32) -> anyhow::Result<Self> {
                if grid_size == u32::MAX {
                    return Err(anyhow::anyhow!("grid_size must be less than u32::MAX"));
                }
                if grid_size & (grid_size - 1) != 0 {
                    return Err(anyhow::anyhow!("grid_size must be a power of 2"));
                }
                let mut nodes = slotmap::SlotMap::with_key();
                let upper = nodes.insert(TriQuadNode::EmptyLeaf);
                let lower = nodes.insert(TriQuadNode::EmptyLeaf);
                Ok(Self {
                    root: TriQuadPair { upper, lower },
                    nodes,
                    grid_size,
                })
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
                    TriQuadNode::Leaf(_) | TriQuadNode::EmptyLeaf => EntryCore {
                        x,
                        y,
                        posture: if (x & dense_mask) + (y & dense_mask) < dense_mask {
                            TilePosture::LowerHalf
                        } else {
                            TilePosture::UpperHalf
                        },
                        dense_mask,
                        node: slot,
                    },
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

        impl<T: Default + Send + Sync + 'static> TriQuadTree<T> {
            /// Insert a node at the given position, with the given side length. If there is already
            /// finer geometry at this location, do nothing (the finer geometry takes precedence).
            ///
            /// Otherwise, insert the node at this level, call the callbacks, then recursively
            /// insert neighbors to fill nodes on the way back up, with the most coarse representation
            /// that will fit.
            fn insert_at(
                &mut self,
                x: u32,
                y: u32,
                side_length: u32,
                callbacks: &mut impl ChangeCallbacks<T>,
            ) {
                let starting_entry = self.core_entry(x, y);
                if starting_entry.side_length() < side_length {
                    return;
                } else if starting_entry.side_length() == side_length {
                    todo!("Fill single");
                } else {
                    todo!("Fill multiple then recurse back");
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
                self.x & !self.dense_mask..self.x | self.dense_mask
            }
            pub(crate) fn y_range(&self) -> Range<u32> {
                self.y & !self.dense_mask..self.y | self.dense_mask
            }
            pub(crate) fn debug_describe(&self) -> String {
                format!(
                    "{} half of [{},{}) x [{},{})",
                    match self.posture {
                        TilePosture::LowerHalf => "Lower",
                        TilePosture::UpperHalf => "Upper",
                    },
                    self.x_range().start,
                    self.x_range().end + 1,
                    self.y_range().start,
                    self.y_range().end + 1
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
            use std::cell::RefCell;

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
        }
    }
}
