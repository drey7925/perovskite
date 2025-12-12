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
    enum TilePosture {
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

    pub(crate) mod tri_quad {
        use crate::game_state::mapgen::far_mesh::far_mesh::TilePosture;

        slotmap::new_key_type! {
            struct NodeKey;
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

            pub(crate) fn get(&self, x: u32, y: u32) -> Option<&T> {
                let entry =
                    self.traverse_rects(&self.root, x, y, self.grid_size - 1, self.grid_size >> 1);
                if let Some(entry) = entry {
                    match self.nodes.get(entry.node).unwrap() {
                        TriQuadNode::Leaf(t) => Some(t),
                        _ => None,
                    }
                } else {
                    None
                }
            }

            fn traverse_nodes(
                &self,
                slot: NodeKey,
                x: u32,
                y: u32,
                dense_mask: u32,
                leading_mask: u32,
            ) -> Option<TriQuadEntry<T>> {
                println!("node ({},{},{},{})", x, y, dense_mask, leading_mask);
                match self.nodes.get(slot).unwrap() {
                    TriQuadNode::Leaf(t) => {
                        return Some(TriQuadEntry {
                            x,
                            y,
                            // TODO: determine posture
                            posture: TilePosture::LowerHalf,
                            dense_mask,
                            node: slot,
                            phantom: std::marker::PhantomData,
                        });
                    }
                    TriQuadNode::EmptyLeaf => return None,
                    TriQuadNode::Internal { rect, tris } => {
                        let cx = (x & leading_mask) != 0;
                        let cy = (y & leading_mask) != 0;

                        println!("cx {} cy {}", cx, cy);
                        match (cx, cy) {
                            (true, false) => {
                                return self.traverse_nodes(
                                    tris[0],
                                    x,
                                    y,
                                    dense_mask >> 1,
                                    leading_mask >> 1,
                                );
                            }
                            (false, true) => {
                                return self.traverse_nodes(
                                    tris[1],
                                    x,
                                    y,
                                    dense_mask >> 1,
                                    leading_mask >> 1,
                                );
                            }

                            (false, false) | (true, true) => {
                                return self.traverse_rects(
                                    rect,
                                    x,
                                    y,
                                    dense_mask >> 1,
                                    leading_mask >> 1,
                                );
                            }
                        };
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
            ) -> Option<TriQuadEntry<T>> {
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

        pub(crate) struct TriQuadEntry<T> {
            x: u32,
            y: u32,
            posture: TilePosture,
            dense_mask: u32,
            node: NodeKey,
            phantom: std::marker::PhantomData<T>,
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

            #[test]
            fn test_traversal_simple_pair() {
                let nodes = RefCell::new(slotmap::SlotMap::with_key());
                let root = make_pair(&nodes, TriQuadNode::Leaf('a'), TriQuadNode::Leaf('b'));
                let tree = TriQuadTree {
                    root,
                    nodes: nodes.into_inner(),
                    grid_size: 16,
                };
                assert_eq!(tree.get(12, 12), Some(&'b'));
                assert_eq!(tree.get(3, 10), Some(&'a'));
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
                assert_eq!(tree.get(15, 15), Some(&'j'));
                println!("e");
                assert_eq!(tree.get(11, 11), Some(&'e'));
                println!("c");
                assert_eq!(tree.get(1, 15), Some(&'c'));
                println!("b");
                assert_eq!(tree.get(1, 1), Some(&'a'));
                println!("b");
                assert_eq!(tree.get(15, 1), Some(&'b'));
                assert_eq!(tree.get(15, 7), Some(&'b'));
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
                assert_eq!(tree.get(15, 15), Some(&'j'));
                println!("g");
                assert_eq!(tree.get(12, 8), Some(&'g'));
                println!("h");
                assert_eq!(tree.get(8, 12), Some(&'h'));
                println!("e");
                assert_eq!(tree.get(9, 9), Some(&'e'));
                println!("f");
                assert_eq!(tree.get(11, 10), Some(&'f'));
                println!("c");
                assert_eq!(tree.get(1, 15), Some(&'c'));
                println!("b");
                assert_eq!(tree.get(1, 1), Some(&'a'));
                println!("b");
                assert_eq!(tree.get(15, 1), Some(&'b'));
                assert_eq!(tree.get(15, 7), Some(&'b'));
            }
        }
    }
}
