// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

use super::game_map::MapChunk;
use crate::game_state::event::EventInitiator;
use perovskite_core::coordinates::{BlockCoordinate, ChunkCoordinate};
use std::ops::RangeInclusive;

pub trait MapgenInterface: Send + Sync {
    /// Generate a chunk using this map generator.
    ///
    /// Args:
    ///     coord: The coordinate of the chunk to generate.
    ///     chunk: The chunk to fill.
    ///
    /// Implementations should strive to remain consistent, even over
    /// different versions of the same code. The requisite level of consistency
    /// is subjective and depends on scale - e.g. noise that affects major topography (like elevation)
    /// should be consistent to avoid massive jumps in the map, whereas
    /// inconsistent noise that affects minor details (like trees) will have a smaller effect
    /// on the gameplay experience, and inconsistent IID noise that affects single blocks (like
    /// ore generation) will probably not be noticed.
    ///
    /// ^ Caveat: This consistency is mostly important for gameplay experience, and to a small
    /// extent performance/view distance (due to terrain_range_hint). However, sudden jumps and
    /// large inconsistencies are OK if they match your intended gameplay/style, or if you need
    /// to make a major change and accept such inconsistencies as a consequence of that change.
    ///
    /// Users of the map must NOT assume block-for-block consistency between calls to fill_chunk
    /// (noting that it will likely be called only once, barring crashes that cause unsaved
    /// changes to the map).
    fn fill_chunk(&self, coord: ChunkCoordinate, chunk: &mut MapChunk);

    /// Provide an estimate of chunk coordinate Y values (one value -> 16 blocks) where terrain
    /// is most likely to be seen at the given (X, Z) chunk coordinate, or None for no hint
    ///
    /// This is used for speeding up chunk loading. Favor speed over exact precision, but prefer to
    /// err on the side of including a chunk in the range if unsure.
    fn terrain_range_hint(&self, _chunk_x: i32, _chunk_z: i32) -> Option<RangeInclusive<i32>> {
        None
    }

    /// Prints debugging information regarding map generation. The definition of this is up to
    /// the implementor, and can include whatever information is most useful for developing this
    /// specific mapgen.
    ///
    /// By default, does nothing.
    fn dump_debug(&self, _pos: BlockCoordinate, _initiator: &EventInitiator<'_>) {}
}

mod far_mesh {
    use std::ops::Deref;

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
    // In diagrams below, O represents the origin.
    //
    // The fundamental tile with tile coordinate (0,0,Up) has one corner at the origin, and support on the
    // first quadrant (positive x and y), see callout A below.
    //
    // The fundamental tile with tile coordinate (0,0,Down) is adjacent to it, making a parallelogram in the
    // first quadrant, see callout B below.
    //
    // World space:
    //    ^-----/
    //   / \ B /
    //  / A \ /
    // O----=v
    //
    // Lattice space:
    //
    // X----X
    // |\ B |
    // | \  |
    // |  \ |
    // | A \|
    // O----X
    // Specific to the current implementation of the lattice using triangles

    // Something something right triangular quadtree using barycentric coordinates given as integers
    // with bit-twiddles to traverse

    #[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
    enum TilePosture {
        Up,
        Down,
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

    /// A node of the triangular quadtree.
    enum TriQuadNode<T> {
        EmptyLeaf,
        Leaf(T),
        Internal {
            rect: Box<TriQuadPair<T>>,
            tris: [Box<TriQuadNode<T>>; 2],
        },
    }

    impl<T: Clone> TriQuadNode<T> {
        fn traverse(&self, x: u32, y: u32, dense_mask: u32, leading_mask: u32) -> Option<&T> {
            println!("node ({},{},{},{})", x, y, dense_mask, leading_mask);
            match self {
                TriQuadNode::Leaf(t) => return Some(t),
                TriQuadNode::EmptyLeaf => return None,
                TriQuadNode::Internal { rect, tris } => {
                    let cx = (x & leading_mask) != 0;
                    let cy = (y & leading_mask) != 0;

                    println!("cx {} cy {}", cx, cy);
                    match (cx, cy) {
                        (true, false) => {
                            return tris[0].traverse(x, y, dense_mask >> 1, leading_mask >> 1);
                        }
                        (false, true) => {
                            return tris[1].traverse(x, y, dense_mask >> 1, leading_mask >> 1);
                        }

                        (false, false) | (true, true) => {
                            return rect.traverse_impl(x, y, dense_mask >> 1, leading_mask >> 1);
                        }
                    };
                }
            }
        }
    }

    struct TriQuadPair<T> {
        upper: TriQuadNode<T>,
        lower: TriQuadNode<T>,
    }

    impl<T: Clone> TriQuadPair<T> {
        fn traverse(&self, x: u32, y: u32, grid_size: u32) -> Option<&T> {
            // Start with a mask
            assert!(grid_size != u32::MAX);
            assert!(grid_size & (grid_size - 1) == 0);
            let dense_mask = grid_size - 1;
            let leading_mask = grid_size >> 1;
            self.traverse_impl(x, y, dense_mask, leading_mask)
        }
        fn traverse_impl(&self, x: u32, y: u32, dense_mask: u32, leading_mask: u32) -> Option<&T> {
            println!("pair ({},{},{},{})", x, y, dense_mask, leading_mask);
            let sum = (x & dense_mask) + (y & dense_mask);
            println!("sum {} = {} + {}", sum, x & dense_mask, y & dense_mask);
            let tqn = if sum < dense_mask {
                &self.lower
            } else {
                &self.upper
            };
            tqn.traverse(x, y, dense_mask, leading_mask)
        }
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
            TilePosture::Down
        } else {
            TilePosture::Up
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

        #[test]
        fn test_traversal_simple_pair() {
            let tree = TriQuadPair::<char> {
                lower: TriQuadNode::Leaf('a'),
                upper: TriQuadNode::Leaf('b'),
            };
            assert_eq!(tree.traverse(12, 12, 16), Some(&'b'));
            assert_eq!(tree.traverse(3, 10, 16), Some(&'a'));
        }

        #[test]
        fn test_traversal_one_deep() {
            let tree = TriQuadPair::<char> {
                lower: TriQuadNode::Leaf('a'),
                upper: TriQuadNode::Internal {
                    rect: Box::new(TriQuadPair {
                        lower: TriQuadNode::Leaf('e'),
                        upper: TriQuadNode::Leaf('j'),
                    }),
                    tris: [
                        Box::new(TriQuadNode::Leaf('b')),
                        Box::new(TriQuadNode::Leaf('c')),
                    ],
                },
            };
            println!("j");
            assert_eq!(tree.traverse(15, 15, 16), Some(&'j'));
            println!("e");
            assert_eq!(tree.traverse(11, 11, 16), Some(&'e'));
            println!("c");
            assert_eq!(tree.traverse(1, 15, 16), Some(&'c'));
            println!("b");
            assert_eq!(tree.traverse(1, 1, 16), Some(&'a'));
            println!("b");
            assert_eq!(tree.traverse(15, 1, 16), Some(&'b'));
            assert_eq!(tree.traverse(15, 7, 16), Some(&'b'));
        }

        #[test]
        fn test_traversal_two_deep() {
            let tree = TriQuadPair::<char> {
                lower: TriQuadNode::Leaf('a'),
                upper: TriQuadNode::Internal {
                    rect: Box::new(TriQuadPair {
                        lower: TriQuadNode::Internal {
                            rect: Box::new(TriQuadPair {
                                lower: TriQuadNode::Leaf('e'),
                                upper: TriQuadNode::Leaf('f'),
                            }),
                            tris: [
                                Box::new(TriQuadNode::Leaf('g')),
                                Box::new(TriQuadNode::Leaf('h')),
                            ],
                        },
                        upper: TriQuadNode::Leaf('j'),
                    }),
                    tris: [
                        Box::new(TriQuadNode::Leaf('b')),
                        Box::new(TriQuadNode::Leaf('c')),
                    ],
                },
            };
            println!("j");
            assert_eq!(tree.traverse(15, 15, 16), Some(&'j'));
            println!("g");
            assert_eq!(tree.traverse(12, 8, 16), Some(&'g'));
            println!("h");
            assert_eq!(tree.traverse(8, 12, 16), Some(&'h'));
            println!("e");
            assert_eq!(tree.traverse(9, 9, 16), Some(&'e'));
            println!("f");
            assert_eq!(tree.traverse(11, 10, 16), Some(&'f'));
            println!("c");
            assert_eq!(tree.traverse(1, 15, 16), Some(&'c'));
            println!("b");
            assert_eq!(tree.traverse(1, 1, 16), Some(&'a'));
            println!("b");
            assert_eq!(tree.traverse(15, 1, 16), Some(&'b'));
            assert_eq!(tree.traverse(15, 7, 16), Some(&'b'));
        }
    }
}
