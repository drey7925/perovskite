use std::{collections::HashSet, ops::Deref, sync::Arc, time::Duration};

use anyhow::Result;
use cgmath::InnerSpace;
use cuberef_core::{
    block_id::BlockId,
    coordinates::{ChunkCoordinate, ChunkOffset},
};
use parking_lot::{Condvar, Mutex};
use rustc_hash::FxHashSet;
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::{
    block_renderer::{BlockRenderer, ClientBlockTypeManager},
    game_state::{
        chunk::{self, ChunkOffsetExt, ClientChunk},
        ChunkManagerClonedView, ClientState, FastChunkNeighbors, FastNeighborLockCache,
    },
};

// Responsible for reconciling a chunk with data from other nearby chunks (e.g. lighting, neighbor calculations)
pub(crate) struct NeighborPropagator {
    pub(crate) client_state: Arc<ClientState>,
    pub(crate) queue: Mutex<FxHashSet<ChunkCoordinate>>,
    pub(crate) cond: Condvar,
    pub(crate) shutdown: CancellationToken,
}
impl NeighborPropagator {
    pub(crate) fn run_neighbor_propagator(self: Arc<Self>) -> Result<()> {
        tracy_client::set_thread_name!("async_neighbor_propagator");
        let mut scratchpad = Box::new([0u8; 48 * 48 * 48]);
        while !self.shutdown.is_cancelled() {
            // This is duplicated in MeshWorker to allow the two to use different strategies, and also
            // report different span and plot names.
            let chunks = {
                // This is really ugly and deadlock-prone. Figure out whether we need this or not.
                // The big deadlock risk is if this line happens under the queue lock:
                //   our thread waits for physics_state to unlock
                //   physics_state waits for mapchunks to unlock
                //   network thread is holding chunks and waiting for queue to unlock
                //
                // At the moment, the deadlocks are removed, but this still seems brittle.
                let pos = self.client_state.last_position().position;
                let mut lock = self.queue.lock();
                if lock.is_empty() {
                    plot!("nprop_queue_length", 0.);
                    self.cond.wait_for(&mut lock, Duration::from_secs(1));
                }

                let _span = span!("mesh_worker sort");
                let mut chunks: Vec<_> = lock.iter().copied().collect();

                plot!("nprop_queue_length", chunks.len() as f64);

                // Prioritize the closest chunks
                chunks.sort_by_key(|x| {
                    let center = x.with_offset(ChunkOffset { x: 8, y: 8, z: 8 });
                    let offset =
                        cgmath::vec3(center.x as f64, center.y as f64, center.z as f64) - pos;
                    offset.magnitude2() as u64
                });
                chunks.shrink_to(MESH_BATCH_SIZE);
                for coord in chunks.iter() {
                    assert!(lock.remove(coord), "Task should have been in the queue");
                }
                drop(lock);
                chunks
            };
            {
                let _span = span!("nprop_work");
                for coord in chunks {
                    propagate_neighbor_data(
                        &self.client_state.block_types,
                        &self.client_state.chunks.cloned_neighbors_fast(coord),
                        &mut scratchpad,
                    )?;
                }
            }
        }
        Ok(())
    }
}

// Responsible for turning a single chunk into a mesh
pub(crate) struct MeshWorker {
    pub(crate) client_state: Arc<ClientState>,
    pub(crate) queue: Mutex<FxHashSet<ChunkCoordinate>>,
    pub(crate) cond: Condvar,
    pub(crate) shutdown: CancellationToken,
}
impl MeshWorker {
    pub(crate) fn run_mesh_worker(self: Arc<Self>) -> Result<()> {
        let mut scratchpad = Box::new([0u8; 48 * 48 * 48]);
        tracy_client::set_thread_name!("async_mesh_worker");
        while !self.shutdown.is_cancelled() {
            let chunks = {
                // This is really ugly and deadlock-prone. Figure out whether we need this or not.
                // The big deadlock risk is if this line happens under the queue lock:
                //   our thread waits for physics_state to unlock
                //   physics_state waits for mapchunks to unlock
                //   network thread is holding chunks and waiting for queue to unlock
                //
                // At the moment, the deadlocks are removed, but this still seems brittle.
                let pos = self.client_state.last_position().position;
                let mut lock = self.queue.lock();
                if lock.is_empty() {
                    plot!("mesh_queue_length", 0.);
                    self.cond.wait_for(&mut lock, Duration::from_secs(1));
                }

                let _span = span!("mesh_worker sort");
                let mut chunks: Vec<_> = lock.iter().copied().collect();

                plot!("mesh_queue_length", chunks.len() as f64);

                // Prioritize the closest chunks
                chunks.sort_by_key(|x| {
                    let center = x.with_offset(ChunkOffset { x: 8, y: 8, z: 8 });
                    let offset =
                        cgmath::vec3(center.x as f64, center.y as f64, center.z as f64) - pos;
                    offset.magnitude2() as u64
                });
                chunks.shrink_to(MESH_BATCH_SIZE);
                for coord in chunks.iter() {
                    assert!(lock.remove(coord), "Task should have been in the queue");
                }
                drop(lock);
                chunks
            };
            {
                let _span = span!("mesh_worker work");
                for coord in chunks {
                    let neighbors = &self.client_state.chunks.cloned_neighbors_fast(coord);
                    propagate_neighbor_data(
                        &self.client_state.block_types,
                        neighbors,
                        &mut scratchpad,
                    )?;
                    if let Some(chunk) = neighbors.get((0, 0, 0)) {
                        chunk.mesh_with(&self.client_state.block_renderer)?;
                    }
                }
            }
        }
        Ok(())
    }
}
const MESH_BATCH_SIZE: usize = 16;

pub(crate) fn propagate_neighbor_data(
    block_manager: &ClientBlockTypeManager,
    neighbors: &FastChunkNeighbors,
    scratchpad: &mut [u8; 48 * 48 * 48],
) -> Result<()> {
    let neighbor_cache = FastNeighborLockCache::new(neighbors);
    if let Some(current_chunk) = neighbors.get((0, 0, 0)) {
        let mut chunk_data = {
            let _span = span!("lighting");
            // We use a caller-provided scratchpad to avoid reallocating
            scratchpad.fill(0);
            // Search through the entire neighborhood, looking for any light sources
            // i, j, k are offsets from `base`

            let own_view = current_chunk.chunk_data();

            let mut queue = vec![];
            // First, scan through the neighborhood looking for light sources
            for i in -16i32..32 {
                for j in -16i32..32 {
                    for k in -16i32..32 {
                        // Check if we have lighting for this block
                        let light_level =
                            if (0..16).contains(&i) && (0..16).contains(&j) && (0..16).contains(&k)
                            {
                                block_manager.light_emission(own_view.get(ChunkOffset { x: i as u8, y: j as u8, z: k as u8 }))
                            } else {
                                neighbor_cache
                                    .get((i.div_euclid(16), j.div_euclid(16), k.div_euclid(16)))
                                    .map(|x| {
                                        block_manager.light_emission(x.get(ChunkOffset {
                                            x: i.rem_euclid(16) as u8,
                                            y: j.rem_euclid(16) as u8,
                                            z: k.rem_euclid(16) as u8,
                                        }))
                                    })
                                    .unwrap_or(0)
                            };
                        if light_level > 0 {
                            // We have some light. Check if it could possibly reach our own block
                            queue.push((i, j, k, light_level));

                            if (0..16).contains(&i) && (0..16).contains(&j) && (0..16).contains(&k)
                            {
                                // temporary
                                println!("lighting!");
                                for p in -3..=3 {
                                    for q in -3..=3 {
                                        for r in -3..=3 {
                                            scratchpad[(i + p + 16) as usize * 48 * 48
                                                + (j + q + 16) as usize * 48
                                                + (k + r + 16) as usize] = light_level;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Then, 

            // We can't have this write-lock until after we've finished propagating light
            // We do want to keep the lock, so we return it from the scope.
            drop(own_view);
            let mut chunk_data = current_chunk.chunk_data_mut();
            let lightmap = chunk_data.lightmap_mut();
            for i in -1i32..17 {
                for j in -1i32..17 {
                    for k in -1i32..17 {
                        lightmap[(i, j, k).as_extended_index()] =
                            scratchpad[(i + 16) as usize * 48 * 48
                                + (j + 16) as usize * 48
                                + (k + 16) as usize];
                    }
                }
            }

            chunk_data
        };

        {
            let _span = span!("nprop");
            for i in -1i32..17 {
                for j in -1i32..17 {
                    for k in -1i32..17 {
                        if (0..16).contains(&i) && (0..16).contains(&j) && (0..16).contains(&k) {
                            // This isn't a neighbor, and we don't need to fill it in.
                            // Note: This check is important for deadlock safety - if we try to do the lookup, we'll fail to
                            // get the necessary lock because buf already has a write lock.
                            continue;
                        }
                        let neighbor = neighbor_cache
                            .get((i.div_euclid(16), j.div_euclid(16), k.div_euclid(16)))
                            .map(|x| {
                                x.get(ChunkOffset {
                                    x: i.rem_euclid(16) as u8,
                                    y: j.rem_euclid(16) as u8,
                                    z: k.rem_euclid(16) as u8,
                                })
                            })
                            .unwrap_or(block_manager.air_block());
                        chunk_data.block_ids_mut()[(i, j, k).as_extended_index()] = neighbor;
                    }
                }
            }
        }
    }
    Ok(())
}
