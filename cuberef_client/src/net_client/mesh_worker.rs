use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use cgmath::InnerSpace;
use cuberef_core::coordinates::{ChunkCoordinate, ChunkOffset};
use parking_lot::{Condvar, Mutex, MutexGuard};
use rustc_hash::FxHashSet;
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::{
    block_renderer::ClientBlockTypeManager,
    game_state::{
        chunk::ChunkOffsetExt, ClientState, FastChunkNeighbors, FastNeighborLockCache,
        FastNeighborSliceCache,
    },
};

// Responsible for reconciling a chunk with data from other nearby chunks (e.g. lighting, neighbor calculations)
pub(crate) struct NeighborPropagator {
    client_state: Arc<ClientState>,
    queue: Mutex<FxHashSet<ChunkCoordinate>>,
    queue_len: AtomicUsize,
    token: Mutex<()>,
    cond: Condvar,
    shutdown: CancellationToken,
    mesh_workers: Vec<Arc<MeshWorker>>,
}
impl NeighborPropagator {
    pub(crate) fn new(
        client_state: Arc<ClientState>,
        mesh_workers: Vec<Arc<MeshWorker>>,
    ) -> (Arc<Self>, tokio::task::JoinHandle<Result<()>>) {
        let worker = Arc::new(Self {
            client_state,
            queue: Mutex::new(FxHashSet::default()),
            queue_len: AtomicUsize::new(0),
            token: Mutex::new(()),
            cond: Condvar::new(),
            shutdown: CancellationToken::new(),
            mesh_workers,
        });
        let handle = {
            let worker_clone = worker.clone();
            tokio::task::spawn_blocking(move || worker_clone.run_neighbor_propagator())
        };
        (worker, handle)
    }

    /// Estimates how many tasks are left in the queue. This is a rough estimate suitable
    /// for flow control, but may be stale or approximate.
    pub(crate) fn queue_len(&self) -> usize {
        self.queue_len.load(Ordering::Relaxed)
    }

    pub(crate) fn enqueue(&self, coord: ChunkCoordinate) {
        self.queue.lock().insert(coord);
        self.cond.notify_one();
    }

    pub(crate) fn cancel(&self) {
        self.shutdown.cancel();
        self.cond.notify_one();
    }

    /// Pauses this neighbor propagator and borrows its token. Once the token is dropped,
    /// this neighbor propagator will resume.
    /// This can be used to preempt the neighbor propagator thread's work to allow a higher-priority
    /// neighbor propagation to run inline in a message handler.
    pub(crate) fn borrow_token<'a>(&'a self) -> NeighborPropagationToken<'a> {
        let _span = span!("mesh_worker borrow_token");
        NeighborPropagationToken(self.token.lock())
    }

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
                self.queue_len.store(chunks.len(), Ordering::Relaxed);

                // Prioritize the closest chunks
                chunks.sort_by_key(|x| {
                    let center = x.with_offset(ChunkOffset { x: 8, y: 8, z: 8 });
                    let offset =
                        cgmath::vec3(center.x as f64, center.y as f64, center.z as f64) - pos;
                    offset.magnitude2() as u64
                });

                if chunks.len() > MESH_BATCH_SIZE {
                    chunks.resize_with(MESH_BATCH_SIZE, || unreachable!());
                }

                for coord in chunks.iter() {
                    assert!(lock.remove(coord), "Task should have been in the queue");
                }
                drop(lock);
                chunks
            };
            {
                let _span = span!("nprop_work");
                let mut token = NeighborPropagationToken(self.token.lock());
                for coord in chunks {
                    let should_enqueue = propagate_neighbor_data(
                        &self.client_state.block_types,
                        &self.client_state.chunks.cloned_neighbors_fast(coord),
                        &mut scratchpad,
                        &mut token,
                    )?;
                    if should_enqueue {
                        let index = coord.hash_u64() % (self.mesh_workers.len() as u64);
                        self.mesh_workers[index as usize].enqueue(coord);
                    }
                }
            }
        }
        Ok(())
    }
}

/// An opaque token allowing the neighbor propagation algorithm to run.
/// It's used to ensure that two threads don't attempt neighbor propagation at the same time,
/// which could lead to deadlocks.
pub(crate) struct NeighborPropagationToken<'a>(MutexGuard<'a, ()>);

// Responsible for turning a single chunk into a mesh
pub(crate) struct MeshWorker {
    client_state: Arc<ClientState>,
    queue: Mutex<FxHashSet<ChunkCoordinate>>,
    queue_len: AtomicUsize,
    cond: Condvar,
    shutdown: CancellationToken,
    // There is no run token, because these workers can be run in parallel
}
impl MeshWorker {
    pub(crate) fn new(
        client_state: Arc<ClientState>,
    ) -> (Arc<Self>, tokio::task::JoinHandle<Result<()>>) {
        let worker = Arc::new(Self {
            client_state,
            queue: Mutex::new(FxHashSet::default()),
            queue_len: AtomicUsize::new(0),
            cond: Condvar::new(),
            shutdown: CancellationToken::new(),
        });
        let handle = {
            let worker_clone = worker.clone();
            tokio::task::spawn_blocking(move || worker_clone.run_mesh_worker())
        };
        (worker, handle)
    }

    pub(crate) fn enqueue(&self, coord: ChunkCoordinate) {
        self.queue.lock().insert(coord);
        self.cond.notify_one();
    }

    pub(crate) fn queue_len(&self) -> usize {
        self.queue_len.load(Ordering::Relaxed)
    }

    pub(crate) fn cancel(&self) {
        self.shutdown.cancel();
        self.cond.notify_one();
    }

    pub(crate) fn run_mesh_worker(self: Arc<Self>) -> Result<()> {
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
                // This doesn't have any hard atomicity guarantees
                self.queue_len.store(chunks.len(), Ordering::Relaxed);
                plot!("mesh_queue_length", chunks.len() as f64);

                // Prioritize the closest chunks
                chunks.sort_by_key(|x| {
                    let center = x.with_offset(ChunkOffset { x: 8, y: 8, z: 8 });
                    let offset =
                        cgmath::vec3(center.x as f64, center.y as f64, center.z as f64) - pos;
                    offset.magnitude2() as u64
                });
                if chunks.len() > NPROP_BATCH_SIZE {
                    chunks.resize_with(NPROP_BATCH_SIZE, || unreachable!());
                }

                for coord in chunks.iter() {
                    assert!(lock.remove(coord), "Task should have been in the queue");
                }
                drop(lock);
                chunks
            };
            {
                let _span = span!("mesh_worker work");

                plot!("mesh_queue work size", chunks.len() as f64);
                for coord in chunks {
                    let neighbors = &self.client_state.chunks.cloned_neighbors_fast(coord);
                    if let Some(chunk) = neighbors.get((0, 0, 0)) {
                        chunk.mesh_with(&self.client_state.block_renderer)?;
                    }
                }
            }
        }
        Ok(())
    }
}
const MESH_BATCH_SIZE: usize = 32;
const NPROP_BATCH_SIZE: usize = 128;

#[inline]
fn rem_euclid_16_u8(i: i32) -> u8 {
    // Even with a constant value of 16, rem_euclid generates pretty bad assembly on x86_64: https://godbolt.org/z/T8zjsYezs
    (i & 0xf) as u8
}
#[inline]
fn rem_euclid_16_i32(i: i32) -> i32 {
    // Even with a constant value of 16, rem_euclid generates pretty bad assembly on x86_64: https://godbolt.org/z/T8zjsYezs
    i & 0xf
}

#[test]
pub fn test_rem_euclid() {
    for i in i32::MIN..=i32::MAX {
        assert_eq!(rem_euclid_16_u8(i) as i32, i.rem_euclid(16));
    }
}

pub(crate) fn propagate_neighbor_data(
    block_manager: &ClientBlockTypeManager,
    neighbors: &FastChunkNeighbors,
    scratchpad: &mut [u8; 48 * 48 * 48],
    _token: &mut NeighborPropagationToken,
) -> Result<bool> {
    let neighbor_cache = FastNeighborLockCache::new(&neighbors);
    let slice_cache = FastNeighborSliceCache::new(&neighbor_cache);

    if let Some(current_chunk) = neighbors.get((0, 0, 0)) {
        let mut current_chunk = current_chunk.chunk_data_mut();
        {
            let _span = span!("chunk precheck");
            // Fast-pass checks
            let air = block_manager.air_block();
            if current_chunk.block_ids().iter().all(|&x| x == air) {
                current_chunk.set_state(
                    crate::game_state::chunk::BlockIdState::BlockIdsWithNeighborsAuditNoRender,
                );
                return Ok(true);
            }
        }

        {
            let _span = span!("nprop");
            for i in -1i32..17 {
                for j in -1i32..17 {
                    for k in -1i32..17 {
                        if (0..16).contains(&i) && (0..16).contains(&j) && (0..16).contains(&k) {
                            // This isn't a neighbor, and we don't need to fill it in (it's already present)
                            // Note: This check is important for deadlock safety - if we try to do the lookup, we'll fail to
                            // get the read lock because buf already has a write lock.
                            continue;
                        }
                        let neighbor = slice_cache
                            .get((i.div_euclid(16), j.div_euclid(16), k.div_euclid(16)))
                            .map(|x| {
                                x[ChunkOffset {
                                    x: rem_euclid_16_u8(i),
                                    y: rem_euclid_16_u8(j),
                                    z: rem_euclid_16_u8(k),
                                }
                                .as_extended_index()]
                            })
                            .unwrap_or(block_manager.air_block());

                        current_chunk.block_ids_mut()[(i, j, k).as_extended_index()] = neighbor;
                    }
                }
            }
            if current_chunk
                .block_ids()
                .iter()
                .all(|&x| block_manager.is_solid_opaque(x))
            {
                current_chunk.set_state(
                    crate::game_state::chunk::BlockIdState::BlockIdsWithNeighborsAuditNoRender,
                );
                return Ok(true);
            }
        }

        {
            let _span = span!("lighting");
            // We use a caller-provided scratchpad to avoid reallocating
            // Search through the entire neighborhood, looking for any light sources
            // i, j, k are offsets from `base`
            cuberef_core::lighting::do_lighting_pass(
                scratchpad,
                |block| block_manager.light_emission(block),
                |block| block_manager.propagates_light(block),
                |i, j, k| {
                    // i needs special handling - it's -1..=1, rather than -16..32
                    if i == 0 && (0..16).contains(&j) && (0..16).contains(&k) {
                        let start = (0, j, k).as_extended_index();
                        let finish = start + 16;
                        current_chunk.block_ids()[start..finish].try_into().unwrap()
                    } else {
                        slice_cache
                            .get((i, j.div_euclid(16), k.div_euclid(16)))
                            .map(|x| {
                                let start = (0, rem_euclid_16_i32(j), rem_euclid_16_i32(k))
                                    .as_extended_index();
                                let finish = start + 16;
                                x[start..finish].try_into().unwrap()
                            })
                            .unwrap_or([block_manager.air_block(); 16])
                    }
                },
                |i, j, k| {
                    if (0..16).contains(&i) && (0..16).contains(&j) && (0..16).contains(&k) {
                        current_chunk.get_block(ChunkOffset {
                            x: i as u8,
                            y: j as u8,
                            z: k as u8,
                        })
                    } else {
                        slice_cache
                            .get((i.div_euclid(16), j.div_euclid(16), k.div_euclid(16)))
                            .map(|x| {
                                x[ChunkOffset {
                                    x: rem_euclid_16_u8(i),
                                    y: rem_euclid_16_u8(j),
                                    z: rem_euclid_16_u8(k),
                                }
                                .as_extended_index()]
                            })
                            .unwrap_or(block_manager.air_block())
                    }
                },
            );

            let lightmap = current_chunk.lightmap_mut();
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
        }

        current_chunk.set_state(crate::game_state::chunk::BlockIdState::BlockIdsReadyToRender);

        Ok(true)
    } else {
        Ok(false)
    }
}
