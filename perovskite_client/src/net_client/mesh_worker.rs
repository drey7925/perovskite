use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use crate::client_state::{
    block_types::ClientBlockTypeManager, chunk::ChunkOffsetExt, ClientState, FastChunkNeighbors,
};
use anyhow::{Context, Result};
use cgmath::InnerSpace;
use parking_lot::{Condvar, Mutex};
use perovskite_core::block_id::BlockId;
use perovskite_core::lighting::ChunkBuffer;
pub(crate) use perovskite_core::lighting::{
    propagate_light, LightScratchpad, Lightfield, NeighborBuffer,
};
use perovskite_core::{
    block_id::special_block_defs::UNLOADED_CHUNK_BLOCK_ID,
    coordinates::{ChunkCoordinate, ChunkOffset},
};
use rustc_hash::FxHashSet;
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

// Responsible for reconciling a chunk with data from other nearby chunks (e.g. lighting, neighbor calculations)
pub(crate) struct NeighborPropagator {
    client_state: Arc<ClientState>,
    queue: Mutex<FxHashSet<ChunkCoordinate>>,
    queue_len: AtomicUsize,
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
            shutdown: client_state.shutdown.clone(),
            client_state,
            queue: Mutex::new(FxHashSet::default()),
            queue_len: AtomicUsize::new(0),
            cond: Condvar::new(),
            mesh_workers,
        });
        let handle = {
            let worker_clone = worker.clone();
            tokio::task::spawn_blocking(move || worker_clone.run_neighbor_propagator())
        };
        (worker, handle)
    }

    /// Estimates how many tasks are left in the scratchpad.visit_queue. This is a rough estimate suitable
    /// for flow control, but may be stale or approximate.
    pub(crate) fn queue_len(&self) -> usize {
        self.queue_len.load(Ordering::Relaxed)
    }

    pub(crate) fn enqueue(&self, coord: ChunkCoordinate) {
        let mut guard = self.queue.lock();
        guard.insert(coord);
        self.cond.notify_one();
        drop(guard);
    }

    pub(crate) fn cancel(&self) {
        self.shutdown.cancel();
        self.cond.notify_one();
    }

    pub(crate) fn run_neighbor_propagator(self: Arc<Self>) -> Result<()> {
        tracy_client::set_thread_name!("async_neighbor_propagator");
        let mut scratchpad = LightScratchpad::default();
        let mut chunk_neighbor_scratchpad = FastChunkNeighbors::default();
        if self
            .client_state
            .settings
            .load()
            .render
            .testonly_noop_meshing
        {
            while !self.shutdown.is_cancelled() {
                let mut lock = self.queue.lock();
                lock.clear();
                self.cond.wait_for(&mut lock, Duration::from_secs(1));
            }
        }
        while !self.shutdown.is_cancelled() {
            // This is duplicated in MeshWorker to allow the two to use different strategies, and also
            // report different span and plot names.
            // This is quite ugly and deadlock-prone. Figure out whether we need this or not.
            // The big deadlock risk is if this line happens under the queue lock:
            //   our thread waits for physics_state to unlock
            //   physics_state waits for mapchunks to unlock
            //   network thread is holding chunks and waiting for queue to unlock
            //
            // At the moment, the deadlocks are removed, but this still seems brittle.
            let pos = self.client_state.weakly_ordered_last_position().position;
            let mut lock = self.queue.lock();
            if lock.is_empty() {
                plot!("nprop_queue_length", 0.);
                self.cond.wait_for(&mut lock, Duration::from_secs(1));
            }

            let sort_span = span!("nprop sort");
            let mut chunks: Vec<_> = lock.iter().copied().collect();

            plot!("nprop_queue_length", chunks.len() as f64);
            self.queue_len.store(chunks.len(), Ordering::Relaxed);

            // Prioritize the closest chunks
            let chunks = if chunks.len() > MESH_BATCH_SIZE {
                let (before, _, _) = chunks.select_nth_unstable_by_key(MESH_BATCH_SIZE, |x| {
                    let center = x.with_offset(ChunkOffset { x: 8, y: 8, z: 8 });
                    let offset =
                        cgmath::vec3(center.x as f64, center.y as f64, center.z as f64) - pos;
                    offset.magnitude2() as u64
                });
                &*before
            } else {
                &chunks
            };

            for coord in chunks.iter() {
                assert!(lock.remove(coord), "Task should have been in the queue");
            }
            drop(lock);
            drop(sort_span);

            {
                let _span = span!("nprop_work");

                for &coord in chunks {
                    self.client_state.chunks.cloned_neighbors_fast(
                        coord,
                        &self.client_state.block_types,
                        &mut chunk_neighbor_scratchpad,
                    );

                    let should_mesh = propagate_neighbor_data(
                        &self.client_state.block_types,
                        &chunk_neighbor_scratchpad,
                        &mut scratchpad,
                    )?;
                    if should_mesh {
                        let index = coord.hash_u64() % (self.mesh_workers.len() as u64);
                        self.mesh_workers[index as usize].enqueue(coord);
                    }
                }
            }
        }
        Ok(())
    }
}

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
            shutdown: client_state.shutdown.clone(),
            client_state,
            queue: Mutex::new(FxHashSet::default()),
            queue_len: AtomicUsize::new(0),
            cond: Condvar::new(),
        });
        let handle = {
            let worker_clone = worker.clone();
            tokio::task::spawn_blocking(move || worker_clone.run_mesh_worker())
        };
        (worker, handle)
    }

    pub(crate) fn enqueue(&self, coord: ChunkCoordinate) {
        let mut guard = self.queue.lock();
        guard.insert(coord);
        self.cond.notify_one();
        drop(guard);
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
            let chunks = if self
                .client_state
                .settings
                .load()
                .render
                .testonly_noop_meshing
            {
                self.queue.lock().clear();
                self.cond
                    .wait_for(&mut self.queue.lock(), Duration::from_secs(1));
                vec![]
            } else {
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
                    self.client_state
                        .chunks
                        .maybe_mesh_and_maybe_promote(coord, &self.client_state.block_renderer)?;
                }
            }
        }
        Ok(())
    }
}

// Responsible for turning a single chunk into a mesh
pub(crate) struct MeshBatcher {
    client_state: Arc<ClientState>,
    shutdown: CancellationToken,
    // There is no run token, because these workers can be run in parallel
}
impl MeshBatcher {
    pub(crate) fn run_batcher(self: Arc<Self>) -> Result<()> {
        tracy_client::set_thread_name!("async_mesh_batcher");
        while !self.shutdown.is_cancelled() {
            // Sleep for a bit to allow more work to accumulate
            std::thread::sleep(Duration::from_millis(50));
            let pos = self.client_state.weakly_ordered_last_position().position;
            self.client_state
                .chunks
                .do_batch_round(pos, self.client_state.block_renderer.vk_ctx())?;
        }

        Ok(())
    }

    pub(crate) fn new(
        client_state: Arc<ClientState>,
    ) -> (Arc<Self>, tokio::task::JoinHandle<Result<()>>) {
        let worker = Arc::new(Self {
            shutdown: client_state.shutdown.clone(),
            client_state,
        });
        let handle = {
            let worker_clone = worker.clone();
            tokio::task::spawn_blocking(move || worker_clone.run_batcher())
        };
        (worker, handle)
    }

    pub(crate) fn cancel(&self) {
        self.shutdown.cancel();
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
fn div_euclid_16_i32(i: i32) -> i32 {
    // Even with a constant value of 16, rem_euclid generates pretty bad assembly on x86_64: https://godbolt.org/z/T8zjsYezs
    i >> 4
}

// Too slow in debug mode
#[cfg(not(debug_assertions))]
#[test]
pub fn test_rem_euclid() {
    for i in i32::MIN..=i32::MAX {
        assert_eq!(rem_euclid_16_u8(i) as i32, i.rem_euclid(16));
    }
}

// Too slow in debug mode
#[cfg(not(debug_assertions))]
#[test]
pub fn test_div_euclid() {
    for i in i32::MIN..=i32::MAX {
        assert_eq!(div_euclid_16_i32(i), i.div_euclid(16));
    }
}

#[derive(Clone, Copy)]
struct ChunkBuffer18<'a>(&'a [BlockId; 18 * 18 * 18]);

impl ChunkBuffer for ChunkBuffer18<'_> {
    fn get(&self, offset: ChunkOffset) -> BlockId {
        self.0[offset.as_extended_index()]
    }

    fn vertical_slice(&self, x: u8, z: u8) -> &[BlockId; 16] {
        let min_offset = ChunkOffset::new(x, 0, z).as_extended_index();
        let max_offset = min_offset + 16;
        // https://github.com/rust-lang/rust/issues/90091 would be nice once stabilized
        self.0[min_offset..max_offset].try_into().unwrap()
    }
}

struct FcnWithCenter<'a> {
    neighbors: &'a FastChunkNeighbors,
    center: ChunkBuffer18<'a>,
}
impl<'a> NeighborBuffer for FcnWithCenter<'a> {
    type Chunk<'b>
        = ChunkBuffer18<'b>
    where
        Self: 'b;

    fn get(&self, dx: i32, dy: i32, dz: i32) -> Option<Self::Chunk<'a>> {
        if dx == 0 && dy == 0 && dz == 0 {
            Some(self.center)
        } else {
            self.neighbors.get((dx, dy, dz)).map(ChunkBuffer18)
        }
    }

    fn inbound_light(&self, dx: i32, dy: i32, dz: i32) -> Lightfield {
        self.neighbors.inbound_light((dx, dy, dz))
    }
}

pub(crate) fn propagate_neighbor_data(
    block_manager: &ClientBlockTypeManager,
    neighbors: &FastChunkNeighbors,
    scratchpad: &mut LightScratchpad,
) -> Result<bool> {
    if !neighbors.should_mesh() {
        return Ok(false);
    }
    if let Some(current_chunk) = neighbors.center() {
        let mut current_chunk = current_chunk.chunk_data_mut();
        {
            let _span = span!("chunk precheck");
            // Fast-pass checks
            if current_chunk.is_empty_optimization_hint() {
                current_chunk.set_state(crate::client_state::chunk::ChunkRenderState::NoRender);
                return Ok(true);
            }
        }

        let center_ids_mut = current_chunk
            .block_ids_mut()
            .context("Mutable block IDs should be non-empty because the chunk is not all air")?;

        {
            let _span = span!("nprop");
            for x in -1i32..17 {
                for z in -1i32..17 {
                    for y in -1i32..17 {
                        if (0..16).contains(&x) && (0..16).contains(&y) && (0..16).contains(&z) {
                            // This isn't a neighbor, and we don't need to fill it in (it's already present)
                            // Note: This check is important for deadlock safety - if we try to do the lookup, we'll fail to
                            // get the read lock because buf already has a write lock. It is not merely an optimization
                            continue;
                        }
                        let neighbor = neighbors
                            .get((
                                div_euclid_16_i32(x),
                                div_euclid_16_i32(y),
                                div_euclid_16_i32(z),
                            ))
                            .map(|block_ids| {
                                block_ids[ChunkOffset {
                                    x: rem_euclid_16_u8(x),
                                    y: rem_euclid_16_u8(y),
                                    z: rem_euclid_16_u8(z),
                                }
                                .as_extended_index()]
                            })
                            .unwrap_or(UNLOADED_CHUNK_BLOCK_ID);
                        center_ids_mut[(x, y, z).as_extended_index()] = neighbor;
                    }
                }
            }
            if center_ids_mut
                .iter()
                .all(|&x| block_manager.is_solid_opaque(x))
            {
                current_chunk.set_state(crate::client_state::chunk::ChunkRenderState::NoRender);
                return Ok(true);
            }
        }

        {
            let _span = span!("lighting");
            let fcn_with_center = FcnWithCenter {
                neighbors,
                center: ChunkBuffer18(&*center_ids_mut),
            };

            propagate_light(
                fcn_with_center,
                scratchpad,
                |id| block_manager.propagates_light(id),
                |id| block_manager.light_emission(id),
            );

            let lightmap = current_chunk.lightmap_mut();
            for x in -1i32..17 {
                for z in -1i32..17 {
                    for y in -1i32..17 {
                        lightmap[(x, y, z).as_extended_index()] =
                            scratchpad.get_packed_u4_u4(x, y, z);
                    }
                }
            }
        }

        current_chunk.set_state(crate::client_state::chunk::ChunkRenderState::ReadyToRender);

        Ok(true)
    } else {
        Ok(false)
    }
}
