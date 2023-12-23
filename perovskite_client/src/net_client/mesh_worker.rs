use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use cgmath::InnerSpace;
use parking_lot::{Condvar, Mutex};
use perovskite_core::coordinates::{ChunkCoordinate, ChunkOffset};
use rustc_hash::FxHashSet;
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::{
    block_renderer::ClientBlockTypeManager,
    game_state::{chunk::ChunkOffsetExt, ClientState, FastChunkNeighbors},
};

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
            client_state,
            queue: Mutex::new(FxHashSet::default()),
            queue_len: AtomicUsize::new(0),
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

    pub(crate) fn run_neighbor_propagator(self: Arc<Self>) -> Result<()> {
        tracy_client::set_thread_name!("async_neighbor_propagator");
        let mut scratchpad = Box::new([0u8; 48 * 48 * 48]);
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
                    let should_mesh = propagate_neighbor_data(
                        &self.client_state.block_types,
                        &self.client_state.chunks.cloned_neighbors_fast(coord),
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

#[test]
pub fn test_rem_euclid() {
    for i in i32::MIN..=i32::MAX {
        assert_eq!(rem_euclid_16_u8(i) as i32, i.rem_euclid(16));
    }
}

#[test]
pub fn test_div_euclid() {
    for i in i32::MIN..=i32::MAX {
        assert_eq!(div_euclid_16_i32(i), i.div_euclid(16));
    }
}

pub(crate) fn propagate_neighbor_data(
    block_manager: &ClientBlockTypeManager,
    neighbors: &FastChunkNeighbors,
    scratchpad: &mut [u8; 48 * 48 * 48],
) -> Result<bool> {
    if let Some(current_chunk) = neighbors.center() {
        let mut current_chunk = current_chunk.chunk_data_mut();
        {
            let _span = span!("chunk precheck");
            // Fast-pass checks
            let air = block_manager.air_block();
            if current_chunk.block_ids().iter().all(|&x| x == air) {
                current_chunk.set_state(crate::game_state::chunk::BlockIdState::NoRender);
                return Ok(true);
            }
        }

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
                            .unwrap_or(block_manager.air_block());
                        current_chunk.block_ids_mut()[(x, y, z).as_extended_index()] = neighbor;
                    }
                }
            }
            if current_chunk
                .block_ids()
                .iter()
                .all(|&x| block_manager.is_solid_opaque(x))
            {
                current_chunk.set_state(crate::game_state::chunk::BlockIdState::NoRender);
                return Ok(true);
            }
        }

        {
            let _span = span!("lighting");

            #[inline]
            fn check_propagation_and_push<F>(
                queue: &mut Vec<(i32, i32, i32, u8)>,
                scratchpad: &mut [u8; 48 * 48 * 48],
                i: i32,
                j: i32,
                k: i32,
                light_level: u8,
                light_propagation: F,
            ) where
                F: Fn(i32, i32, i32) -> bool,
            {
                if i < -16 || j < -16 || k < -16 || i >= 32 || j >= 32 || k >= 32 {
                    return;
                }
                if !light_propagation(i, j, k) {
                    return;
                }
                let old_level = scratchpad
                    [(i + 16) as usize * 48 * 48 + (k + 16) as usize * 48 + (j + 16) as usize];
                // Take the maximum value of the upper and lower nibbles independently
                let max_level = ((old_level & 0xf).max(light_level & 0xf))
                    | (old_level & 0xf0).max(light_level & 0xf0);
                if max_level == old_level {
                    return;
                }

                scratchpad
                    [(i + 16) as usize * 48 * 48 + (k + 16) as usize * 48 + (j + 16) as usize] =
                    max_level;
                let i_dist = (-1 - i).max(i - 16);
                let j_dist = (-1 - j).max(j - 16);
                let k_dist = (-1 - k).max(k - 16);
                let dist = i_dist + j_dist + k_dist;
                let max_level = (light_level >> 4).max(light_level & 0xf);
                if dist < (max_level as i32) {
                    queue.push((i, j, k, light_level));
                }
            }

            scratchpad.fill(0);

            let mut queue = vec![];

            // First, scan through the neighborhood looking for light sources
            // Indices are reversed in order to achieve better cache locality
            // x is the minor index, z is intermediate, and y is the major index

            let mut light_propagation_cache: bitvec::BitArr!(for 48*48*48) =
                bitvec::array::BitArray::ZERO;

            for x_coarse in -1i32..=1 {
                for z_coarse in -1i32..=1 {
                    for y_coarse in -1i32..=1 {
                        let slice = if z_coarse == 0 && y_coarse == 0 && x_coarse == 0 {
                            // The center chunk is not in the slice cache
                            Some(current_chunk.block_ids())
                        } else {
                            neighbors.get((x_coarse, y_coarse, z_coarse))
                        };

                        let global_inbound_lights =
                            neighbors.inbound_light((x_coarse, y_coarse, z_coarse));
                        if let Some(slice) = slice {
                            for x_fine in 0i32..16 {
                                for z_fine in 0i32..16 {
                                    let x = x_coarse * 16 + x_fine;
                                    let z = z_coarse * 16 + z_fine;
                                    let min_index = (x_fine, 0, z_fine).as_extended_index();
                                    let max_index = min_index + 16;
                                    let subslice = &slice[min_index..max_index];
                                    // consider unrolling this loop
                                    let mut global_light =
                                        global_inbound_lights.get(x_fine as u8, z_fine as u8);
                                    for (y_fine, &block_id) in
                                        subslice.iter().enumerate().rev().take(16)
                                    {
                                        let y = y_coarse * 16 + y_fine as i32;
                                        let propagates_light =
                                            block_manager.propagates_light(block_id);
                                        light_propagation_cache.set(
                                            ((x + 16) * 48 * 48 + (z + 16) * 48 + (y + 16))
                                                as usize,
                                            propagates_light,
                                        );
                                        let light_emission = block_manager.light_emission(block_id);
                                        if !propagates_light {
                                            global_light = false;
                                        }
                                        let global_bits = if global_light { 15 << 4 } else { 0 };
                                        let effective_emission = light_emission | global_bits;
                                        if effective_emission > 0 {
                                            check_propagation_and_push(
                                                &mut queue,
                                                scratchpad,
                                                x,
                                                y,
                                                z,
                                                effective_emission,
                                                |_, _, _| true,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let propagates_light_check = |x: i32, y: i32, z: i32| {
                light_propagation_cache
                    [(x + 16) as usize * 48 * 48 + (z + 16) as usize * 48 + (y + 16) as usize]
            };

            // Then, while the queue is non-empty, attempt to propagate light
            while let Some((x, y, z, light_level)) = queue.pop() {
                let decremented = ((light_level & 0xf).saturating_sub(0x1))
                    | ((light_level & 0xf0).saturating_sub(0x10));
                check_propagation_and_push(
                    &mut queue,
                    scratchpad,
                    x - 1,
                    y,
                    z,
                    decremented,
                    propagates_light_check,
                );
                check_propagation_and_push(
                    &mut queue,
                    scratchpad,
                    x + 1,
                    y,
                    z,
                    decremented,
                    propagates_light_check,
                );
                check_propagation_and_push(
                    &mut queue,
                    scratchpad,
                    x,
                    y - 1,
                    z,
                    decremented,
                    propagates_light_check,
                );
                check_propagation_and_push(
                    &mut queue,
                    scratchpad,
                    x,
                    y + 1,
                    z,
                    decremented,
                    propagates_light_check,
                );
                check_propagation_and_push(
                    &mut queue,
                    scratchpad,
                    x,
                    y,
                    z - 1,
                    decremented,
                    propagates_light_check,
                );
                check_propagation_and_push(
                    &mut queue,
                    scratchpad,
                    x,
                    y,
                    z + 1,
                    decremented,
                    propagates_light_check,
                );
            }

            let lightmap = current_chunk.lightmap_mut();
            for x in -1i32..17 {
                for z in -1i32..17 {
                    for y in -1i32..17 {
                        lightmap[(x, y, z).as_extended_index()] =
                            scratchpad[(x + 16) as usize * 48 * 48
                                + (z + 16) as usize * 48
                                + (y + 16) as usize];
                    }
                }
            }
        }

        current_chunk.set_state(crate::game_state::chunk::BlockIdState::ReadyToRender);

        Ok(true)
    } else {
        Ok(false)
    }
}
