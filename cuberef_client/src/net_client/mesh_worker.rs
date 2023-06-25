use std::{sync::Arc, collections::HashSet, time::Duration};

use anyhow::Result;
use cgmath::InnerSpace;
use cuberef_core::coordinates::{ChunkCoordinate, ChunkOffset};
use parking_lot::{Mutex, Condvar};
use tokio_util::sync::CancellationToken;
use tracy_client::{plot, span};

use crate::game_state::{ClientState, chunk::maybe_mesh_chunk};

pub(crate) struct MeshWorker {
    pub(crate) client_state: Arc<ClientState>,
    pub(crate) queue: Mutex<HashSet<ChunkCoordinate>>,
    pub(crate) cond: Condvar,
    pub(crate) shutdown: CancellationToken,
}
impl MeshWorker {
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

                plot!("mesh_queue_length", chunks.len() as f64);

                // Prioritize the closest chunks
                chunks.sort_by_key(|x| {
                    let center = x.with_offset(ChunkOffset { x: 8, y: 8, z: 8 });
                    let distance =
                        cgmath::vec3(center.x as f64, center.y as f64, center.z as f64) - pos;
                    distance.magnitude2() as i64
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
                    maybe_mesh_chunk(
                        coord,
                        &self.client_state.chunks.read_lock(),
                        &self.client_state.cube_renderer,
                    )?;
                }
            }
        }
        Ok(())
    }
}
const MESH_BATCH_SIZE: usize = 16;
