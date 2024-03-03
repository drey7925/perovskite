/// This implementation is EXTREMELY unstable while it is under active development.
/// Do not assume any functionality or performance guarantees while different techniques
/// are under investigation.
use std::{
    fmt::Debug,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use cgmath::{vec3, Vector3, Zero};
use parking_lot::{Mutex, RwLock};
use perovskite_core::{
    block_id::BlockId, coordinates::BlockCoordinate, protocol::entities::EntityMove,
};
use rustc_hash::FxHashMap;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracy_client::span;

use crate::{
    database::database_engine::GameDatabase,
    game_state::blocks::{BlockTypeHandle, ExtendedData},
    CachelineAligned,
};

use super::{GameState, ServerGameMap};

/// This entity is present and valid
const CONTROL_PRESENT: u8 = 1;
/// This entity should remain loaded even when it's in unloaded chunks
const CONTROL_STICKY: u8 = 2;

/// This entity has a coroutine controlling it.
const CONTROL_AUTONOMOUS: u8 = 4;
/// This entity should follow simple physics without needing a coroutine to drive it.
const CONTROL_SIMPLE_PHYSICS: u8 = 8;

/// This entity has no upcoming move and needs a move to be calculated
const CONTROL_AUTONOMOUS_NEEDS_CALC: u8 = 16;

/// This entity has a coroutine, but it's suspended while waiting for a completion from tokio
const CONTROL_SUSPENDED: u8 = 32;

struct Completion<T: Send + Sync + 'static> {
    /// The entity ID we'll deliver the result to
    entity_id: u64,
    /// The index in the entity array. If we don't find the given ID there, we'll drop the completion.
    index: usize,
    /// The result we're delivering.
    result: T,
}

// A deferred call to be invoked on a tokio executor
struct LockDeferredCall<T: Send + Sync + 'static> {
    // TODO future scaffolding, not yet implemented
    //
    // Design notes: This represents the future that we want to invoke on a Tokio
    // executor.
    //
    // An EntityCoroutineServices method requiring locks would *try* to get the lock it wants.
    // If it can't get the lock immediately, it will return a _ScheduleMapLockRequest
    // with a LockDeferredCall that the coroutine should return as-is. The deferred call should
    // do the intended task in a blocking manner.
    //
    // Once this happens, the entity loop code should queue up a tokio task that will
    // 1. call the deferred call, obtaining a result
    // 2. prepare a completion, and send it over the mpsc sender (TODO decide whether a sender
    // or a permit, as a matter of flow control)
    //     The decision for permit vs sender is non-trivial. If it's a permit, then we get to
    //     do flow control here in the entity system (but we may need to rerun the coroutine since
    //     we'll have nowhere to put the deferred call in cases of backpressure)
    //     If it's a sender, then backlogs will lead to tasks building up in tokio. We may need to
    //     keep an eye on the number of outstanding tokio tasks and avoid invoking coroutines when
    //     facing such a backlog. This can probably be itself deferred for a future version.
    deferred_call: Box<dyn FnOnce() -> T>,
    sender: Option<tokio::sync::mpsc::Sender<Completion<T>>>,
}

fn id_to_shard(id: u64) -> usize {
    // Put 256 consecutive entities in the same shard, rather than round-robining directly
    (id as usize / 256) % NUM_ENTITY_SHARDS
}

// todo scale this
pub(crate) const NUM_ENTITY_SHARDS: usize = 1;
pub struct EntityManager {
    shards: [CachelineAligned<EntityShard>; NUM_ENTITY_SHARDS],
    workers: Mutex<Vec<tokio::task::JoinHandle<Result<()>>>>,
    cancellation: CancellationToken,
    next_id: AtomicU64,
}
impl EntityManager {
    pub(crate) fn new(db: Arc<dyn GameDatabase>) -> EntityManager {
        EntityManager {
            shards: std::array::from_fn(|shard_id| {
                CachelineAligned(EntityShard::new(shard_id, db.clone()))
            }),
            workers: Mutex::new(Vec::new()),
            cancellation: CancellationToken::new(),
            // TODO next_id needs to be initialized from the database eventually
            next_id: AtomicU64::new(0),
        }
    }

    pub(crate) fn start_workers(self: Arc<Self>, game_state: Arc<GameState>) {
        let workers = &mut self.workers.lock();
        assert!(workers.is_empty());
        for shard_id in 0..NUM_ENTITY_SHARDS {
            let cancellation = self.cancellation.clone();
            let worker =
                EntityShardWorker::new(self.clone(), shard_id, game_state.clone(), cancellation);
            let handle = tokio::spawn(async move { worker.run_loop().await });
            workers.push(handle);
        }
    }
    pub(crate) fn request_shutdown(&self) {
        self.cancellation.cancel();
    }
    pub(crate) async fn await_shutdown(&self) -> Result<()> {
        for handle in self.workers.lock().drain(..) {
            handle.await??;
        }
        Ok(())
    }

    pub fn get_shard(&self, id: u64) -> &EntityShard {
        &self.shards[id_to_shard(id)]
    }
    fn assign_next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
    /// Creates a new entity. API is subject to change (this is a placeholder)
    ///
    /// Note that this runs asynchronously, so the entity may not appear until
    /// a bit later
    pub async fn new_entity(
        &self,
        position: Vector3<f64>,
        coroutine: Option<Pin<Box<dyn EntityCoroutine>>>,
    ) -> u64 {
        let id = self.assign_next_id();
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .send(EntityAction::Insert(id, position, coroutine))
            .await
            .context("Entity receiver disappeared")
            .unwrap();
        id
    }
    pub fn new_entity_blocking(
        &self,
        position: Vector3<f64>,
        coroutine: Option<Pin<Box<dyn EntityCoroutine>>>,
    ) -> u64 {
        let id = self.assign_next_id();
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .blocking_send(EntityAction::Insert(id, position, coroutine))
            .context("Entity receiver disappeared")
            .unwrap();
        id
    }

    pub(crate) fn shards(&self) -> &[CachelineAligned<EntityShard>] {
        &self.shards
    }

    pub async fn remove(&self, entity_id: u64) {
        let shard = self.get_shard(entity_id);
        shard
            .pending_actions_tx
            .send(EntityAction::Remove(entity_id))
            .await
            .context("Entity receiver disappeared")
            .unwrap();
    }
    pub async fn remove_blocking(&self, entity_id: u64) {
        let shard = self.get_shard(entity_id);
        shard
            .pending_actions_tx
            .blocking_send(EntityAction::Remove(entity_id))
            .context("Entity receiver disappeared")
            .unwrap();
    }

    pub async fn set_kinematics(
        &self,
        id: u64,
        position: Vector3<f64>,
        current: Movement,
        next: Option<Movement>,
    ) {
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .send(EntityAction::SetKinematics(id, position, current, next))
            .await
            .context("Entity receiver disappeared")
            .unwrap();
    }

    pub async fn set_kinematics_blocking(
        &self,
        id: u64,
        position: Vector3<f64>,
        current: Movement,
        next: Option<Movement>,
    ) {
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .blocking_send(EntityAction::SetKinematics(id, position, current, next))
            .context("Entity receiver disappeared")
            .unwrap();
    }
}

// Note that if an entity has neither CONTROL_AUTONOMOUS nor CONTROL_SIMPLE_PHYSICS, it will
// only move when programatically requested to do so by a caller
#[non_exhaustive]
pub enum EntityMoveDecision {
    /// Once the current movement ends, start the specified movement.
    /// This will give the smoothest animation.
    QueueUpMovement(Movement),
    /// Get rid of the current movement, set the position, start the new movement immediately, and queue up the following movement (if it's Some).
    /// Note that this may lead to visual glitches due to unsmoothed movement and network latency.
    ResetMovement(Vector3<f64>, Movement, Option<Movement>),
    /// Despawn the entity immediately
    ImmediateDespawn,
    /// Ask again later, in this many seconds
    AskAgainLater(f32),
    /// Stop moving, and stop calling the coroutine asking for more moves, until
    /// manually re-enabled from non-coroutine code
    StopCoroutineControl,
}

pub enum CoroutineResult {
    /// The coroutine returned successfully
    Successful(EntityMoveDecision),

    #[allow(private_interfaces)]
    #[doc(hidden)]
    /// Schedule a future call to the coroutine at a later point in time when a map
    /// chunk can be locked. This cannot be constructed by hand.
    ///
    /// At the moment, this schedules onto a tokio task pool as a blocking task.
    _Deferred(LockDeferredCall<EntityMoveDecision>),
}
impl From<EntityMoveDecision> for CoroutineResult {
    fn from(m: EntityMoveDecision) -> Self {
        Self::Successful(m)
    }
}

/// A single step in the path plan for an entity.
/// At the moment, the entity core array can only hold a single step at a time.
pub struct Movement {
    pub velocity: Vector3<f32>,
    pub acceleration: Vector3<f32>,
    pub face_direction: f32,
    pub move_time: f32,
}
impl Movement {
    /// Returns a movement that will stop and stay at the current position forever.
    /// This movement takes a very long time, so the entity will not move, and will
    /// not have its control coroutine called again.
    pub fn stop_and_stay(face_direction: f32) -> Self {
        Self {
            velocity: Vector3::zero(),
            acceleration: Vector3::zero(),
            face_direction,
            move_time: f32::MAX,
        }
    }
    pub fn pos_after(&self, start: Vector3<f64>, time: f32) -> Vector3<f64> {
        Vector3::new(
            qproj(start.x, self.velocity.x, self.acceleration.x, time),
            qproj(start.y, self.velocity.y, self.acceleration.y, time),
            qproj(start.z, self.velocity.z, self.acceleration.z, time),
        )
    }
}

/// Services available to the entity coroutine.
///
/// This is the ONLY way that coroutines should interact with the game state.
///
/// It's technically possible for a coroutine to sneak an Arc<GameState> or similar into its own
/// state and try to use it - however **that will likely lead to deadlocks**.
pub struct EntityCoroutineServices<'a> {
    map: &'a ServerGameMap,
}
impl<'a> EntityCoroutineServices<'a> {
    /// Gets the block at the specified coordinate, or None if the chunk isn't loaded.
    /// Forwards to the game map's try_get_block.
    pub fn try_get_block(&self, coord: BlockCoordinate) -> Option<BlockId> {
        self.map.try_get_block(coord)
    }
}

/// Warning: This trait is not stable and may change in the future. If `coroutine_trait` is stabilized,
/// it may replace this trait. Regardless, it can still evolve quickly.
pub trait EntityCoroutine: Send + Sync + 'static {
    /// Called when the entity's next move needs to be planned.
    ///
    /// Args:
    ///   * current_position: The current position of the entity
    ///   * whence: The position of the entity at the time when the current move finishes. The returned move will start from approximately that place.
    ///   * when: The time (seconds from now) when the current move finishes. The returned move will start at that time.
    ///
    /// Returns:
    ///     The next move for the entity
    fn plan_move(
        self: Pin<&mut Self>,
        services: &EntityCoroutineServices<'_>,
        current_position: Vector3<f64>,
        whence: Vector3<f64>,
        when: f32,
    ) -> CoroutineResult;
}

const ID_MASK: u64 = (1 << 48) - 1;

/// The core of entity data. For performance reasons, this is stored in struct-of-arrays
/// form, so that SIMD operations can be used to access it.
/// For now, these are just standard vectors, with further SIMD optimizations coming later.
/// The in-memory representation is subject to change and other code should not assume any particular
/// layout.
struct EntityCoreArray {
    // The next index to insert into. It may either be an unused slot or at the end of the arrays
    // This is an optimization for allocation speed; for now let's set it aside
    // next_insert: usize,
    // The physical len of the arrays.
    len: usize,

    base_time: Instant,

    // Lookup table from ID to index. This should only be used for scalar operations on a single entity (or a few)
    id_lookup: FxHashMap<u64, usize>,

    // Entity ID, unique, assigned by the server. Only 48 bits available to allow for possible
    // packing later
    id: Vec<u64>,
    // Flags
    control: Vec<u8>,
    // Tracks the last modification that actually conveys new information (e.g. to a client)
    // Simply exhausting a move and starting the next queued move does not count.
    last_nontrivial_modification: Vec<u64>,
    // Position
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    // Velocity
    xv: Vec<f32>,
    yv: Vec<f32>,
    zv: Vec<f32>,
    // Acceleration
    xa: Vec<f32>,
    ya: Vec<f32>,
    za: Vec<f32>,
    // Face direction, radians
    theta_y: Vec<f32>,
    // The total time for the current move
    move_time: Vec<f32>,
    // The time remaining in the current move
    move_time_elapsed: Vec<f32>,
    recalc_in: Vec<f32>,

    // Buffer one additional move so we don't have to stop while waiting for it to be calculated
    next_xv: Vec<f32>,
    next_yv: Vec<f32>,
    next_zv: Vec<f32>,
    next_xa: Vec<f32>,
    next_ya: Vec<f32>,
    next_za: Vec<f32>,
    next_theta_y: Vec<f32>,
    next_move_time: Vec<f32>,

    mod_count: Vec<u64>,

    // TODO: Once thin_box is stabilized, we can use it here
    coroutine: Vec<Option<Pin<Box<dyn EntityCoroutine>>>>,
}
impl EntityCoreArray {
    /// Checks some preconditions that we want to assume on the vectors to make
    /// iteration faster.
    fn check_preconditions(&self) {
        // We assume that all vectors have the same length.
        assert_eq!(self.id.len(), self.len);
        assert_eq!(self.control.len(), self.len);
        assert_eq!(self.last_nontrivial_modification.len(), self.len);
        assert_eq!(self.x.len(), self.len);
        assert_eq!(self.y.len(), self.len);
        assert_eq!(self.z.len(), self.len);
        assert_eq!(self.xv.len(), self.len);
        assert_eq!(self.yv.len(), self.len);
        assert_eq!(self.zv.len(), self.len);
        assert_eq!(self.xa.len(), self.len);
        assert_eq!(self.ya.len(), self.len);
        assert_eq!(self.za.len(), self.len);
        assert_eq!(self.theta_y.len(), self.len);
        assert_eq!(self.move_time_elapsed.len(), self.len);
        assert_eq!(self.recalc_in.len(), self.len);
        assert_eq!(self.next_xv.len(), self.len);
        assert_eq!(self.next_yv.len(), self.len);
        assert_eq!(self.next_zv.len(), self.len);
        assert_eq!(self.next_xa.len(), self.len);
        assert_eq!(self.next_ya.len(), self.len);
        assert_eq!(self.next_za.len(), self.len);
        assert_eq!(self.next_theta_y.len(), self.len);
        assert_eq!(self.next_move_time.len(), self.len);
        assert_eq!(self.mod_count.len(), self.len);
        assert_eq!(self.coroutine.len(), self.len);
    }

    fn update_times(&mut self, delta_time: f32) -> Duration {
        let _span = span!("entity_update_times");

        self.check_preconditions();
        assert!(delta_time >= 0.0);
        let mut next_event = std::f32::MAX;

        for i in 0..self.len {
            let next_event_i = self.update_time_single(i, delta_time);
            if next_event_i < next_event {
                next_event = next_event_i;
            }
        }
        // Duration is unhappy with f32::MAX. Let's clamp it to a reasonable time to wait
        // for the next event.
        const MAX_WAIT_TIME: f32 = 10.0;

        let next_event = Duration::from_secs_f32(next_event.min(MAX_WAIT_TIME).max(0.0));

        next_event
    }

    /// Run coroutines present on entities that need coroutine based recalcs.
    ///
    /// Args:
    ///   * if_remaining_time_less: Only run coroutines if the remaining move time is less than this parameter.
    ///     This allows a simple prioritization of coroutines that are in greater danger of running out of move time.
    ///
    /// TODO: more and better scheduling/prioritization algorithms
    fn run_coroutines(&mut self, services: &EntityCoroutineServices<'_>, delta_time: f32) {
        let _span = span!("entity_run_coroutines");
        // TODO remove this logspam
        tracing::info!("Running coroutines for {} entities", self.len);

        self.check_preconditions();
        for i in 0..self.len {
            self.run_coro_single(i, services, delta_time);
        }
    }

    fn reset_movement(
        &mut self,
        index: usize,
        pos: Vector3<f64>,
        m1: Movement,
        m2: Option<Movement>,
    ) {
        self.last_nontrivial_modification[index] = self.to_offset(Instant::now());
        self.x[index] = pos.x;
        self.y[index] = pos.y;
        self.z[index] = pos.z;
        self.xv[index] = m1.velocity.x;
        self.yv[index] = m1.velocity.y;
        self.zv[index] = m1.velocity.z;
        self.xa[index] = m1.acceleration.x;
        self.ya[index] = m1.acceleration.y;
        self.za[index] = m1.acceleration.z;
        self.theta_y[index] = m1.face_direction;
        self.move_time[index] = m1.move_time;
        self.move_time_elapsed[index] = 0.0;
        self.recalc_in[index] = if m2.is_some() {
            f32::MAX
        } else if self.control[index] & CONTROL_AUTONOMOUS == 0 {
            f32::MAX
        } else {
            0.0
        };
        if let Some(m2) = m2 {
            self.next_xv[index] = m2.velocity.x;
            self.next_yv[index] = m2.velocity.y;
            self.next_zv[index] = m2.velocity.z;
            self.next_xa[index] = m2.acceleration.x;
            self.next_ya[index] = m2.acceleration.y;
            self.next_za[index] = m2.acceleration.z;
            self.next_theta_y[index] = m2.face_direction;
            self.next_move_time[index] = m2.move_time;
        } else {
            self.next_xv[index] = 0.0;
            self.next_yv[index] = 0.0;
            self.next_zv[index] = 0.0;
            self.next_xa[index] = 0.0;
            self.next_ya[index] = 0.0;
            self.next_za[index] = 0.0;
            self.next_theta_y[index] = 0.0;
            self.next_move_time[index] = f32::MAX;
        }
        self.mod_count[index] += 2;
    }

    fn new() -> EntityCoreArray {
        EntityCoreArray {
            len: 0,
            id_lookup: FxHashMap::default(),
            id: vec![],
            control: vec![],
            last_nontrivial_modification: vec![],
            x: vec![],
            y: vec![],
            z: vec![],
            xv: vec![],
            yv: vec![],
            zv: vec![],
            xa: vec![],
            ya: vec![],
            za: vec![],
            theta_y: vec![],
            move_time: vec![],
            move_time_elapsed: vec![],
            recalc_in: vec![],
            next_xv: vec![],
            next_yv: vec![],
            next_zv: vec![],
            next_xa: vec![],
            next_ya: vec![],
            next_za: vec![],
            next_theta_y: vec![],
            next_move_time: vec![],
            mod_count: vec![],
            coroutine: vec![],
            base_time: Instant::now(),
        }
    }

    fn to_offset(&self, when: Instant) -> u64 {
        let offset = when
            .duration_since(self.base_time)
            .as_nanos()
            .try_into()
            .unwrap();
        if offset & (1 << 63) != 0 {
            tracing::warn!("Offset {offset} from {:?} is halfway to overflowing a u64 nanos counter. Either you've managed a very impressive uptime, or something's gone wrong.", self.base_time);
        }
        offset
    }

    fn from_offset(&self, i: u64) -> Instant {
        self.base_time + Duration::from_nanos(i)
    }

    fn insert(
        &mut self,
        id: u64,
        position: Vector3<f64>,
        coroutine: Option<Pin<Box<dyn EntityCoroutine>>>,
    ) -> usize {
        // scan for the first unused slot
        let index = self
            .control
            .iter()
            .enumerate()
            .find(|(_, &control)| control & CONTROL_PRESENT == 0)
            .map(|(i, _)| i);
        let control = if coroutine.is_some() {
            CONTROL_PRESENT | CONTROL_AUTONOMOUS | CONTROL_AUTONOMOUS_NEEDS_CALC
        } else {
            CONTROL_PRESENT
        };
        match index {
            Some(i) => {
                tracing::info!("Inserting {} at index {} in entity array", id, i);
                self.id_lookup.insert(id, i);
                self.id[i] = id;
                self.control[i] = control;
                self.last_nontrivial_modification[i] = self.to_offset(Instant::now());
                self.x[i] = position.x;
                self.y[i] = position.y;
                self.z[i] = position.z;
                self.xv[i] = 0.0;
                self.yv[i] = 0.0;
                self.zv[i] = 0.0;
                self.xa[i] = 0.0;
                self.ya[i] = 0.0;
                self.za[i] = 0.0;
                self.theta_y[i] = 0.0;
                self.move_time[i] = 0.0;
                self.move_time_elapsed[i] = 0.0;
                self.recalc_in[i] = 0.0;
                self.next_xv[i] = 0.0;
                self.next_yv[i] = 0.0;
                self.next_zv[i] = 0.0;
                self.next_xa[i] = 0.0;
                self.next_ya[i] = 0.0;
                self.next_za[i] = 0.0;
                self.next_theta_y[i] = 0.0;
                self.next_move_time[i] = 0.0;
                self.coroutine[i] = coroutine;
                self.recalc_in[i] = 0.0;
                self.mod_count[i] = 0;
                i
            }
            None => {
                tracing::info!("Inserting {} at end of entity array", id);
                self.id_lookup.insert(id, self.len);
                // we need to grow the array
                self.len += 1;
                self.id.push(id);
                self.control.push(control);
                self.last_nontrivial_modification
                    .push(self.to_offset(Instant::now()));
                self.x.push(position.x);
                self.y.push(position.y);
                self.z.push(position.z);
                self.xv.push(0.0);
                self.yv.push(0.0);
                self.zv.push(0.0);
                self.xa.push(0.0);
                self.ya.push(0.0);
                self.za.push(0.0);
                self.theta_y.push(0.0);
                self.move_time.push(0.0);
                self.move_time_elapsed.push(0.0);
                self.recalc_in.push(0.0);
                self.next_xv.push(0.0);
                self.next_yv.push(0.0);
                self.next_zv.push(0.0);
                self.next_xa.push(0.0);
                self.next_ya.push(0.0);
                self.next_za.push(0.0);
                self.next_theta_y.push(0.0);
                self.next_move_time.push(0.0);
                self.mod_count.push(0);
                self.coroutine.push(coroutine);

                self.len - 1
            }
        }
    }

    fn remove(&mut self, id: u64) -> Result<(), EntityError> {
        if let Some(i) = self.id_lookup.remove(&id) {
            self.len -= 1;
            self.id.swap_remove(i);
            self.control.swap_remove(i);
            self.last_nontrivial_modification.swap_remove(i);
            self.x.swap_remove(i);
            self.y.swap_remove(i);
            self.z.swap_remove(i);
            self.xv.swap_remove(i);
            self.yv.swap_remove(i);
            self.zv.swap_remove(i);
            self.xa.swap_remove(i);
            self.ya.swap_remove(i);
            self.za.swap_remove(i);
            self.theta_y.swap_remove(i);
            self.move_time.swap_remove(i);
            self.move_time_elapsed.swap_remove(i);
            self.recalc_in.swap_remove(i);
            self.next_xv.swap_remove(i);
            self.next_yv.swap_remove(i);
            self.next_zv.swap_remove(i);
            self.next_xa.swap_remove(i);
            self.next_ya.swap_remove(i);
            self.next_za.swap_remove(i);
            self.next_theta_y.swap_remove(i);
            self.next_move_time.swap_remove(i);
            self.coroutine.swap_remove(i);
            // fix up the id lookup since we rearranged stuff
            if i != self.len {
                self.id_lookup.insert(self.id[i], i);
            }
            Ok(())
        } else {
            Err(EntityError::NotFound)
        }
    }

    fn set_kinematics(
        &mut self,
        id: u64,
        position: Vector3<f64>,
        current: Movement,
        next: Option<Movement>,
    ) -> Result<usize, EntityError> {
        let index = self.find_entity(id)?;
        self.reset_movement(index, position, current, next);
        Ok(index)
    }

    fn find_entity(&self, id: u64) -> Result<usize, EntityError> {
        self.id_lookup
            .get(&id)
            .copied()
            .ok_or(EntityError::NotFound)
    }

    pub(crate) fn for_each_entity(
        &self,
        since: Option<Instant>,
        mut f: impl FnMut(IterEntity) -> Result<()>,
    ) -> Result<()> {
        for i in 0..self.len {
            if self.control[i] & CONTROL_PRESENT == 0 {
                continue;
            }
            let lnm = self.from_offset(self.last_nontrivial_modification[i]);
            if since.is_some_and(|x| x > lnm) {
                continue;
            }

            let entity = IterEntity {
                id: self.id[i],
                starting_position: Vector3::new(self.x[i], self.y[i], self.z[i]),
                instantaneous_position: Vector3::new(
                    qproj(self.x[i], self.xv[i], self.xa[i], self.move_time_elapsed[i]),
                    qproj(self.y[i], self.yv[i], self.ya[i], self.move_time_elapsed[i]),
                    qproj(self.z[i], self.zv[i], self.za[i], self.move_time_elapsed[i]),
                ),
                current_move: Movement {
                    velocity: Vector3::new(self.xv[i], self.yv[i], self.zv[i]),
                    acceleration: Vector3::new(self.xa[i], self.ya[i], self.za[i]),
                    face_direction: self.theta_y[i],
                    move_time: self.move_time[i],
                },
                next_move: if self.next_move_time[i] == f32::MAX {
                    None
                } else {
                    Some(Movement {
                        velocity: Vector3::new(self.next_xv[i], self.next_yv[i], self.next_zv[i]),
                        acceleration: Vector3::new(
                            self.next_xa[i],
                            self.next_ya[i],
                            self.next_za[i],
                        ),
                        face_direction: self.next_theta_y[i],
                        move_time: self.next_move_time[i],
                    })
                },
                last_nontrivial_modification: lnm,
                current_move_elapsed: self.move_time_elapsed[i],
                mod_count: self.mod_count[i],
            };
            f(entity)?;
        }
        Ok(())
    }

    // We *really* want this to be inlined into the loop
    #[inline(always)]
    fn run_coro_single(
        &mut self,
        i: usize,
        services: &EntityCoroutineServices<'_>,
        delta_time: f32,
    ) {
        self.recalc_in[i] -= delta_time;
        if self.recalc_in[i] < 0.0
            && self.control[i] & (CONTROL_AUTONOMOUS | CONTROL_PRESENT)
                == (CONTROL_AUTONOMOUS | CONTROL_PRESENT)
        {
            self.control[i] |= CONTROL_AUTONOMOUS_NEEDS_CALC;
        }

        if self.control[i] & (CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS | CONTROL_PRESENT)
            != (CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS | CONTROL_PRESENT)
        {
            return;
        }
        let remaining_time = self.move_time[i] - self.move_time_elapsed[i];

        if let Some(coroutine) = self.coroutine[i].as_mut() {
            // do some kinematics
            let estimated_x = qproj(self.x[i], self.xv[i], self.xa[i], self.move_time[i]);
            let estimated_y = qproj(self.y[i], self.yv[i], self.ya[i], self.move_time[i]);
            let estimated_z = qproj(self.z[i], self.zv[i], self.za[i], self.move_time[i]);
            match coroutine.as_mut().plan_move(
                services,
                vec3(self.x[i], self.y[i], self.z[i]),
                vec3(estimated_x, estimated_y, estimated_z),
                remaining_time,
            ) {
                CoroutineResult::Successful(EntityMoveDecision::QueueUpMovement(movement)) => {
                    println!("Queueing up movement for entity {}", self.id[i]);
                    println!(
                        "Current movement time: {} of {}, next {}",
                        self.move_time_elapsed[i], self.move_time[i], movement.move_time
                    );
                    self.last_nontrivial_modification[i] = self.to_offset(Instant::now());
                    self.next_xv[i] = movement.velocity.x;
                    self.next_yv[i] = movement.velocity.y;
                    self.next_zv[i] = movement.velocity.z;
                    self.next_xa[i] = movement.acceleration.x;
                    self.next_ya[i] = movement.acceleration.y;
                    self.next_za[i] = movement.acceleration.z;
                    self.next_theta_y[i] = movement.face_direction;
                    self.next_move_time[i] = movement.move_time;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    // We'll trigger a recalc when the movement gets advanced; no need to use recalc_in
                    self.recalc_in[i] = f32::MAX;
                }
                CoroutineResult::Successful(EntityMoveDecision::AskAgainLater(delay)) => {
                    self.recalc_in[i] = delay;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                }
                CoroutineResult::Successful(EntityMoveDecision::ImmediateDespawn) => {
                    self.last_nontrivial_modification[i] = self.to_offset(Instant::now());
                    // todo notify watchers that the entity is despawning
                    log::warn!(
                        "Entity {} is being despawned; notification not yet implemented",
                        self.id[i]
                    );
                    self.control[i] = 0;
                }
                CoroutineResult::Successful(EntityMoveDecision::ResetMovement(pos, m1, m2)) => {
                    self.reset_movement(i, pos, m1, m2);
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                }
                CoroutineResult::Successful(EntityMoveDecision::StopCoroutineControl) => {
                    self.control[i] &= !(CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS);
                }
                CoroutineResult::_Deferred(_) => todo!(),
            }
        }
    }

    // We *really* want this to be inlined into the loop
    #[inline(always)]
    /// Update the time elapsed for an entity
    /// Returns the time when the move will end and a new calculation is needed
    fn update_time_single(&mut self, i: usize, delta_time: f32) -> f32 {
        self.move_time_elapsed[i] += delta_time;
        // We don't update recalc_in, the coroutine handler will do that

        // if we finished a move
        if self.move_time_elapsed[i] >= self.move_time[i] {
            // Then we're going to need a new move to be queued up soon
            if self.control[i] & CONTROL_AUTONOMOUS != 0 {
                self.control[i] |= CONTROL_AUTONOMOUS_NEEDS_CALC;
            }
            // Apply the effect of the current move we just finished
            self.x[i] = qproj(self.x[i], self.xv[i], self.xa[i], self.move_time[i]);
            self.y[i] = qproj(self.y[i], self.yv[i], self.ya[i], self.move_time[i]);
            self.z[i] = qproj(self.z[i], self.zv[i], self.za[i], self.move_time[i]);
            // And move the next move into the current one
            self.xv[i] = self.next_xv[i];
            self.yv[i] = self.next_yv[i];
            self.zv[i] = self.next_zv[i];
            self.xa[i] = self.next_xa[i];
            self.ya[i] = self.next_ya[i];
            self.za[i] = self.next_za[i];
            self.theta_y[i] = self.next_theta_y[i];
            self.move_time[i] = self.next_move_time[i];
            self.move_time_elapsed[i] = 0.0;
            // And reset the next move to be a stop-and-stay
            self.next_xa[i] = 0.0;
            self.next_ya[i] = 0.0;
            self.next_za[i] = 0.0;
            self.next_xv[i] = 0.0;
            self.next_yv[i] = 0.0;
            self.next_zv[i] = 0.0;
            self.next_move_time[i] = f32::MAX;

            self.recalc_in[i] = f32::MAX;
            self.mod_count[i] += 1;
        }
        // We now have updated the move (if necessary); here's how long is left in it.
        let remaining_time_kinematics = self.move_time[i] - self.move_time_elapsed[i];
        if self.control[i] & (CONTROL_AUTONOMOUS_NEEDS_CALC) != 0 {
            remaining_time_kinematics.min(self.recalc_in[i])
        } else {
            remaining_time_kinematics
        }
    }
}
/// Project a coordinate into the future.
#[inline]
fn qproj(s: f64, v: f32, a: f32, t: f32) -> f64 {
    s + (v * t + 0.5 * a * t * t) as f64
}

enum EntityAction {
    Insert(u64, Vector3<f64>, Option<Pin<Box<dyn EntityCoroutine>>>),
    Remove(u64),
    SetKinematics(u64, Vector3<f64>, Movement, Option<Movement>),
}

/// The entities stored in a shard of the game map.
pub struct EntityShard {
    shard_id: usize,
    db: Arc<dyn GameDatabase>,
    core: RwLock<EntityCoreArray>,
    pending_actions_rx: Mutex<tokio::sync::mpsc::Receiver<EntityAction>>,
    pending_actions_tx: tokio::sync::mpsc::Sender<EntityAction>,
}
impl EntityShard {
    pub(super) fn new(shard_id: usize, db: Arc<dyn GameDatabase>) -> EntityShard {
        tracing::warn!("FIXME: Need to load entities from DB for {shard_id}");
        let core = RwLock::new(EntityCoreArray::new());
        const ENTITY_QUEUE_SIZE: usize = 64;
        let (tx, rx) = tokio::sync::mpsc::channel(ENTITY_QUEUE_SIZE);
        EntityShard {
            shard_id,
            db,
            core,
            pending_actions_rx: Mutex::new(rx),
            pending_actions_tx: tx,
        }
    }

    pub(crate) fn for_each_entity(
        &self,
        since: Option<Instant>,
        f: impl FnMut(IterEntity) -> Result<()>,
    ) -> Result<()> {
        self.core.read().for_each_entity(since, f)
    }
}

pub(crate) struct IterEntity {
    pub(crate) id: u64,
    pub(crate) starting_position: Vector3<f64>,
    pub(crate) instantaneous_position: Vector3<f64>,
    pub(crate) current_move: Movement,
    pub(crate) next_move: Option<Movement>,
    pub(crate) last_nontrivial_modification: Instant,
    pub(crate) current_move_elapsed: f32,
    pub(crate) mod_count: u64,
}

pub(crate) struct EntityShardWorker {
    entities: Arc<EntityManager>,
    shard_id: usize,
    game_state: Arc<GameState>,
    cancellation: CancellationToken,
}

const COMMAND_BATCH_SIZE: usize = 16;
//const SCHEDULING_BUFFER: Duration = Duration::from_millis(100);
impl EntityShardWorker {
    pub(crate) fn new(
        entities: Arc<EntityManager>,
        shard_id: usize,
        game_state: Arc<GameState>,
        cancellation: CancellationToken,
    ) -> EntityShardWorker {
        EntityShardWorker {
            entities,
            shard_id,
            game_state,
            cancellation,
        }
    }

    pub(crate) async fn run_loop(&self) -> Result<()> {
        let mut last_iteration = Instant::now();
        let mut rx_lock = self.entities.shards[self.shard_id]
            .pending_actions_rx
            .lock();
        let mut rx_messages = Vec::with_capacity(COMMAND_BATCH_SIZE);

        'main_loop: while !self.cancellation.is_cancelled() {
            let now = Instant::now();
            let dt = now.duration_since(last_iteration);

            let mut next_awakening = {
                let mut lock = self.entities.shards[self.shard_id].core.write();
                let services = EntityCoroutineServices {
                    map: &self.game_state.map,
                };
                let dt_f32 = dt.as_secs_f32();
                lock.run_coroutines(&services, dt_f32);
                let next_awakening_times = lock.update_times(dt_f32);
                tracing::debug!(
                    "Entity worker for shard {} next awakening times: {:?}",
                    self.shard_id,
                    next_awakening_times
                );
                drop(lock);
                now + next_awakening_times
            };

            last_iteration = now;

            // TODO: This should use select! and some signal that lets us detect
            // new entries, as well as cancellations
            rx_messages.clear();
            'poll: loop {
                tracing::debug!(
                    "Entity worker for shard {} polling with sleep for {:?}",
                    self.shard_id,
                    next_awakening - Instant::now()
                );
                tokio::select! {
                    // TODO evaluate whether biased sampling makes sense here
                    biased;
                    count = rx_lock.recv_many(&mut rx_messages, COMMAND_BATCH_SIZE) => {
                        tracing::debug!("Entity worker for shard {} received {} messages", self.shard_id, count);
                        match count {
                            0 => {
                                tracing::warn!("Entity worker for shard {} shutting down because sender disappeared", self.shard_id);
                                break 'main_loop;
                            },
                            _ => {
                                next_awakening = next_awakening.min(self.handle_messages(&mut rx_messages));
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(next_awakening.into()) => {
                        break 'poll;
                    }
                    _ = self.cancellation.cancelled() => {
                        tracing::info!("Entity worker for shard {} shutting down", self.shard_id);
                        break 'main_loop;
                    },
                }
            }
        }
        Ok(())
    }

    fn handle_messages(&self, rx_messages: &mut Vec<EntityAction>) -> Instant {
        let mut lock = self.entities.shards[self.shard_id].core.write();
        let mut indices: smallvec::SmallVec<[usize; COMMAND_BATCH_SIZE]> =
            smallvec::SmallVec::new();
        for action in rx_messages.drain(..) {
            match action {
                EntityAction::Insert(id, position, coroutine) => {
                    let index = lock.insert(id, position, coroutine);
                    indices.push(index);
                }
                EntityAction::Remove(id) => match lock.remove(id) {
                    Ok(_) => {}
                    Err(EntityError::NotFound) => {
                        tracing::warn!("Tried to remove non-existent entity {}", id);
                    }
                },
                EntityAction::SetKinematics(id, position, movement, next_movement) => {
                    match lock.set_kinematics(id, position, movement, next_movement) {
                        Ok(index) => {
                            indices.push(index);
                        }
                        Err(EntityError::NotFound) => {
                            tracing::warn!(
                                "Tried to set kinematics for non-existent entity {}",
                                id
                            );
                        }
                    }
                }
            }
        }
        indices.sort();
        indices.dedup();
        let mut next_event = std::f32::MAX;
        for index in indices {
            lock.run_coro_single(
                index,
                &EntityCoroutineServices {
                    map: &self.game_state.map,
                },
                0.0,
            );
            next_event = next_event.min(lock.update_time_single(index, 0.0));
        }
        Instant::now() + Duration::from_secs_f32(next_event.min(10.0))
    }
}

#[derive(Debug, Error)]
pub enum EntityError {
    #[error("Entity not found")]
    NotFound,
}
