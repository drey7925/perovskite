use std::ops::DerefMut;
/// This implementation is EXTREMELY unstable while it is under active development.
/// Do not assume any functionality or performance guarantees while different techniques
/// are under investigation.
use std::{
    any::Any,
    fmt::Debug,
    ops::Range,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Error, Result};
use cgmath::{vec3, Vector3, Zero};
use circular_buffer::CircularBuffer;

use futures::Future;
use lazy_static::lazy_static;
use parking_lot::{Mutex, RwLock};
use perovskite_core::{
    block_id::BlockId,
    constants::textures::FALLBACK_UNKNOWN_TEXTURE,
    coordinates::BlockCoordinate,
    protocol::{
        self,
        entities::{EntityAppearance, EntityTypeAssignment, ServerEntityTypeAssignments},
    },
    util::{TraceBuffer, TraceLog},
};
use prost::Message;

use rustc_hash::FxHashMap;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracy_client::span;

use crate::game_state::event::EventInitiator;
use crate::{
    database::database_engine::{GameDatabase, KeySpace},
    formats, run_handler, CachelineAligned,
};

use super::{blocks::ExtendedDataHolder, event::HandlerContext, GameState};

/// This entity is present and valid
const CONTROL_PRESENT: u8 = 1;
/// This entity should remain loaded even when it's in unloaded chunks
const CONTROL_STICKY: u8 = 2;

/// This entity has a coroutine controlling it.
const CONTROL_AUTONOMOUS: u8 = 4;
/// This entity should follow simple physics without needing a coroutine to drive it.
const CONTROL_SIMPLE_PHYSICS: u8 = 8;

/// This entity has space in its move queue and needs a move to be calculated
const CONTROL_AUTONOMOUS_NEEDS_CALC: u8 = 16;

/// This entity has a coroutine, but it's suspended while waiting for a completion from tokio
const CONTROL_SUSPENDED: u8 = 32;

/// This entity's current move is synthetic (no moves were available) and should be dequeued as soon as possible
const CONTROL_DEQUEUE_WHEN_READY: u8 = 64;

struct Completion<T: Send + Sync + 'static> {
    /// The entity ID we'll deliver the result to
    entity_id: u64,
    /// The index in the entity array. If we don't find the given ID there, we'll drop the completion.
    index: usize,
    /// The result we're delivering.
    result: T,
    /// The trace buffer (may be an empty one)
    trace_buffer: TraceBuffer,
}

// A deferred call to be invoked on a tokio executor, same as Deferral but private
struct DeferredPrivate<T: Send + Sync + 'static> {
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
    deferred_call: Pin<Box<(dyn Future<Output = T> + Send + 'static)>>,

    trace_buffer: TraceBuffer,
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
    types: Arc<EntityTypeManager>,
}

impl EntityManager {
    pub(crate) fn predictive_position(&self, entity_id: u64, tick: u64) -> Option<Vector3<f64>> {
        let shard = self.get_shard(entity_id);
        let core = shard.core.read();
        let index = core.id_lookup.get(&entity_id).copied()?;

        let elapsed_ticks = tick.saturating_sub(core.move_start_tick[index]);
        let elapsed = elapsed_ticks as f32 / 1_000_000_000.0;
        if elapsed < core.move_time[index] {
            return Some(Vector3::new(
                qproj(core.x[index], core.xv[index], core.xa[index], elapsed),
                qproj(core.y[index], core.yv[index], core.ya[index], elapsed),
                qproj(core.z[index], core.zv[index], core.za[index], elapsed),
            ));
        }
        let mut remaining = elapsed - core.move_time[index];
        let mut acc_pos = Vector3::new(
            qproj(
                core.x[index],
                core.xv[index],
                core.xa[index],
                core.move_time[index],
            ),
            qproj(
                core.y[index],
                core.yv[index],
                core.ya[index],
                core.move_time[index],
            ),
            qproj(
                core.z[index],
                core.zv[index],
                core.za[index],
                core.move_time[index],
            ),
        );
        for movement in core.move_queue[index].as_slice() {
            let m = &movement.movement;
            if remaining < m.move_time {
                return Some(Vector3::new(
                    qproj(acc_pos.x, m.velocity.x, m.acceleration.x, remaining),
                    qproj(acc_pos.y, m.velocity.y, m.acceleration.y, remaining),
                    qproj(acc_pos.z, m.velocity.z, m.acceleration.z, remaining),
                ));
            } else {
                remaining -= m.move_time;
                acc_pos = Vector3::new(
                    qproj(acc_pos.x, m.velocity.x, m.acceleration.x, m.move_time),
                    qproj(acc_pos.y, m.velocity.y, m.acceleration.y, m.move_time),
                    qproj(acc_pos.z, m.velocity.z, m.acceleration.z, m.move_time),
                )
            }
        }

        None
    }
}

impl EntityManager {
    pub(crate) fn new(
        db: Arc<dyn GameDatabase>,
        entity_types: Arc<EntityTypeManager>,
        base_time: Instant,
    ) -> EntityManager {
        EntityManager {
            shards: std::array::from_fn(|shard_id| {
                CachelineAligned(EntityShard::new(shard_id, db.clone(), base_time))
            }),
            workers: Mutex::new(Vec::new()),
            cancellation: CancellationToken::new(),
            // TODO next_id needs to be initialized from the database eventually
            //   for now, just use 1
            // We never want to have an id of 0 since we need a sentinel value in
            // the RPC protos.
            next_id: AtomicU64::new(1),
            types: entity_types,
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
        for handle in std::mem::take(self.workers.lock().deref_mut()).into_iter() {
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
        entity_def: EntityTypeId,
        trailing_entities: Option<TrailingEntities>,
    ) -> u64 {
        let id = self.assign_next_id();
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .send(EntityAction::Insert(
                id,
                position,
                coroutine,
                entity_def,
                trailing_entities,
            ))
            .await
            .context("Entity receiver disappeared")
            .unwrap();
        id
    }
    pub fn new_entity_blocking(
        &self,
        position: Vector3<f64>,
        coroutine: Option<Pin<Box<dyn EntityCoroutine>>>,
        entity_def: EntityTypeId,
        trailing_entities: Option<TrailingEntities>,
    ) -> u64 {
        let id = self.assign_next_id();
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .blocking_send(EntityAction::Insert(
                id,
                position,
                coroutine,
                entity_def,
                trailing_entities,
            ))
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
        next: InitialMoveQueue,
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
        next: InitialMoveQueue,
    ) {
        let shard = self.get_shard(id);
        shard
            .pending_actions_tx
            .blocking_send(EntityAction::SetKinematics(id, position, current, next))
            .context("Entity receiver disappeared")
            .unwrap();
    }

    fn get_by_type(&self, entity_type: &EntityTypeId) -> &EntityDef {
        match self.types.types.get(entity_type.class.0 as usize) {
            Some(def) => def.as_ref().unwrap_or(&UNDEFINED_ENTITY),
            None => &UNDEFINED_ENTITY,
        }
    }

    pub fn types(&self) -> &EntityTypeManager {
        &self.types
    }
}

lazy_static! {
    pub(crate) static ref UNDEFINED_ENTITY: EntityDef = EntityDef {
        move_queue_type: MoveQueueType::SingleMove,
        class_name: "builtin:undefined".to_string(),
        client_info: EntityAppearance {
            // TODO generate from the OBJ file
            custom_mesh: vec![],
            attachment_offset: Some(vec3(0.0, 0.0, 0.0).try_into().unwrap()),
        attachment_offset_in_model_space: false
        },
    };
}

// Note that if an entity has neither CONTROL_AUTONOMOUS nor CONTROL_SIMPLE_PHYSICS, it will
// only move when programatically requested to do so by a caller
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EntityMoveDecision {
    /// Add this movement to the queue.
    /// Queueing movements will give the smoothest animation as long as the queue
    /// is not allowed to be exhausted
    QueueSingleMovement(Movement),
    /// Add these movements to the queue.
    ///
    /// Note: The type of this variant may change, e.g. if we decide to avoid
    /// memory allocations
    QueueUpMultiple(Vec<Movement>),
    /// Get rid of the current movement, set the position, start the new movement immediately, and queue up the following movement (if it's Some).
    /// Note that this may lead to visual glitches due to unsmoothed movement and network latency.
    ///
    /// This can be used to change the type of move queue if needed.
    ResetMovement(Vector3<f64>, Movement, InitialMoveQueue),
    /// Despawn the entity immediately
    ImmediateDespawn,
    /// Ask again later, in this many seconds
    AskAgainLater(f32),
    /// Ask again later, but we're flexible about when to ask again
    AskAgainLaterFlexible(Range<f32>),
    /// Stop moving, and stop calling the coroutine asking for more moves, until
    /// manually re-enabled from non-coroutine code
    StopCoroutineControl,
}

#[must_use = "coroutine result does nothing unless returned to the entity scheduler"]
pub enum CoroutineResult {
    /// The coroutine returned successfully
    Successful(EntityMoveDecision),

    #[allow(private_interfaces)]
    #[doc(hidden)]
    /// Invoke this deferred call and use its result as the coroutine result. This cannot be constructed by hand.
    ///
    /// At the moment, this schedules onto a tokio task pool as a blocking task.
    _DeferredMoveResult(DeferredPrivate<EntityMoveDecision>),

    #[allow(private_interfaces)]
    #[doc(hidden)]
    /// Schedule a future call to the coroutine at a later point when the result is available.
    /// This cannot be constructed by hand.
    ///
    /// At the moment, this schedules onto a tokio task pool as a blocking task.
    _DeferredReenterCoroutine(DeferredPrivate<ContinuationResult>),
}
impl From<EntityMoveDecision> for CoroutineResult {
    fn from(m: EntityMoveDecision) -> Self {
        Self::Successful(m)
    }
}
impl CoroutineResult {
    pub fn with_trace_buffer(self, buffer: TraceBuffer) -> Self {
        match self {
            Self::Successful(decision) => {
                buffer.log("CoroutineResult::Successful");
                drop(buffer);
                Self::Successful(decision)
            }
            CoroutineResult::_DeferredMoveResult(x) => {
                buffer.log("attaching");
                CoroutineResult::_DeferredMoveResult(DeferredPrivate {
                    deferred_call: x.deferred_call,
                    trace_buffer: buffer,
                })
            }
            CoroutineResult::_DeferredReenterCoroutine(x) => {
                buffer.log("attaching");
                CoroutineResult::_DeferredReenterCoroutine(DeferredPrivate {
                    deferred_call: x.deferred_call,
                    trace_buffer: buffer,
                })
            }
        }
    }

    fn log_trace(&self, arg: &'static str) {
        match self {
            CoroutineResult::Successful(_) => {}
            CoroutineResult::_DeferredMoveResult(x) => x.trace_buffer.log(arg),
            CoroutineResult::_DeferredReenterCoroutine(x) => x.trace_buffer.log(arg),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StoredMovement {
    pub(crate) sequence: u64,
    pub(crate) start_tick: u64,
    pub(crate) movement: Movement,
}

/// A single step in the path plan for an entity.
#[derive(Debug, Copy, Clone)]
pub struct Movement {
    /// The initial velocity in the movement, blocks/sec
    pub velocity: Vector3<f32>,
    /// The acceleration to apply throughout the movement, blocks/sec/sec
    /// This need not be parallel to the initial velocity, however the
    /// initial implementation of trailing entities may produce nonsensical results
    /// when non-parallel to the initial velocity
    pub acceleration: Vector3<f32>,
    /// The direction the entity is facing, in radians. 0 means facing +z
    pub face_direction: f32,
    /// The pitch up/down, in radians. Positive means pitch up
    pub pitch: f32,
    /// The length of the move, in seconds
    pub move_time: f32,
}
impl Movement {
    /// Returns a movement that will stop and stay at the current position for a given amount of time
    pub fn stop_and_stay(face_direction: f32, pitch: f32, move_time: f32) -> Self {
        Self {
            velocity: Vector3::zero(),
            acceleration: Vector3::zero(),
            face_direction,
            pitch,
            move_time,
        }
    }
    pub fn pos_after(&self, start: Vector3<f64>, time: f32) -> Vector3<f64> {
        Vector3::new(
            qproj(start.x, self.velocity.x, self.acceleration.x, time),
            qproj(start.y, self.velocity.y, self.acceleration.y, time),
            qproj(start.z, self.velocity.z, self.acceleration.z, time),
        )
    }
    pub fn pos_after_move(&self, start: Vector3<f64>) -> Vector3<f64> {
        self.pos_after(start, self.move_time)
    }
}

const LARGEST_POSSIBLE_QUEUE_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub enum InitialMoveQueue {
    /// Store up a single movement. If none, the callback will be reinvoked shortly
    SingleMove(Option<Movement>),
    /// Store up to eight moves. If the input has cardinality less than 8, the callback
    /// will be reinvoked shortly
    // Impl note: We expect this to be used fairly rarely, since most entities should
    // avoid reinitializing their movement queue too often. Also, we want to keep the
    // size of the return enum from CoroutineResult small, so we box this.
    Buffer8(Box<arrayvec::ArrayVec<Movement, 8>>),
    // TODO: Deeper buffers if necessary. 8 moves of one block each take less than 100 msec
    // when running an entity at the target max speed of 320 km/h
    Buffer64(Box<arrayvec::ArrayVec<Movement, 64>>),
}
impl InitialMoveQueue {
    fn has_slack(&self) -> bool {
        match self {
            InitialMoveQueue::SingleMove(m) => m.is_none(),
            InitialMoveQueue::Buffer8(m) => m.len() < 8,
            InitialMoveQueue::Buffer64(m) => m.len() < 32,
        }
    }
}

/// Services available to the entity coroutine.
///
/// This is the ONLY way that coroutines should interact with the game state.
///
/// It's technically possible for a coroutine to sneak an Arc<GameState> or similar into its own
/// state and try to use it - however **that will likely lead to deadlocks**.
pub struct EntityCoroutineServices<'a> {
    game_state: &'a Arc<GameState>,
    sender: &'a tokio::sync::mpsc::Sender<Completion<ContinuationResult>>,
}
impl<'a> EntityCoroutineServices<'a> {
    /// Gets the block at the specified coordinate, or None if the chunk isn't loaded.
    /// Forwards to the game map's try_get_block.
    pub fn try_get_block(&self, coord: BlockCoordinate) -> Option<BlockId> {
        self.game_state.game_map().try_get_block(coord)
    }

    pub fn get_block(
        &self,
        coord: BlockCoordinate,
    ) -> DeferrableResult<Result<BlockId>, BlockCoordinate> {
        //let rng = &mut rand::thread_rng();
        // if rng.gen_bool(0.01) {
        //     let map_clone = self.map.clone();
        //     return DeferrableResult::Deferred(Deferral {
        //         deferred_call: Box::new(move || (map_clone.get_block(coord), coord)),
        //     });
        // }

        if let Some(block) = self.try_get_block(coord) {
            (Ok(block), coord).into()
        } else {
            let map_clone = self.game_state.game_map_clone();
            DeferrableResult::Deferred(Deferral {
                deferred_call: Box::pin(async move {
                    tokio::task::block_in_place(move || (map_clone.get_block(coord), coord))
                }),
            })
        }
    }

    /// Modifies the block at the specified coordinate, if it is possible to do so without blocking.
    /// Otherwise, returns None.
    ///
    /// Implementation note: At this time, the need to block is checked before the mutator is run, so
    /// the mutator will only be invoked once in most cases (either inline if blocking can be avoided, or
    /// as a deferred computation if blocking was required).
    ///
    /// Note that this may change in the future, and it's possible for the mutator to be invoked multiple times.
    /// As with all calls to mutate_block_atomically, it's always possible that the block has changed between the time
    /// when the caller decides to call (based on some old block value) and when the mutator is actually invoked.
    ///
    /// The mutator itself must be very careful to avoid blocking for performance, *just like any other coroutine.*
    pub fn mutate_block_atomically<T: Send + Sync + 'static>(
        &self,
        coord: BlockCoordinate,
        mut mutator: impl FnMut(&mut BlockId, &mut ExtendedDataHolder) -> Result<T>
            + Send
            + Sync
            + 'static,
    ) -> DeferrableResult<Result<T>> {
        match self
            .game_state
            .game_map()
            .try_mutate_block_atomically(coord, &mut mutator)
        {
            Ok(Some(t)) => DeferrableResult::AvailableNow(Ok(t)),
            Ok(None) => {
                let map_clone = self.game_state.game_map_clone();
                tracing::debug!("MBA deferring");
                DeferrableResult::Deferred(Deferral {
                    deferred_call: Box::pin(async move {
                        tracing::debug!("MBA invoking");
                        tokio::task::block_in_place(move || {
                            (map_clone.mutate_block_atomically(coord, mutator), ())
                        })
                    }),
                })
            }
            Err(e) => DeferrableResult::AvailableNow(Err(e)),
        }
    }

    /// Spawns a future onto the entity coroutine's executor.
    ///
    /// For now, this just calls tokio::task::spawn with a delay, but this may change in the future.
    ///
    /// Note that this may not be the most efficient way to do delayed actions. For now, it's
    /// probably sufficient assuming that these actions are happening at a reasonable scale.
    pub fn spawn_delayed(
        &self,
        delay: Duration,
        task: impl FnOnce(&HandlerContext) + Send + 'static,
    ) {
        let ctx = HandlerContext {
            tick: self.game_state.tick(),
            initiator: EventInitiator::Plugin("entity_coroutine".to_string()),
            game_state: self.game_state.clone(),
        };
        tokio::task::spawn(async move {
            tokio::time::sleep(delay).await;
            tokio::task::block_in_place(|| task(&ctx));
        });
    }

    /// Spawns a future onto the entity coroutine's async executor.
    ///
    /// For now, this is just spawned onto the tokio runtime, but this may change in the future.
    ///
    /// Warning: If the function hangs indefinitely, the game cannot exit.
    pub fn spawn_async<T: Send + Sync + 'static, Fut>(
        &self,
        task: impl FnOnce(HandlerContext<'static>) -> Fut,
    ) -> Deferral<T>
    where
        Fut: Future<Output = T> + Send + 'static,
    {
        let ctx: HandlerContext<'static> = HandlerContext {
            tick: self.game_state.tick(),
            initiator: EventInitiator::Plugin("entity_coroutine".to_string()),
            game_state: self.game_state.clone(),
        };
        let fut = task(ctx);
        Deferral {
            deferred_call: Box::pin(async move {
                let result = fut.await;
                (result, ())
            }),
        }
    }
}

/// An abstraction over a computation that needs to be deferred because it can't
/// be run within the coroutine.
///
/// This is returned from various helpers in [EntityCoroutineServices] that may need to block
/// or run expensive computations.
#[must_use = "coroutine result does nothing unless returned to the entity scheduler"]
pub struct Deferral<T: Send + 'static, Residual: Debug + Send + 'static = ()> {
    deferred_call: Pin<Box<dyn Future<Output = (T, Residual)> + Send + 'static>>,
}
impl Deferral<Result<BlockId>, BlockCoordinate> {
    /// Returns a CoroutineResult that will instruct the entity system to run the deferred computation,
    /// and then re-enter the coroutine by calling [EntityCoroutine::continuation] with a ContinuationResult
    /// containing a [ContinuationResultValue::GetBlock] with the result of the deferred computation
    ///
    /// Args:
    ///     tag: The tag that will be passed in the continuation result when the deferred computation is
    ///         reentered
    pub fn defer_and_reinvoke(self, tag: u32) -> CoroutineResult {
        tracing::debug!("deferring and reinvoking");
        CoroutineResult::_DeferredReenterCoroutine(DeferredPrivate {
            deferred_call: Box::pin(async move {
                let (t, resid) = self.deferred_call.await;
                if let Ok(block) = t {
                    ContinuationResult {
                        value: ContinuationResultValue::GetBlock(block, resid),
                        tag,
                    }
                } else {
                    ContinuationResult {
                        value: ContinuationResultValue::Error(t.unwrap_err()),
                        tag,
                    }
                }
            }),
            trace_buffer: TraceBuffer::empty(),
        })
    }
}

macro_rules! impl_deferral {
    ($T:ty, $Variant: ident) => {
        impl Deferral<Result<$T>, ()> {
            #[must_use = "coroutine result does nothing unless returned to the entity scheduler"]
            pub fn defer_and_reinvoke(self, tag: u32) -> CoroutineResult {
                CoroutineResult::_DeferredReenterCoroutine(DeferredPrivate {
                    deferred_call: Box::pin(async move {
                        let (t, _) = self.deferred_call.await;
                        match t {
                            Ok(t) => ContinuationResult {
                                value: ContinuationResultValue::$Variant(t),
                                tag,
                            },
                            Err(e) => ContinuationResult {
                                value: ContinuationResultValue::Error(e),
                                tag,
                            },
                        }
                    }),
                    trace_buffer: TraceBuffer::empty(),
                })
            }
        }
        impl Deferral<Option<$T>, ()> {
            #[must_use = "coroutine result does nothing unless returned to the entity scheduler"]
            pub fn defer_and_reinvoke(self, tag: u32) -> CoroutineResult {
                CoroutineResult::_DeferredReenterCoroutine(DeferredPrivate {
                    deferred_call: Box::pin(async move {
                        let (t, _) = self.deferred_call.await;
                        match t {
                            Some(t) => ContinuationResult {
                                value: ContinuationResultValue::$Variant(t),
                                tag,
                            },
                            None => ContinuationResult {
                                value: ContinuationResultValue::None,
                                tag,
                            },
                        }
                    }),
                    trace_buffer: TraceBuffer::empty(),
                })
            }
        }
        impl Deferral<$T> {
            #[must_use = "coroutine result does nothing unless returned to the entity scheduler"]
            pub fn defer_and_reinvoke(self, tag: u32) -> CoroutineResult {
                CoroutineResult::_DeferredReenterCoroutine(DeferredPrivate {
                    deferred_call: Box::pin(async move {
                        let (t, _) = self.deferred_call.await;
                        ContinuationResult {
                            value: ContinuationResultValue::$Variant(t),
                            tag,
                        }
                    }),
                    trace_buffer: TraceBuffer::empty(),
                })
            }
        }
    };
}

impl_deferral!(String, String);
impl_deferral!(f64, FloatNumber);
impl_deferral!(u64, Integer);
impl_deferral!(Vector3<f64>, Vector);
impl_deferral!(bool, Boolean);
impl_deferral!(EntityMoveDecision, EntityDecision);
impl_deferral!(HeapResult, HeapResult);

impl Deferral<ContinuationResultValue, ()> {
    pub fn defer_and_reinvoke(self, tag: u32) -> CoroutineResult {
        CoroutineResult::_DeferredReenterCoroutine(DeferredPrivate {
            deferred_call: Box::pin(async move {
                let (value, _) = self.deferred_call.await;
                ContinuationResult { tag, value }
            }),
            trace_buffer: TraceBuffer::empty(),
        })
    }
}

impl<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static> Deferral<T, Residual> {
    pub fn and_then(
        self,
        f: impl FnOnce(T) -> EntityMoveDecision + Send + Sync + 'static,
    ) -> CoroutineResult {
        CoroutineResult::_DeferredMoveResult(DeferredPrivate {
            deferred_call: Box::pin(async move {
                let (t, _residual) = self.deferred_call.await;
                f(t)
            }),
            trace_buffer: TraceBuffer::empty(),
        })
    }
}

impl<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static> From<T>
    for DeferrableResult<T, Residual>
{
    fn from(t: T) -> Self {
        Self::AvailableNow(t)
    }
}

impl<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static> From<Deferral<T, Residual>>
    for DeferrableResult<T, Residual>
{
    fn from(t: Deferral<T, Residual>) -> Self {
        Self::Deferred(t)
    }
}

impl<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static>
    DeferrableResult<T, Residual>
{
    pub fn map<U: Send + Sync + 'static>(
        self,
        f: impl FnOnce(T) -> U + Send + Sync + 'static,
    ) -> DeferrableResult<U, Residual> {
        match self {
            Self::AvailableNow(t) => DeferrableResult::AvailableNow(f(t)),
            Self::Deferred(d) => DeferrableResult::Deferred(d.map(f)),
        }
    }
}

impl<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static> Deferral<T, Residual> {
    pub fn map<U: Send + Sync + 'static>(
        self,
        f: impl FnOnce(T) -> U + Send + Sync + 'static,
    ) -> Deferral<U, Residual> {
        Deferral {
            deferred_call: Box::pin(async move {
                let (t, residual) = self.deferred_call.await;
                (f(t), residual)
            }),
        }
    }
}

pub enum ReenterableResult<T: Send + Sync + 'static> {
    AvailableNow(T),
    Deferred(Deferral<ContinuationResultValue>),
}
impl<T: Send + Sync + 'static> From<T> for ReenterableResult<T> {
    fn from(t: T) -> Self {
        Self::AvailableNow(t)
    }
}

/// Either a T, or a deferred call to get a T
#[must_use = "coroutine result does nothing unless returned to the entity scheduler"]
pub enum DeferrableResult<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static = ()> {
    AvailableNow(T),
    Deferred(Deferral<T, Residual>),
}
impl<T: Send + Sync + 'static, Residual: Debug + Send + Sync + 'static> From<(T, Residual)>
    for DeferrableResult<T, Residual>
{
    fn from(t: (T, Residual)) -> Self {
        Self::AvailableNow(t.0)
    }
}

#[non_exhaustive]
#[derive(Debug)]
/// The value returned to a coroutine from a deferred call after it yields to the scheduler.
///
/// For perfomance, the various return types are broken out into their respective variants.
///
/// The returned variant will match the one documented in the docstring of the deferred call.
pub enum ContinuationResultValue {
    /// A GetBlock deferred call's result
    GetBlock(BlockId, BlockCoordinate),
    // TODO: Other dedicated return types for common map operations
    /// A string result from a deferred call
    String(String),
    /// A number result from a deferred call
    FloatNumber(f64),
    /// A u64 integer result from a deferred call
    Integer(u64),
    /// A vector result from a deferred call
    Vector(Vector3<f64>),
    /// A boolean result from a deferred call
    Boolean(bool),
    /// A movement result from a deferred call
    EntityDecision(EntityMoveDecision),
    /// An anyhow::Error result from a deferred call
    Error(anyhow::Error),
    /// A heap-allocated result from a deferred call
    HeapResult(HeapResult),
    /// None. Don't even reinvoke the coroutine
    None,
}
/// A heap-allocated result from a deferred call
pub type HeapResult = Box<dyn Any + Send + Sync>;

pub struct ContinuationResult {
    /// The tag that was passed when deferring the call
    pub tag: u32,
    /// The result of the deferred call
    pub value: ContinuationResultValue,
}

/// Warning: This trait is not stable and may change in the future. If `coroutine_trait` is stabilized,
/// it may replace this trait. Regardless, it can still evolve quickly.
pub trait EntityCoroutine: Send + Sync + Debug + 'static {
    /// Called when the entity's next move needs to be planned.
    ///
    /// Args:
    ///   * current_position: The current position of the entity
    ///   * whence: The position of the entity at the time when the current move finishes. The returned move will start from approximately that place.
    ///   * when: The time (seconds from now) when the current move finishes. The returned move will start at that time.
    ///   * queue_space: The amount of space available in the move queue
    ///
    /// Returns:
    ///     The next move for the entity
    fn plan_move(
        self: Pin<&mut Self>,
        services: &EntityCoroutineServices<'_>,
        current_position: Vector3<f64>,
        whence: Vector3<f64>,
        when: f32,
        queue_space: usize,
    ) -> CoroutineResult;

    fn continuation(
        self: Pin<&mut Self>,
        services: &EntityCoroutineServices<'_>,
        current_position: Vector3<f64>,
        whence: Vector3<f64>,
        when: f32,
        queue_space: usize,
        continuation_result: ContinuationResult,
        trace_buffer: TraceBuffer,
    ) -> CoroutineResult {
        trace_buffer.log("WARN: In default continuation, dropping the completion.");
        drop(continuation_result);
        self.plan_move(services, current_position, whence, when, queue_space)
    }
}

const ID_MASK: u64 = (1 << 48) - 1;

#[derive(Debug)]
enum MoveQueue {
    SingleMove(Option<StoredMovement>),
    Buffer8(Box<CircularBuffer<8, StoredMovement>>),
    Buffer64(Box<CircularBuffer<64, StoredMovement>>),
}

impl MoveQueue {
    fn pop(&mut self) -> Option<StoredMovement> {
        match self {
            MoveQueue::SingleMove(m) => m.take(),
            MoveQueue::Buffer8(b) => {
                let m = b.pop_front();
                b.make_contiguous();
                m
            }
            MoveQueue::Buffer64(b) => {
                let m = b.pop_front();
                b.make_contiguous();
                m
            }
        }
    }

    fn peek(&self) -> Option<&StoredMovement> {
        match self {
            MoveQueue::SingleMove(m) => m.as_ref(),
            MoveQueue::Buffer8(b) => b.front(),
            MoveQueue::Buffer64(b) => b.front(),
        }
    }

    fn remaining_capacity(&self) -> usize {
        match self {
            MoveQueue::SingleMove(x) => {
                if x.is_some() {
                    0
                } else {
                    1
                }
            }
            MoveQueue::Buffer8(b) => b.capacity() - b.len(),
            MoveQueue::Buffer64(b) => b.capacity() - b.len(),
        }
    }

    fn pending_movements(&self) -> usize {
        match self {
            MoveQueue::SingleMove(x) => x.is_some() as usize,
            MoveQueue::Buffer8(b) => b.len(),
            MoveQueue::Buffer64(b) => b.len(),
        }
    }

    /// Pushes a movement into the queue. If the queue is full, this will panic.
    fn push(&mut self, movement: StoredMovement) {
        match self {
            MoveQueue::SingleMove(x) => {
                assert!(x.is_none());
                *x = Some(movement);
            }
            MoveQueue::Buffer8(b) => {
                b.try_push_back(movement)
                    .map_err(|_| "Buffer overflow")
                    .unwrap();
                b.make_contiguous();
            }
            MoveQueue::Buffer64(b) => {
                b.try_push_back(movement)
                    .map_err(|_| "Buffer overflow")
                    .unwrap();
                b.make_contiguous();
            }
        }
    }

    fn occupancy(&self) -> u64 {
        match self {
            MoveQueue::SingleMove(x) => x.is_some() as u64,
            MoveQueue::Buffer8(b) => b.len() as u64,
            MoveQueue::Buffer64(b) => b.len() as u64,
        }
    }

    fn as_slice(&self) -> &[StoredMovement] {
        match self {
            MoveQueue::SingleMove(x) => x.as_ref().map(std::slice::from_ref).unwrap_or(&[]),
            MoveQueue::Buffer8(x) => {
                let (left_slice, right_slice) = x.as_slices();
                if right_slice.is_empty() {
                    left_slice
                } else {
                    panic!("Buffer8 should be contiguous");
                }
            }
            MoveQueue::Buffer64(x) => {
                let (left_slice, right_slice) = x.as_slices();
                if right_slice.is_empty() {
                    left_slice
                } else {
                    panic!("Buffer64 should be contiguous");
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            MoveQueue::SingleMove(x) => x.is_none(),
            MoveQueue::Buffer8(x) => x.is_empty(),
            MoveQueue::Buffer64(x) => x.is_empty(),
        }
    }
}

/// The core of entity data. For performance reasons, this is stored in struct-of-arrays
/// form, so that SIMD operations can be used to access it.
/// For now, these are just standard vectors, with further SIMD optimizations coming later.
/// The in-memory representation is subject to change and other code should not assume any particular
/// layout.
#[derive(Debug)]
struct EntityCoreArray {
    // The next index to insert into. It may either be an unused slot or at the end of the arrays
    // This is an optimization for allocation speed; for now let's set it aside
    // next_insert: usize,
    // The physical len of the arrays.
    len: usize,

    // The same timebase as game_state
    base_time: Instant,

    // Lookup table from ID to index. This should only be used for scalar operations on a single entity (or a few)
    id_lookup: FxHashMap<u64, usize>,

    // Entity ID, unique, assigned by the server. Only 48 bits available to allow for possible
    // packing later
    id: Vec<u64>,
    // Flags
    control: Vec<u8>,
    // The type of entity this actually is
    class: Vec<u32>,
    // Tracks the last modification that actually conveys new information (e.g. to a client)
    // Simply exhausting a move and starting the next queued move does not count.
    last_nontrivial_modification: Vec<u64>,
    // The current movement is broken out into struct-of-arrays so that SIMD can be used
    // for fast checks.
    //
    // The current move's sequence number.
    current_move_seq: Vec<u64>,
    // Position at the start of the current move
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    // Velocity at the start of the current move
    xv: Vec<f32>,
    yv: Vec<f32>,
    zv: Vec<f32>,
    // Acceleration throughout the current move
    xa: Vec<f32>,
    ya: Vec<f32>,
    za: Vec<f32>,
    // Face direction, radians, during the current move
    theta_y: Vec<f32>,
    // Pitch during the current move
    theta_x: Vec<f32>,
    // The total time for the current move
    move_time: Vec<f32>,
    // The time already finished in the current move
    move_start_tick: Vec<u64>,
    // When we next need to recalculate the current move
    recalc_at: Vec<u64>,

    move_queue: Vec<MoveQueue>,

    // TODO: Once thin_box is stabilized, we can use it here
    coroutine: Vec<Option<Pin<Box<dyn EntityCoroutine>>>>,
    // This is an annoying indirect pointer, but we probably won't use it for the hot path
    trailing_entities: Vec<Option<TrailingEntities>>,
}
impl EntityCoreArray {
    /// Checks some preconditions that we want to assume on the vectors to make
    /// iteration faster.
    fn check_preconditions(&self) {
        // We assume that all vectors have the same length.
        assert_eq!(self.id.len(), self.len);
        assert_eq!(self.class.len(), self.len);
        assert_eq!(self.control.len(), self.len);
        assert_eq!(self.last_nontrivial_modification.len(), self.len);
        assert_eq!(self.current_move_seq.len(), self.len);
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
        assert_eq!(self.theta_x.len(), self.len);
        assert_eq!(self.move_start_tick.len(), self.len);
        assert_eq!(self.recalc_at.len(), self.len);
        assert_eq!(self.move_queue.len(), self.len);
        assert_eq!(self.coroutine.len(), self.len);
        assert_eq!(self.trailing_entities.len(), self.len);
    }

    // Update the times for all entities
    //
    // Args:
    //   * update_tick: The tick to use for the update
    //
    // Returns the tick when the next event should occur
    fn update_times(&mut self, update_tick: u64) -> u64 {
        let _span = span!("entity_update_times");
        self.check_preconditions();

        let mut next_event = u64::MAX;

        for i in 0..self.len {
            let next_event_i = self.update_time_single(i, update_tick);
            if next_event_i < next_event {
                next_event = next_event_i;
            }
        }
        const MAX_WAIT_TIME_TICKS: u64 = 10 * 1_000_000_000;
        let awakening_bound = update_tick + MAX_WAIT_TIME_TICKS;

        next_event.min(awakening_bound).max(update_tick)
    }

    /// Run coroutines present on entities that need coroutine based recalcs.
    ///
    /// Args:
    ///   * services: The coroutine services
    ///   * delta_time: The amount of time that has passed since the last poll cycle
    ///   * sender: The channel for receiving completions from deferred tasks
    ///
    /// Returns: The next tick at which we should awaken
    ///
    /// TODO: more and better scheduling/prioritization algorithms
    fn run_coroutines(
        &mut self,
        services: &EntityCoroutineServices<'_>,
        game_state: &GameState,
        sender: &tokio::sync::mpsc::Sender<Completion<ContinuationResult>>,
    ) -> u64 {
        let _span = span!("entity_run_coroutines");
        self.check_preconditions();
        let mut next_event = u64::MAX;
        for i in 0..self.len {
            next_event =
                next_event.min(self.run_coro_single(i, services, None, sender, game_state.tick()));
        }

        next_event
    }

    fn reset_movement(
        &mut self,
        index: usize,
        pos: Vector3<f64>,
        initial_movement: Movement,
        queued_movements: InitialMoveQueue,
        tick: u64,
    ) {
        self.last_nontrivial_modification[index] = tick;
        self.x[index] = pos.x;
        self.y[index] = pos.y;
        self.z[index] = pos.z;
        self.xv[index] = initial_movement.velocity.x;
        self.yv[index] = initial_movement.velocity.y;
        self.zv[index] = initial_movement.velocity.z;
        self.xa[index] = initial_movement.acceleration.x;
        self.ya[index] = initial_movement.acceleration.y;
        self.za[index] = initial_movement.acceleration.z;
        self.theta_y[index] = initial_movement.face_direction;
        self.theta_x[index] = initial_movement.pitch;
        self.move_time[index] = initial_movement.move_time;
        self.move_start_tick[index] = tick;
        self.recalc_at[index] = if !queued_movements.has_slack() {
            // todo test this logic
            u64::MAX
        } else if self.control[index] & CONTROL_AUTONOMOUS == 0 {
            u64::MAX
        } else {
            tick
        };
        self.current_move_seq[index] =
            self.current_move_seq[index] + LARGEST_POSSIBLE_QUEUE_SIZE as u64 + 1;
        match queued_movements {
            InitialMoveQueue::SingleMove(movement) => {
                self.move_queue[index] = MoveQueue::SingleMove(movement.map(|m| StoredMovement {
                    sequence: self.current_move_seq[index] + LARGEST_POSSIBLE_QUEUE_SIZE as u64 + 2,
                    movement: m,
                    start_tick: tick,
                }));
            }
            InitialMoveQueue::Buffer8(movements) => {
                let mut current_tick = tick;
                let mut buffer: CircularBuffer<8, StoredMovement> = movements
                    .into_iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let movement = StoredMovement {
                            sequence: self.current_move_seq[i]
                                + LARGEST_POSSIBLE_QUEUE_SIZE as u64
                                + 2
                                + i as u64,
                            movement: m,
                            start_tick: current_tick,
                        };
                        current_tick += (m.move_time * 1_000_000_000.0) as u64;
                        movement
                    })
                    .collect();
                buffer.make_contiguous();
                self.move_queue[index] = MoveQueue::Buffer8(Box::new(buffer));
            }
            InitialMoveQueue::Buffer64(movements) => {
                let mut current_tick = tick;
                let mut buffer: CircularBuffer<64, StoredMovement> = movements
                    .into_iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let movement = StoredMovement {
                            sequence: self.current_move_seq[i]
                                + LARGEST_POSSIBLE_QUEUE_SIZE as u64
                                + 2
                                + i as u64,
                            movement: m,
                            start_tick: current_tick,
                        };
                        current_tick += (m.move_time * 1_000_000_000.0) as u64;
                        movement
                    })
                    .collect();
                buffer.make_contiguous();
                self.move_queue[index] = MoveQueue::Buffer64(Box::new(buffer));
            }
        }
    }

    fn new(base_time: Instant) -> EntityCoreArray {
        EntityCoreArray {
            len: 0,
            id_lookup: FxHashMap::default(),
            id: vec![],
            control: vec![],
            class: vec![],
            last_nontrivial_modification: vec![],
            current_move_seq: vec![],
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
            theta_x: vec![],
            move_time: vec![],
            move_start_tick: vec![],
            recalc_at: vec![],
            move_queue: vec![],
            coroutine: vec![],
            trailing_entities: vec![],
            base_time,
        }
    }

    fn insert(
        &mut self,
        id: u64,
        entity_class: EntityTypeId,
        entity_def: &EntityDef,
        position: Vector3<f64>,
        coroutine: Option<Pin<Box<dyn EntityCoroutine>>>,
        trailing_entities: Option<TrailingEntities>,
        services: &EntityCoroutineServices<'_>,
        completion_sender: &tokio::sync::mpsc::Sender<Completion<ContinuationResult>>,
        tick: u64,
    ) -> usize {
        // scan for the first unused slot
        let index = self
            .control
            .iter()
            .enumerate()
            .find(|(_, &control)| control & CONTROL_PRESENT == 0)
            .map(|(i, _)| i);
        let control = if coroutine.is_some() {
            CONTROL_PRESENT | CONTROL_AUTONOMOUS
        } else {
            CONTROL_PRESENT
        };
        let queue = match entity_def.move_queue_type {
            MoveQueueType::SingleMove => MoveQueue::SingleMove(None),
            MoveQueueType::Buffer8 => MoveQueue::Buffer8(CircularBuffer::boxed()),
            MoveQueueType::Buffer64 => MoveQueue::Buffer64(CircularBuffer::boxed()),
        };
        let i = match index {
            Some(i) => {
                tracing::debug!("Inserting {} at index {} in entity array", id, i);
                self.id_lookup.insert(id, i);
                self.id[i] = id;
                self.control[i] = control;
                self.class[i] = entity_class.class.0;
                // TODO use the data buffer of the entity class as well
                self.last_nontrivial_modification[i] = tick;
                self.current_move_seq[i] = 1;
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
                self.theta_x[i] = 0.0;
                self.move_time[i] = 0.0;
                self.move_start_tick[i] = 0;
                self.recalc_at[i] = tick;
                self.move_queue[i] = queue;
                self.coroutine[i] = coroutine;
                self.trailing_entities[i] = trailing_entities;
                i
            }
            None => {
                tracing::debug!("Inserting {} at end of entity array", id);
                self.id_lookup.insert(id, self.len);
                // we need to grow the array
                self.len += 1;
                self.id.push(id);
                self.class.push(entity_class.class.0);
                self.control.push(control);
                self.last_nontrivial_modification.push(tick);
                self.current_move_seq.push(1);
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
                self.theta_x.push(0.0);
                self.move_time.push(0.0);
                self.move_start_tick.push(0);
                self.recalc_at.push(tick);
                self.move_queue.push(queue);
                self.coroutine.push(coroutine);
                self.trailing_entities.push(trailing_entities);

                self.len - 1
            }
        };

        if let Some(coroutine) = &mut self.coroutine[i] {
            println!("insert");
            let handler_result = match run_handler!(
                || Ok(coroutine.as_mut().plan_move(
                    services,
                    position,
                    position,
                    0.0,
                    self.move_queue[i].remaining_capacity(),
                )),
                "initial_coroutine",
                &EventInitiator::Engine
            ) {
                Ok(handler_result) => handler_result,
                Err(e) => {
                    self.coro_panic_unwind(i, e);
                    return i;
                }
            };

            match handler_result {
                // This is the initial move, so it starts now
                CoroutineResult::Successful(EntityMoveDecision::QueueSingleMovement(movement)) => {
                    self.last_nontrivial_modification[i] = tick;
                    self.xv[i] = movement.velocity.x;
                    self.yv[i] = movement.velocity.y;
                    self.zv[i] = movement.velocity.z;
                    self.xa[i] = movement.acceleration.x;
                    self.ya[i] = movement.acceleration.y;
                    self.za[i] = movement.acceleration.z;
                    self.theta_y[i] = movement.face_direction;
                    self.theta_x[i] = movement.pitch;
                    self.move_time[i] = movement.move_time;
                    self.move_start_tick[i] = tick;

                    // We only got one move. We need another.
                    self.control[i] |= CONTROL_AUTONOMOUS_NEEDS_CALC;
                    self.recalc_at[i] = u64::MAX;
                    tracing::debug!("{}: queued movement (initial)", self.id[i]);
                }
                CoroutineResult::Successful(EntityMoveDecision::QueueUpMultiple(movements)) => {
                    self.last_nontrivial_modification[i] = tick;
                    if movements.is_empty() {
                        panic!("QueueUpMultiple with no movements");
                    }
                    let first = &movements[0];
                    self.xv[i] = first.velocity.x;
                    self.yv[i] = first.velocity.y;
                    self.zv[i] = first.velocity.z;
                    self.xa[i] = first.acceleration.x;
                    self.ya[i] = first.acceleration.y;
                    self.za[i] = first.acceleration.z;
                    self.theta_y[i] = first.face_direction;
                    self.theta_x[i] = first.pitch;
                    self.move_time[i] = first.move_time;
                    self.move_start_tick[i] = tick;
                    let mut start_tick = tick + (first.move_time * 1_000_000_000.0) as u64;
                    for (seq, m) in movements.into_iter().enumerate().skip(1) {
                        tracing::debug!("{}: queued movement {}", self.id[i], seq);
                        self.move_queue[i].push(StoredMovement {
                            // Current is 1 so next is 2
                            sequence: seq as u64 + 1,
                            start_tick,
                            movement: m,
                        });
                        start_tick += (m.move_time * 1_000_000_000.0) as u64;
                    }
                    if self.move_queue[i].remaining_capacity() > 0 {
                        // We need to recalc
                        self.control[i] |= CONTROL_AUTONOMOUS_NEEDS_CALC;
                        self.recalc_at[i] = u64::MAX;
                    }
                }
                CoroutineResult::Successful(EntityMoveDecision::AskAgainLater(delay)) => {
                    self.recalc_at[i] = (delay.max(0.0) * 1_000_000_000.0) as u64;
                    self.move_time[i] = delay;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    self.control[i] |= CONTROL_DEQUEUE_WHEN_READY;
                }
                CoroutineResult::Successful(EntityMoveDecision::AskAgainLaterFlexible(delay)) => {
                    // Flexible timings are an optimization, not a requirement
                    // Because initializing an entity is complex, simply treat this as a non-range ask-again.
                    self.recalc_at[i] = (delay.start.max(0.0) * 1_000_000_000.0) as u64;
                    self.move_time[i] = delay.start;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    self.control[i] |= CONTROL_DEQUEUE_WHEN_READY;
                }
                CoroutineResult::Successful(EntityMoveDecision::ImmediateDespawn) => {
                    self.last_nontrivial_modification[i] = tick;
                    self.control[i] = 0;
                }
                CoroutineResult::Successful(EntityMoveDecision::ResetMovement(pos, m1, m2)) => {
                    self.reset_movement(i, pos, m1, m2, tick);
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                }
                CoroutineResult::Successful(EntityMoveDecision::StopCoroutineControl) => {
                    self.control[i] &= !(CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS);
                    self.control[i] |= CONTROL_DEQUEUE_WHEN_READY;
                }
                CoroutineResult::_DeferredMoveResult(deferral) => {
                    self.control[i] |= CONTROL_SUSPENDED;
                    self.control[i] |= CONTROL_DEQUEUE_WHEN_READY;
                    let tx_clone = completion_sender.clone();
                    let id = self.id[i];

                    let trace_buffer = deferral.trace_buffer;
                    tracing::debug!("{}: deferring continuation with a result", id);
                    trace_buffer.log("In insert, preparing to spawn for a deferred result");

                    tokio::task::spawn(async move {
                        trace_buffer.log("spawned, about to run deferred call");

                        let result = deferral.deferred_call.await;
                        trace_buffer.log("Done running deferred call");

                        tx_clone
                            .send(Completion {
                                entity_id: id,
                                index: i,
                                result: ContinuationResult {
                                    value: ContinuationResultValue::EntityDecision(result),
                                    // The tag doesn't matter in this case
                                    tag: 0,
                                },
                                trace_buffer,
                            })
                            .await
                            .unwrap();
                    });
                }
                CoroutineResult::_DeferredReenterCoroutine(deferral) => {
                    self.control[i] |= CONTROL_SUSPENDED;
                    self.control[i] |= CONTROL_DEQUEUE_WHEN_READY;
                    let tx_clone = completion_sender.clone();
                    let id = self.id[i];

                    tracing::debug!("{}: deferring continuation to reenter", id);
                    let trace_buffer = deferral.trace_buffer;
                    trace_buffer.log("In insert, preparing to spawn for a deferred reentry");

                    tokio::task::spawn(async move {
                        trace_buffer.log("spawned, about to run deferred call");
                        let result = deferral.deferred_call.await;
                        trace_buffer.log("Done running deferred call");
                        tx_clone
                            .send(Completion {
                                entity_id: id,
                                index: i,
                                result,
                                trace_buffer,
                            })
                            .await
                            .unwrap();
                    });
                }
            }
        }

        i
    }

    fn remove(&mut self, id: u64) -> Result<(), EntityError> {
        if let Some(i) = self.id_lookup.remove(&id) {
            self.len -= 1;
            self.id.swap_remove(i);
            self.class.swap_remove(i);
            self.control.swap_remove(i);
            self.last_nontrivial_modification.swap_remove(i);
            self.current_move_seq.swap_remove(i);
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
            self.theta_x.swap_remove(i);
            self.move_time.swap_remove(i);
            self.move_start_tick.swap_remove(i);
            self.recalc_at.swap_remove(i);
            self.move_queue.swap_remove(i);
            self.coroutine.swap_remove(i);
            self.trailing_entities.swap_remove(i);

            // fix up the id lookup since we rearranged stuff
            if i != self.len {
                self.id_lookup.insert(self.id[i], i);
            }
            // For debugging only
            // Note that other check_preconditions() calls actually help guide the optimizer
            self.check_preconditions();
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
        next: InitialMoveQueue,
        tick: u64,
    ) -> Result<usize, EntityError> {
        let index = self.find_entity(id)?;
        self.reset_movement(index, position, current, next, tick);
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
        since: Option<u64>,
        mut f: impl FnMut(IterEntity) -> Result<()>,
        tick: u64,
    ) -> Result<()> {
        for i in 0..self.len {
            if self.control[i] & CONTROL_PRESENT == 0 {
                continue;
            }
            let lnm = self.last_nontrivial_modification[i];
            if since.is_some_and(|x| x > lnm) {
                continue;
            }

            let elapsed = self.move_elapsed(i, tick);
            let entity = IterEntity {
                id: self.id[i],
                class: self.class[i],
                starting_position: Vector3::new(self.x[i], self.y[i], self.z[i]),
                instantaneous_position: Vector3::new(
                    qproj(self.x[i], self.xv[i], self.xa[i], elapsed),
                    qproj(self.y[i], self.yv[i], self.ya[i], elapsed),
                    qproj(self.z[i], self.zv[i], self.za[i], elapsed),
                ),
                current_move: StoredMovement {
                    movement: Movement {
                        velocity: Vector3::new(self.xv[i], self.yv[i], self.zv[i]),
                        acceleration: Vector3::new(self.xa[i], self.ya[i], self.za[i]),
                        face_direction: self.theta_y[i],
                        pitch: self.theta_x[i],
                        move_time: self.move_time[i],
                    },
                    start_tick: self.move_start_tick[i],
                    sequence: self.current_move_seq[i],
                },
                next_moves: self.move_queue[i].as_slice(),
                last_nontrivial_modification: lnm,
                trailing_entities: self.trailing_entities[i].as_deref(),
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
        completion: Option<Completion<ContinuationResult>>,
        completion_sender: &tokio::sync::mpsc::Sender<Completion<ContinuationResult>>,
        tick: u64,
    ) -> u64 {
        if self.control[i] & CONTROL_PRESENT == 0 {
            return u64::MAX;
        }
        if self.control[i] & CONTROL_AUTONOMOUS == 0 {
            return u64::MAX;
        }
        if self.control[i] & CONTROL_SUSPENDED != 0 {
            return u64::MAX;
        }
        if self.recalc_at[i] <= tick
            && self.control[i] & (CONTROL_AUTONOMOUS | CONTROL_PRESENT)
                == (CONTROL_AUTONOMOUS | CONTROL_PRESENT)
        {
            self.control[i] |= CONTROL_AUTONOMOUS_NEEDS_CALC;
        }

        if self.control[i] & (CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS | CONTROL_PRESENT)
            != (CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS | CONTROL_PRESENT)
        {
            return u64::MAX;
        }

        let mut last_move_ending_tick = if self.control[i] & CONTROL_DEQUEUE_WHEN_READY != 0 {
            tick
        } else {
            self.move_start_tick[i].saturating_add(self.move_time_in_ticks(i))
        };

        if let Some(coroutine) = self.coroutine[i].as_mut() {
            // do some kinematics
            let mut estimated_x = qproj(self.x[i], self.xv[i], self.xa[i], self.move_time[i]);
            let mut estimated_y = qproj(self.y[i], self.yv[i], self.ya[i], self.move_time[i]);
            let mut estimated_z = qproj(self.z[i], self.zv[i], self.za[i], self.move_time[i]);
            let mut last_seq = self.current_move_seq[i];

            tracing::debug!(
                "running coro. control: {}, last_seq: {}. Estimated position {}, {}, {}. \
             Move time {}, move start tick {}, move end tick {}",
                self.control[i],
                last_seq,
                estimated_x,
                estimated_y,
                estimated_z,
                self.move_time[i],
                self.move_start_tick[i],
                last_move_ending_tick
            );
            tracing::debug!("queue: {:?}", self.move_queue[i]);
            for StoredMovement {
                sequence,
                movement,
                start_tick,
            } in self.move_queue[i].as_slice()
            {
                tracing::debug!("> seq {}", sequence);
                if *start_tick != last_move_ending_tick {
                    tracing::warn!(
                        "> seq {} wrong start tick {}, expected {}",
                        sequence,
                        start_tick,
                        last_move_ending_tick
                    );
                }
                assert!(last_seq < *sequence);
                last_seq = *sequence;
                estimated_x = qproj(
                    estimated_x,
                    movement.velocity.x,
                    movement.acceleration.x,
                    movement.move_time,
                );
                estimated_y = qproj(
                    estimated_y,
                    movement.velocity.y,
                    movement.acceleration.y,
                    movement.move_time,
                );
                estimated_z = qproj(
                    estimated_z,
                    movement.velocity.z,
                    movement.acceleration.z,
                    movement.move_time,
                );
                last_move_ending_tick = last_move_ending_tick
                    .saturating_add((movement.move_time.max(0.0) * 1_000_000_000.0) as u64);
            }

            if self.control[i] & CONTROL_DEQUEUE_WHEN_READY != 0 {
                last_move_ending_tick = tick;
            }

            let estimated_t = (last_move_ending_tick - tick) as f32 / 1_000_000_000.0;

            tracing::debug!(
                "After applying queued moves: Estimated position {}, {}, {} @ {} ({})",
                estimated_x,
                estimated_y,
                estimated_z,
                last_move_ending_tick,
                estimated_t
            );
            let move_queue = &mut self.move_queue[i];

            let planned_move = match completion {
                None => run_handler!(
                    || Ok(coroutine.as_mut().plan_move(
                        services,
                        vec3(self.x[i], self.y[i], self.z[i]),
                        vec3(estimated_x, estimated_y, estimated_z),
                        estimated_t,
                        move_queue.remaining_capacity(),
                    )),
                    "entity_coro",
                    &EventInitiator::Engine
                ),
                Some(c) => match c.result.value {
                    ContinuationResultValue::EntityDecision(m) => {
                        Ok(CoroutineResult::Successful(m))
                    }
                    _ => {
                        c.trace_buffer.log("In run_coro_single, reentering");
                        run_handler!(
                            || Ok(coroutine.as_mut().continuation(
                                services,
                                vec3(self.x[i], self.y[i], self.z[i]),
                                vec3(estimated_x, estimated_y, estimated_z),
                                estimated_t,
                                move_queue.remaining_capacity(),
                                c.result,
                                c.trace_buffer,
                            )),
                            "entity_coro",
                            &EventInitiator::Engine
                        )
                    }
                },
            };

            let planned_move = match planned_move {
                Ok(planned_move) => planned_move,
                Err(e) => {
                    self.coro_panic_unwind(i, e);
                    return u64::MAX;
                }
            };

            planned_move.log_trace("In run_coro_single, after plan_move");

            match planned_move {
                CoroutineResult::Successful(EntityMoveDecision::QueueSingleMovement(movement)) => {
                    tracing::debug!("planned single move: {:?}", movement);
                    self.last_nontrivial_modification[i] = tick;
                    self.move_queue[i].push(StoredMovement {
                        movement,
                        sequence: last_seq + 1,
                        start_tick: last_move_ending_tick,
                    });
                    if self.move_queue[i].remaining_capacity() == 0 {
                        self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    }
                    // We'll trigger a recalc when the movement gets advanced; no need to use recalc_in
                    self.recalc_at[i] = u64::MAX;
                    tracing::debug!("{}: queued movement.", self.id[i]);
                    self.current_move_ending_tick(i)
                }
                CoroutineResult::Successful(EntityMoveDecision::QueueUpMultiple(movements)) => {
                    tracing::debug!("planned multiple move: {:?}", movements);
                    self.last_nontrivial_modification[i] = tick;
                    if movements.is_empty() {
                        // TODO this shouldn't be a panic
                        panic!("QueueUpMultiple with no movements");
                    }
                    for (movement_index, m) in movements.into_iter().enumerate() {
                        self.move_queue[i].push(StoredMovement {
                            sequence: last_seq + movement_index as u64 + 1,
                            movement: m,
                            start_tick: last_move_ending_tick,
                        });
                        last_move_ending_tick += (m.move_time * 1_000_000_000.0) as u64
                    }
                    self.recalc_at[i] = u64::MAX;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;

                    tracing::debug!(
                        "mt: {}, mst: {}, control: {}",
                        self.move_time[i],
                        self.move_start_tick[i],
                        self.control[i]
                    );
                    if self.control[i] & CONTROL_DEQUEUE_WHEN_READY != 0 {
                        tick
                    } else {
                        self.current_move_ending_tick(i)
                    }
                }
                CoroutineResult::Successful(EntityMoveDecision::AskAgainLater(delay)) => {
                    tracing::debug!("delay: {:?}", delay);
                    self.recalc_at[i] = tick + (delay * 1_000_000_000.0) as u64;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    self.current_move_ending_tick(i).min(self.recalc_at[i])
                }
                CoroutineResult::Successful(EntityMoveDecision::AskAgainLaterFlexible(delay)) => {
                    tracing::debug!("delay: {:?}", delay);
                    assert!(delay.end >= delay.start);
                    // Accept a recalc if it's more than at least recalc_in
                    self.recalc_at[i] = tick + (delay.start * 1_000_000_000.0) as u64;
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    // Schedule at the latest allowed time
                    self.current_move_ending_tick(i)
                        .min(tick + (delay.end * 1_000_000_000.0) as u64)
                }
                CoroutineResult::Successful(EntityMoveDecision::ImmediateDespawn) => {
                    self.last_nontrivial_modification[i] = tick;
                    self.control[i] = 0;
                    u64::MAX
                }
                CoroutineResult::Successful(EntityMoveDecision::ResetMovement(pos, m1, m2)) => {
                    self.reset_movement(i, pos, m1, m2, tick);
                    self.control[i] &= !CONTROL_AUTONOMOUS_NEEDS_CALC;
                    self.current_move_ending_tick(i)
                }
                CoroutineResult::Successful(EntityMoveDecision::StopCoroutineControl) => {
                    self.control[i] &= !(CONTROL_AUTONOMOUS_NEEDS_CALC | CONTROL_AUTONOMOUS);
                    self.current_move_ending_tick(i)
                }
                CoroutineResult::_DeferredMoveResult(deferral) => {
                    deferral.trace_buffer.log("_dmr early");
                    self.control[i] |= CONTROL_SUSPENDED;
                    let tx_clone = completion_sender.clone();
                    let id = self.id[i];

                    tracing::debug!("{}: deferring continuation with a result", id);

                    let trace_buffer = deferral.trace_buffer;
                    trace_buffer.log("In run_coro, preparing to spawn for a deferred result");

                    tokio::task::spawn(async move {
                        trace_buffer.log("spawned, about to run deferred call");
                        let result = deferral.deferred_call.await;
                        trace_buffer.log("Done running deferred call");
                        tx_clone
                            .send(Completion {
                                entity_id: id,
                                index: i,
                                result: ContinuationResult {
                                    value: ContinuationResultValue::EntityDecision(result),
                                    // The tag doesn't matter in this case
                                    tag: 0,
                                },
                                trace_buffer,
                            })
                            .await
                            .unwrap();

                        tracing::debug!("{}: sending continuation with a result", id);
                    });
                    // Don't reawaken until we get the completion
                    // TODO: completion stuckness detection
                    u64::MAX
                }
                CoroutineResult::_DeferredReenterCoroutine(deferral) => {
                    self.control[i] |= CONTROL_SUSPENDED;
                    let tx_clone = completion_sender.clone();
                    let id = self.id[i];

                    let trace_buffer = deferral.trace_buffer;
                    trace_buffer.log("In run_coro, preparing to spawn for a deferred result");
                    tracing::debug!("{}: deferring continuation to reenter", id);

                    tokio::task::spawn(async move {
                        trace_buffer.log("spawned, about to run deferred call");
                        let result = deferral.deferred_call.await;
                        trace_buffer.log("Done running deferred call");
                        tx_clone
                            .send(Completion {
                                entity_id: id,
                                index: i,
                                result,
                                trace_buffer,
                            })
                            .await
                            .unwrap();
                        tracing::debug!("{}: sending continuation to reenter", id);
                    });
                    // Don't reawaken until we get the completion
                    // TODO: completion stuckness detection
                    u64::MAX
                }
            }
        } else {
            u64::MAX
        }
    }

    fn current_move_ending_tick(&mut self, i: usize) -> u64 {
        self.move_start_tick[i].saturating_add(self.move_time_in_ticks(i))
    }

    // We *really* want this to be inlined into the loop
    #[inline(always)]
    /// Update the time elapsed for an entity
    /// Returns the time when the move will end and a new calculation is needed (as a tick)
    fn update_time_single(&mut self, i: usize, update_tick: u64) -> u64 {
        // We don't update recalc_in, the coroutine handler will do that

        let force_advance =
            self.control[i] & CONTROL_DEQUEUE_WHEN_READY != 0 && !self.move_queue[i].is_empty();
        if force_advance {
            tracing::debug!(
                "{}: force advance, control {}, mst {}, mt {}, MoveQ available {}",
                self.id[i],
                self.control[i],
                self.move_start_tick[i],
                self.move_time[i],
                self.move_queue[i].pending_movements()
            );
            self.last_nontrivial_modification[i] = update_tick;
        };

        if force_advance || self.move_elapsed(i, update_tick) > self.move_time[i] {
            tracing::debug!(
                "{}: finished move, control {}, mst {}, mt {}, MoveQ available {}",
                self.id[i],
                self.control[i],
                self.move_start_tick[i],
                self.move_time[i],
                self.move_queue[i].pending_movements()
            );

            self.control[i] &= !CONTROL_DEQUEUE_WHEN_READY;
            tracing::debug!(
                "{}: finished move w/ time {}/{}, seq was {}",
                self.id[i],
                self.move_elapsed(i, update_tick),
                self.move_time[i],
                self.current_move_seq[i]
            );
            // Then we're going to need a new move to be queued up soon
            if self.control[i] & CONTROL_AUTONOMOUS != 0 {
                self.control[i] |= CONTROL_AUTONOMOUS_NEEDS_CALC;
            }
            // Apply the effect of the current move we just finished
            self.x[i] = qproj(self.x[i], self.xv[i], self.xa[i], self.move_time[i]);
            self.y[i] = qproj(self.y[i], self.yv[i], self.ya[i], self.move_time[i]);
            self.z[i] = qproj(self.z[i], self.zv[i], self.za[i], self.move_time[i]);
            // And move the next move into the current one
            let next_move = self.move_queue[i].pop().unwrap_or_else(|| {
                tracing::debug!("buffer exhausted");
                self.control[i] |= CONTROL_DEQUEUE_WHEN_READY;

                let new_start_tick = self.move_start_tick[i] + self.move_time_in_ticks(i);

                let move_time = if self.recalc_at[i] == u64::MAX {
                    f32::MAX
                } else {
                    let diff = (self.recalc_at[i] - update_tick) as f32;
                    diff.max(0.0) / 1_000_000_000.0
                };

                StoredMovement {
                    movement: Movement::stop_and_stay(self.theta_y[i], self.theta_x[i], move_time),
                    sequence: self.current_move_seq[i] + 1,
                    start_tick: new_start_tick,
                }
            });
            self.xv[i] = next_move.movement.velocity.x;
            self.yv[i] = next_move.movement.velocity.y;
            self.zv[i] = next_move.movement.velocity.z;
            self.xa[i] = next_move.movement.acceleration.x;
            self.ya[i] = next_move.movement.acceleration.y;
            self.za[i] = next_move.movement.acceleration.z;
            self.theta_y[i] = next_move.movement.face_direction;
            self.theta_x[i] = next_move.movement.pitch;
            self.current_move_seq[i] = next_move.sequence;
            tracing::debug!("new CMS {}", next_move.sequence);
            self.move_start_tick[i] = next_move.start_tick;
            self.move_time[i] = next_move.movement.move_time;

            self.recalc_at[i] = u64::MAX;
        }
        // We now have updated the move (if necessary); here's how long is left in it.
        let kinematic_event_time = self.current_move_ending_tick(i);
        if self.control[i] & (CONTROL_AUTONOMOUS_NEEDS_CALC) != 0 {
            kinematic_event_time.min(self.recalc_at[i])
        } else {
            kinematic_event_time
        }
    }

    /// Returns the elapsed time for the current entity, saturating to 0 if it's negative
    fn move_elapsed(&self, i: usize, tick: u64) -> f32 {
        // We need a saturating subtraction here because we don't want to go negative
        tick.saturating_sub(self.move_start_tick[i]) as f32 / 1_000_000_000.0
    }

    fn move_time_in_ticks(&self, i: usize) -> u64 {
        if self.move_time[i].is_infinite() {
            return u64::MAX;
        }
        let approx_ticks = self.move_time[i].max(0.0) * 1_000_000_000.0;
        if approx_ticks >= u64::MAX as f32 {
            return u64::MAX;
        }
        approx_ticks as u64
    }

    #[cold]
    fn coro_panic_unwind(&mut self, index: usize, err: Error) {
        tracing::error!("coroutine {} panicked: {}", index, err);
        tracing::error!("control: {:x}", self.control[index]);
        tracing::error!("class: {}", self.class[index]);
        tracing::error!(
            "last nontrivial modification: {:?}",
            self.last_nontrivial_modification[index]
        );
        tracing::error!("current move seq: {}", self.current_move_seq[index]);
        tracing::error!(
            "coords {:?} {:?} {:?}",
            self.x[index],
            self.y[index],
            self.z[index]
        );
        tracing::error!(
            "velocities {:?} {:?} {:?}",
            self.xv[index],
            self.yv[index],
            self.zv[index]
        );
        tracing::error!(
            "accelerations {:?} {:?} {:?}",
            self.xa[index],
            self.ya[index],
            self.za[index]
        );
        tracing::error!(
            "theta_y {:?} theta_x {:?}",
            self.theta_y[index],
            self.theta_x[index]
        );
        tracing::error!("move time {:?}", self.move_time[index]);
        tracing::error!("move start tick {:?}", self.move_start_tick[index]);
        tracing::error!("recalc at {:?}", self.recalc_at[index]);
        tracing::error!("move queue: {:?}", self.move_queue[index]);
        self.control[index] &= !CONTROL_AUTONOMOUS;
    }
}
/// Project a coordinate into the future.
#[inline]
fn qproj(s: f64, v: f32, a: f32, t: f32) -> f64 {
    s + (v * t + 0.5 * a * t * t) as f64
}

enum EntityAction {
    Insert(
        u64,
        Vector3<f64>,
        Option<Pin<Box<dyn EntityCoroutine>>>,
        EntityTypeId,
        Option<TrailingEntities>,
    ),
    Remove(u64),
    SetKinematics(u64, Vector3<f64>, Movement, InitialMoveQueue),
}

/// The entities stored in a shard of the game map.
pub struct EntityShard {
    shard_id: usize,
    db: Arc<dyn GameDatabase>,
    core: RwLock<EntityCoreArray>,
    pending_actions_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<EntityAction>>,
    pending_actions_tx: tokio::sync::mpsc::Sender<EntityAction>,
    completion_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Completion<ContinuationResult>>>,
    completion_tx: tokio::sync::mpsc::Sender<Completion<ContinuationResult>>,
}
impl EntityShard {
    pub(super) fn new(
        shard_id: usize,
        db: Arc<dyn GameDatabase>,
        base_time: Instant,
    ) -> EntityShard {
        tracing::warn!("FIXME: Need to load entities from DB for {shard_id}");
        let core = RwLock::new(EntityCoreArray::new(base_time));
        const ENTITY_QUEUE_SIZE: usize = 64;
        const COMPLETION_QUEUE_SIZE: usize = 256;
        let (pending_actions_tx, pending_actions_rx) =
            tokio::sync::mpsc::channel(ENTITY_QUEUE_SIZE);
        let (completion_tx, completion_rx) = tokio::sync::mpsc::channel(COMPLETION_QUEUE_SIZE);
        EntityShard {
            shard_id,
            db,
            core,
            pending_actions_rx: tokio::sync::Mutex::new(pending_actions_rx),
            pending_actions_tx,
            completion_rx: tokio::sync::Mutex::new(completion_rx),
            completion_tx,
        }
    }

    pub(crate) fn for_each_entity(
        &self,
        since: Option<u64>,
        tick: u64,
        f: impl FnMut(IterEntity) -> Result<()>,
    ) -> Result<()> {
        self.core.read().for_each_entity(since, f, tick)
    }
}

pub(crate) struct IterEntity<'a> {
    pub(crate) id: u64,
    pub(crate) class: u32,
    pub(crate) starting_position: Vector3<f64>,
    pub(crate) instantaneous_position: Vector3<f64>,
    pub(crate) current_move: StoredMovement,
    pub(crate) next_moves: &'a [StoredMovement],
    pub(crate) last_nontrivial_modification: u64,
    pub(crate) trailing_entities: Option<&'a [TrailingEntity]>,
}

pub(crate) struct EntityShardWorker {
    entities: Arc<EntityManager>,
    shard_id: usize,
    game_state: Arc<GameState>,
    cancellation: CancellationToken,
}

const COMMAND_BATCH_SIZE: usize = 16;
const COMPLETION_BATCH_SIZE: usize = 32;
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
        let mut rx_lock = self.entities.shards[self.shard_id]
            .pending_actions_rx
            .lock()
            .await;

        let mut completion_lock = self.entities.shards[self.shard_id]
            .completion_rx
            .lock()
            .await;
        let completion_tx = &self.entities.shards[self.shard_id].completion_tx;

        let mut rx_messages = Vec::with_capacity(COMMAND_BATCH_SIZE);
        let mut completions = Vec::with_capacity(COMPLETION_BATCH_SIZE);

        'main_loop: while !self.cancellation.is_cancelled() {
            let mut next_awakening = {
                let mut lock = self.entities.shards[self.shard_id].core.write();
                let services = self.services();
                let update_awakening_tick = lock.update_times(self.game_state.tick());
                let coro_awakening_tick =
                    lock.run_coroutines(&services, &self.game_state, completion_tx);
                tracing::debug!(
                    "Entity worker for shard {} next awakening times: {:?}",
                    self.shard_id,
                    update_awakening_tick
                );
                tracing::debug!(
                    "next awakening: {:?}, {} from now (coro: {}, update: {})",
                    update_awakening_tick.min(coro_awakening_tick),
                    (update_awakening_tick as i64 - self.game_state.tick() as i64),
                    (coro_awakening_tick as i64 - self.game_state.tick() as i64),
                    (update_awakening_tick as i64 - self.game_state.tick() as i64)
                );
                tracing::debug!("At tick {}", self.game_state.tick());
                drop(lock);
                update_awakening_tick.min(coro_awakening_tick)
            };

            rx_messages.clear();
            'poll: loop {
                tracing::debug!(
                    "Entity worker for shard {} polling with sleep for {:?}",
                    self.shard_id,
                    next_awakening
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
                                // then the loop repeats without sleeping
                            }
                        }
                    },
                    count = completion_lock.recv_many(&mut completions, COMPLETION_BATCH_SIZE) => {
                        tracing::debug!("Entity worker for shard {} received {} completions", self.shard_id, count);
                        match count {
                            0 => {
                                tracing::warn!("Entity worker for shard {} shutting down because sender disappeared", self.shard_id);
                                break 'main_loop;
                            },
                            _ => {
                                next_awakening = next_awakening.min(self.handle_completions(&mut completions));
                                // then the loop repeats without sleeping
                            }
                        }
                    }
                    _ = self.game_state.sleep_until_tick(next_awakening) => {
                        tracing::debug!("Entity worker for shard {} waking up", self.shard_id);
                        break 'poll;
                    },
                    _ = self.cancellation.cancelled() => {
                        tracing::debug!("Entity worker for shard {} shutting down", self.shard_id);
                        break 'main_loop;
                    },
                }
            }
        }
        Ok(())
    }

    fn handle_messages(&self, rx_messages: &mut Vec<EntityAction>) -> u64 {
        let mut lock = self.entities.shards[self.shard_id].core.write();
        let completion_tx = &self.entities.shards[self.shard_id].completion_tx;
        let mut indices: smallvec::SmallVec<[usize; COMMAND_BATCH_SIZE]> =
            smallvec::SmallVec::new();
        for action in rx_messages.drain(..) {
            match action {
                EntityAction::Insert(id, position, coroutine, entity_type, trailing_entities) => {
                    let def = self.game_state.entities().get_by_type(&entity_type);

                    let index = lock.insert(
                        id,
                        entity_type,
                        def,
                        position,
                        coroutine,
                        trailing_entities,
                        &self.services(),
                        completion_tx,
                        self.game_state.tick(),
                    );
                    indices.push(index);
                }
                EntityAction::Remove(id) => match lock.remove(id) {
                    Ok(_) => {}
                    Err(EntityError::NotFound) => {
                        tracing::warn!("Tried to remove non-existent entity {}", id);
                    }
                    Err(e) => {
                        tracing::error!("Failed to remove entity {}: {:?}", id, e);
                    }
                },
                EntityAction::SetKinematics(id, position, movement, next_movement) => {
                    match lock.set_kinematics(
                        id,
                        position,
                        movement,
                        next_movement,
                        self.game_state.tick(),
                    ) {
                        Ok(index) => {
                            indices.push(index);
                        }
                        Err(EntityError::NotFound) => {
                            tracing::warn!(
                                "Tried to set kinematics for non-existent entity {}",
                                id
                            );
                        }
                        Err(e) => {
                            tracing::error!("Failed to set kinematics for entity {}: {:?}", id, e);
                        }
                    }
                }
            }
        }
        indices.sort();
        indices.dedup();
        let mut next_event = u64::MAX;
        for index in indices {
            if index != 0 {
                //println!("post control message");
            }
            lock.run_coro_single(
                index,
                &self.services(),
                None,
                completion_tx,
                self.game_state.tick(),
            );
            next_event = next_event.min(lock.update_time_single(index, self.game_state.tick()));
        }
        next_event
    }

    fn handle_completions(&self, completions: &mut Vec<Completion<ContinuationResult>>) -> u64 {
        let mut lock = self.entities.shards[self.shard_id].core.write();
        let completion_tx = &self.entities.shards[self.shard_id].completion_tx;

        let mut next_event = u64::MAX;
        for completion in completions.drain(..) {
            completion
                .trace_buffer
                .log("In handle_completions, about to resume");
            let index = completion.index;

            if index >= lock.len {
                tracing::warn!("Tried to resume non-existent entity at index {}", index);
                continue;
            }

            if lock.control[index] & (CONTROL_PRESENT | CONTROL_AUTONOMOUS | CONTROL_SUSPENDED) == 0
            {
                tracing::warn!("Tried to resume non-existent/non-auton/non-suspended entity at index {} w/ control {:x}", index, lock.control[index]);

                completion
                    .trace_buffer
                    .log("WARN: Resuming nonexistent/non-auton/non-suspended entity");

                continue;
            }
            if lock.id[index] != completion.entity_id {
                tracing::warn!(
                    "Tried to resume entity at index {} w/ mismatched ID {} != {}",
                    index,
                    completion.entity_id,
                    lock.id[index]
                );
                completion.trace_buffer.log("WARN: Resuming mismatched ID");
                continue;
            }
            lock.control[index] &= !CONTROL_SUSPENDED;

            next_event = next_event.min(lock.run_coro_single(
                index,
                &self.services(),
                Some(completion),
                completion_tx,
                self.game_state.tick(),
            ));
        }
        next_event.min(self.game_state.tick() + 10 * 1_000_000_000)
    }

    fn services(&self) -> EntityCoroutineServices<'_> {
        EntityCoroutineServices {
            game_state: &self.game_state,
            sender: &self.entities.shards[self.shard_id].completion_tx,
        }
    }
}

#[derive(Debug, Error)]
pub enum EntityError {
    #[error("Entity not found")]
    NotFound,

    #[error("Duplicate class name")]
    DuplicateClassName,
}

pub enum MoveQueueType {
    SingleMove,
    Buffer8,
    Buffer64,
}

// TODO own and manage these, similar to block types and item defs.
/// Warning: ***API is extremely preliminary and subject to change***
pub struct EntityDef {
    pub move_queue_type: MoveQueueType,
    pub class_name: String,
    pub client_info: EntityAppearance,
}

pub struct EntityClass {
    pub def: EntityDef,
    pub class_id: u32,
}

/// An entity that is trailing behind the primary entity, e.g. a duckling
/// following a duck, or a minecart coupled into a train
#[derive(Debug, Clone, Copy)]
pub struct TrailingEntity {
    /// The class of the trailing entity
    pub class_id: u32,
    /// The distance from the *primary/first* (as opposed to preceding) entity
    pub trailing_distance: f32,
}
pub type TrailingEntities = Box<[TrailingEntity]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityClassId(u32);
impl EntityClassId {
    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Used to identify types of entities when spawning, dropping, etc.
/// Note that this is not an efficient encoding, and should not be
/// used in the hot path (e.g. bulk scans, coroutines, etc)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntityTypeId {
    /// The type of entity
    pub class: EntityClassId,
    /// Any additional data for this entity
    ///
    /// The interpretation of this depends on the entity type,
    /// and is subject to change
    pub data: Option<Box<[u8]>>,
}

pub(crate) const FAKE_ENTITY_CLASS_ID: EntityClassId = EntityClassId(0xFFFFFFFF);
/// The entity class ID for unknown entities and the fallback entity appearance
pub const UNKNOWN_ENTITY_CLASS_ID: EntityClassId = EntityClassId(0);
/// The entity class ID for a player; data is the username in UTF-8 bytes
pub const PLAYER_ENTITY_CLASS_ID: EntityClassId = EntityClassId(1);
/// The entity class ID for a dropped item; data is the item stack proto
pub const DROPPED_ITEM_CLASS_ID: EntityClassId = EntityClassId(2);

const ENTITY_TYPE_MANAGER_META_KEY: &[u8] = b"entity_types";

pub struct EntityTypeManager {
    types: Vec<Option<EntityDef>>,
    by_name: FxHashMap<String, u32>,
}
impl EntityTypeManager {
    pub(crate) fn create_or_load(db: &dyn GameDatabase) -> Result<EntityTypeManager> {
        let db_key = KeySpace::Metadata.make_key(ENTITY_TYPE_MANAGER_META_KEY);

        if let Some(encoded) = db.get(&db_key)? {
            Self::from_encoded(&encoded)
        } else {
            Self::new_empty()
        }
    }

    fn from_encoded(encoded: &[u8]) -> Result<EntityTypeManager> {
        let assignments = ServerEntityTypeAssignments::decode(encoded)?;

        let mut types = match assignments.entity_type.iter().map(|x| x.entity_class).max() {
            Some(max_type) => Vec::with_capacity(max_type as usize + 1),
            None => vec![],
        };
        types.push(make_unknown_entity_appearance());
        let mut by_name = FxHashMap::default();

        for def in assignments.entity_type.into_iter() {
            // The entity class is still None; nobody has registered an entity with that class yet.
            by_name.insert(def.short_name, def.entity_class);
        }

        Ok(EntityTypeManager {
            types,
            by_name: FxHashMap::default(),
        })
    }

    fn new_empty() -> Result<EntityTypeManager> {
        Ok(EntityTypeManager {
            types: vec![make_unknown_entity_appearance()],
            by_name: FxHashMap::from_iter([(
                "builtin:unknown".to_string(),
                UNKNOWN_ENTITY_CLASS_ID.0,
            )]),
        })
    }

    pub(crate) fn to_server_proto(&self) -> ServerEntityTypeAssignments {
        ServerEntityTypeAssignments {
            entity_type: self
                .types
                .iter()
                .enumerate()
                .map(|(class_id, def)| EntityTypeAssignment {
                    short_name: def.as_ref().unwrap().class_name.clone(),
                    entity_class: class_id as u32,
                })
                .collect(),
        }
    }

    pub(crate) fn to_client_protos(&self) -> Vec<protocol::entities::EntityDef> {
        self.types
            .iter()
            .enumerate()
            .flat_map(|(i, x)| {
                x.as_ref().map(|x| protocol::entities::EntityDef {
                    short_name: x.class_name.clone(),
                    entity_class: i as u32,
                    appearance: Some(x.client_info.clone()),
                })
            })
            .collect()
    }

    pub(crate) fn save_to(&self, db: &dyn GameDatabase) -> Result<()> {
        db.put(
            &KeySpace::Metadata.make_key(ENTITY_TYPE_MANAGER_META_KEY),
            &self.to_server_proto().encode_to_vec(),
        )?;
        db.flush()
    }

    pub fn register(&mut self, def: EntityDef) -> Result<EntityClassId> {
        match self.by_name.get(&def.class_name) {
            Some(class_id) => {
                ensure!(
                    self.types[*class_id as usize].is_none(),
                    EntityError::DuplicateClassName
                );
                self.types[*class_id as usize] = Some(def);
                Ok(EntityClassId(*class_id))
            }
            None => {
                let class_id = self.types.len() as u32;
                self.by_name.insert(def.class_name.clone(), class_id);
                self.types.push(Some(def));
                Ok(EntityClassId(class_id))
            }
        }
    }

    pub(crate) fn pre_build(&self) -> Result<()> {
        tracing::debug!(
            "Pre-building entity type manager ({} types)",
            self.types.len()
        );
        // Nothing to do in this version, but we expect to add some precomputes here
        Ok(())
    }
}

const UNKNOWN_ENTITITY_MESH: &[u8] = include_bytes!("media/unknown_entity.obj");

fn make_unknown_entity_appearance() -> Option<EntityDef> {
    Some(EntityDef {
        class_name: "builtin:unknown".to_string(),
        move_queue_type: MoveQueueType::SingleMove,
        client_info: EntityAppearance {
            custom_mesh: vec![formats::load_obj_mesh(
                UNKNOWN_ENTITITY_MESH,
                FALLBACK_UNKNOWN_TEXTURE,
            )
            .unwrap()],
            attachment_offset: vec3(0.0, 0.0, 0.0).try_into().ok(),
            attachment_offset_in_model_space: false,
        },
    })
}
