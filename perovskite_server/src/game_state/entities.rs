// This is an unoptimized, early implementation.
// Actual optimizations will follow based on real-world profiling once
// entities are implemented and have some use-cases.

use std::{
    sync::{atomic::AtomicU64, Weak},
    time::Instant,
};

use anyhow::{bail, Result};
use cgmath::Vector3;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use super::GameState;

static ENTITY_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct EntityId(u64);
impl EntityId {
    fn next() -> EntityId {
        EntityId(ENTITY_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    pub(crate) fn impossible_sentinel() -> EntityId {
        EntityId(u64::MAX)
    }

    pub(crate) fn client_id(&self) -> u64 {
        self.0
    }

    pub(crate) fn is_sentinel(&self) -> bool {
        self.0 == u64::MAX
    }
}

/// An entity that is controlled externally, such as by player movement,
/// an ongoing coroutine, etc.
pub(crate) struct DrivenEntity {
    /// A unique entity ID, does not match the index into the vector
    entity_id: EntityId,
    /// Last time position/velocity was updated. Extrapolate from there
    last_update: Instant,
    /// The position at last_update
    position: Vector3<f64>,
    /// The velocity to use for extapolation from last_update
    velocity: Vector3<f64>,
    /// The face direction of the entity. Only used for rendering
    face_direction: (f64, f64),
    // todo appearance
}
impl DrivenEntity {
    pub fn id(&self) -> EntityId {
        self.entity_id
    }
    pub fn drive(
        &mut self,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        calculated_at: Instant,
    ) {
        self.position = position;
        self.velocity = velocity;
        self.last_update = calculated_at;
    }
    pub fn sample(&self) -> (Vector3<f64>, Vector3<f64>) {
        (
            self.position + (self.velocity * (Instant::now() - self.last_update).as_secs_f64()),
            self.velocity,
        )
    }
    pub fn updated_since(&self, cutoff: Instant) -> bool {
        self.last_update > cutoff
    }
}

struct EntityManagerInner {
    game_state: Weak<GameState>,
    driven_entities: Vec<DrivenEntity>,
}

pub(crate) struct EntityManager {
    inner: RwLock<EntityManagerInner>,
}
impl EntityManager {
    pub(crate) fn new(game_state: Weak<GameState>) -> Self {
        Self {
            inner: RwLock::new(EntityManagerInner {
                driven_entities: Vec::new(),
                game_state,
            }),
        }
    }
    pub(crate) fn insert_entity(
        &self,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
    ) -> Result<EntityId> {
        let entity_id = EntityId::next();
        let mut lock = self.inner.write();
        lock.driven_entities.push(DrivenEntity {
            entity_id,
            last_update: Instant::now(),
            position,
            velocity,
            face_direction: (0.0, 0.0),
        });
        Ok(entity_id)
    }

    pub(crate) fn drive_entity(
        &self,
        id: EntityId,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
    ) -> Result<()> {
        // TODO this is extremely suboptimal for performance, but it works for now
        // We shouldn't be doing a linear search for an entity here
        //
        // We should use something like a generational arena
        let mut lock = self.inner.write();
        for entity in &mut lock.driven_entities {
            if entity.entity_id == id {
                entity.drive(position, velocity, Instant::now());
                return Ok(());
            }
        }
        bail!("Entity {} not found", id.0);
    }

    pub(crate) fn remove_entity(&self, entity_id: EntityId) {
        // TODO this is extremely suboptimal for performance, but it works for now before we scale the entity mechanism
        //
        // In particular: This requires O(N) time to both scan when driving, and when removing
        self.inner
            .write()
            .driven_entities
            .retain(|entity| entity.entity_id != entity_id);
    }

    pub(crate) fn contents(&self) -> MappedRwLockReadGuard<Vec<DrivenEntity>> {
        RwLockReadGuard::map(self.inner.read(), |inner| &inner.driven_entities)
    }
}
