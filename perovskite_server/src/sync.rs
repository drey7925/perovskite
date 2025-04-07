use parking_lot::{Condvar, Mutex, RwLockReadGuard, RwLockWriteGuard};
use std::future::Future;
use std::sync::atomic::Ordering;

pub(crate) struct RwCondvar {
    c: Condvar,
    m: Mutex<()>,
}
impl RwCondvar {
    pub(crate) fn new() -> RwCondvar {
        RwCondvar {
            c: Condvar::new(),
            m: Mutex::new(()),
        }
    }

    pub(crate) fn wait_reader<T>(&self, g: &mut RwLockReadGuard<'_, T>) {
        let guard = self.m.lock();
        RwLockReadGuard::unlocked(g, || {
            // It may seem like there's a lock ordering violation between self.m and g.
            // However, there is not:
            //
            // g held |=============> RwLockReadGuard::unlocked       <====|
            // happens-before      \                                /
            // self.m held        |============> condvar wait <===|
            //
            // The two locks are held concurrently (g -> self.m) when we prepare to wait, but they
            // are not held concurrently when we finish waiting: self.m is released before we try
            // to acquire self.g.
            let mut guard = guard;
            self.c.wait(&mut guard);
        });
    }
    pub(crate) fn wait_writer<T>(&self, g: &mut RwLockWriteGuard<'_, T>) {
        let guard = self.m.lock();
        RwLockWriteGuard::unlocked(g, || {
            // Move the guard in so it gets unlocked before we re-lock g
            let mut guard = guard;
            self.c.wait(&mut guard);
        });
    }
    pub(crate) fn notify_all(&self) {
        self.c.notify_all();
    }
    pub(crate) fn notify_one(&self) {
        self.c.notify_one();
    }
}

/// An atomic variation of std::time::Instant, able to count
/// about 584 years from when it is constructed with new.
/// It cannot represent times before when it was constructed.
///
/// TODO: On machines that do not support 64-bit atomics,
/// provide a fallback that uses a mutex instead.
pub(crate) struct AtomicInstant {
    initial: std::time::Instant,
    offset: std::sync::atomic::AtomicU64,
}
impl AtomicInstant {
    pub(crate) fn new() -> AtomicInstant {
        AtomicInstant {
            initial: std::time::Instant::now(),
            offset: std::sync::atomic::AtomicU64::new(0),
        }
    }
    pub(crate) fn get_acquire(&self) -> std::time::Instant {
        let offset = self.offset.load(std::sync::atomic::Ordering::Acquire);
        self.initial + std::time::Duration::from_nanos(offset)
    }
    pub(crate) fn update_now_release(&self) {
        self.update_to_release(std::time::Instant::now());
    }
    pub(crate) fn get_relaxed(&self) -> std::time::Instant {
        let offset = self.offset.load(std::sync::atomic::Ordering::Relaxed);
        self.initial + std::time::Duration::from_nanos(offset)
    }
    pub(crate) fn update_now_relaxed(&self) {
        self.update_to_relaxed(std::time::Instant::now());
    }
    pub(crate) fn update_to_release(&self, when: std::time::Instant) {
        if when < self.initial {
            panic!(
                "Attempted to set an instant ({:?}) before AtomicInstant was constructed ({:?})",
                when, self.initial
            );
        }
        let offset = when
            .duration_since(self.initial)
            .as_nanos()
            .try_into()
            .unwrap();
        self.offset
            .store(offset, std::sync::atomic::Ordering::Release);
    }
    pub(crate) fn update_to_relaxed(&self, when: std::time::Instant) {
        if when < self.initial {
            panic!(
                "Attempted to set an instant ({:?}) before AtomicInstant was constructed ({:?})",
                when, self.initial
            );
        }
        let offset = when
            .duration_since(self.initial)
            .as_nanos()
            .try_into()
            .unwrap();
        self.offset
            .store(offset, std::sync::atomic::Ordering::Relaxed);
    }
}
