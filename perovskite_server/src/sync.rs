use parking_lot::{Condvar, Mutex, RwLockReadGuard, RwLockWriteGuard};

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

pub(crate) use perovskite_core::util::AtomicInstant;
