use std::ops::{Deref, DerefMut};

pub struct RwCondvar<T, S: SyncBackend> {
    c: S::Condvar<T>,
    m: S::Mutex<()>,
}
impl<T: Send + Sync, S: SyncBackend> RwCondvar<T, S> {
    pub fn new() -> RwCondvar<T, S> {
        RwCondvar {
            c: S::Condvar::new(),
            m: S::Mutex::new(()),
        }
    }

    pub fn wait_reader(&self, g: &mut S::ReadGuard<'_, T>) {
        let guard = self.m.lock();
        S::RwLock::reader_unlocked(g, || {
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
    pub fn wait_writer(&self, g: &mut S::WriteGuard<'_, T>) {
        let guard = self.m.lock();
        S::RwLock::writer_unlocked(g, || {
            // Move the guard in so it gets unlocked before we re-lock g
            let mut guard = guard;
            self.c.wait(&mut guard);
        });
    }
    pub fn notify_all(&self) {
        let _guard = self.m.lock();
        self.c.notify_all();
    }
    pub fn notify_one(&self) {
        let _guard = self.m.lock();
        self.c.notify_one();
    }
}

pub trait GenericMutex<T, S: SyncBackend> {
    fn lock(&self) -> S::Guard<'_, T>;
    fn new(t: T) -> Self;
    fn into_inner(self) -> T;
}
pub trait GenericRwLock<T, S: SyncBackend> {
    fn lock_read(&self) -> S::ReadGuard<'_, T>;
    fn lock_write(&self) -> S::WriteGuard<'_, T>;
    fn new(t: T) -> Self;
    fn downgrade_writer<'a>(guard: S::WriteGuard<'a, T>) -> S::ReadGuard<'a, T>
    where
        T: 'a;
    fn reader_unlocked<R, F: FnOnce() -> R>(guard: &mut S::ReadGuard<'_, T>, f: F) -> R;
    fn writer_unlocked<R, F: FnOnce() -> R>(guard: &mut S::WriteGuard<'_, T>, f: F) -> R;
    fn bump_read(guard: &mut S::ReadGuard<'_, T>);
    fn bump_write(guard: &mut S::WriteGuard<'_, T>);
    fn into_inner(self) -> T;
    fn read_recursive(&self) -> S::ReadGuard<'_, T>;
}

trait GenericCondvar<S: SyncBackend> {
    fn new() -> Self;
    fn wait<Y>(&self, guard: &mut S::Guard<'_, Y>);
    fn notify_one(&self);
    fn notify_all(&self);
}

pub trait SyncBackend: 'static + Sized {
    type Mutex<T: Send + Sync>: GenericMutex<T, Self> + Send + Sync;
    type Guard<'a, T: 'a>: DerefMut<Target = T> + 'a;
    type Condvar<T>: GenericCondvar<Self> + Send + Sync;
    type RwLock<T: Send + Sync>: GenericRwLock<T, Self> + Send + Sync;
    type ReadGuard<'a, T: 'a>: Deref<Target = T> + 'a;
    type WriteGuard<'a, T: 'a>: DerefMut<Target = T> + 'a;
}
mod parking_lot {
    use crate::sync::{GenericCondvar, GenericMutex, GenericRwLock, SyncBackend};

    impl<T> GenericMutex<T, ParkingLotBackend> for parking_lot::Mutex<T> {
        #[inline]
        fn lock(&self) -> parking_lot::MutexGuard<'_, T> {
            parking_lot::Mutex::lock(self)
        }
        #[inline]
        fn new(t: T) -> Self {
            parking_lot::Mutex::new(t)
        }
        #[inline]
        fn into_inner(self) -> T {
            parking_lot::Mutex::into_inner(self)
        }
    }

    impl<T> GenericRwLock<T, ParkingLotBackend> for parking_lot::RwLock<T> {
        #[inline]
        fn lock_read(&self) -> parking_lot::RwLockReadGuard<'_, T> {
            parking_lot::RwLock::read(self)
        }
        #[inline]
        fn lock_write(&self) -> parking_lot::RwLockWriteGuard<'_, T> {
            parking_lot::RwLock::write(self)
        }
        #[inline]
        fn new(t: T) -> Self {
            parking_lot::RwLock::new(t)
        }
        #[inline]
        fn downgrade_writer<'a>(
            guard: parking_lot::RwLockWriteGuard<'a, T>,
        ) -> parking_lot::RwLockReadGuard<'a, T>
        where
            T: 'a,
        {
            parking_lot::RwLockWriteGuard::downgrade(guard)
        }
        #[inline]
        fn reader_unlocked<R, F: FnOnce() -> R>(
            guard: &mut parking_lot::RwLockReadGuard<'_, T>,
            f: F,
        ) -> R {
            parking_lot::RwLockReadGuard::unlocked(guard, f)
        }
        #[inline]
        fn writer_unlocked<R, F: FnOnce() -> R>(
            guard: &mut parking_lot::RwLockWriteGuard<'_, T>,
            f: F,
        ) -> R {
            parking_lot::RwLockWriteGuard::unlocked(guard, f)
        }
        #[inline]
        fn bump_read(guard: &mut parking_lot::RwLockReadGuard<'_, T>) {
            parking_lot::RwLockReadGuard::bump(guard);
        }
        #[inline]
        fn bump_write(guard: &mut parking_lot::RwLockWriteGuard<'_, T>) {
            parking_lot::RwLockWriteGuard::bump(guard);
        }
        #[inline]
        fn into_inner(self) -> T {
            parking_lot::RwLock::into_inner(self)
        }
        #[inline]
        fn read_recursive(&self) -> parking_lot::RwLockReadGuard<'_, T> {
            parking_lot::RwLock::read_recursive(self)
        }
    }

    impl GenericCondvar<ParkingLotBackend> for parking_lot::Condvar {
        fn new() -> Self {
            parking_lot::Condvar::new()
        }
        #[inline]
        fn wait<T>(&self, guard: &mut parking_lot::MutexGuard<'_, T>) {
            parking_lot::Condvar::wait(self, guard)
        }
        #[inline]
        fn notify_one(&self) {
            parking_lot::Condvar::notify_one(self);
        }
        #[inline]
        fn notify_all(&self) {
            parking_lot::Condvar::notify_all(self);
        }
    }

    pub struct ParkingLotBackend;
    impl SyncBackend for ParkingLotBackend {
        type Mutex<T: Send + Sync> = parking_lot::Mutex<T>;
        type Guard<'a, T: 'a> = parking_lot::MutexGuard<'a, T>;
        type Condvar<T> = parking_lot::Condvar;
        type RwLock<T: Send + Sync> = parking_lot::RwLock<T>;
        type ReadGuard<'a, T: 'a> = parking_lot::RwLockReadGuard<'a, T>;
        type WriteGuard<'a, T: 'a> = parking_lot::RwLockWriteGuard<'a, T>;
    }
}

mod loom {
    use crate::sync::{GenericCondvar, GenericMutex, GenericRwLock, SyncBackend};
    use scopeguard::defer;
    use std::mem::MaybeUninit;
    use std::ops::{Deref, DerefMut};
    use std::process::abort;

    pub struct LoomBackend;

    pub struct RwLockWriteGuard<'a, T: 'a> {
        inner: loom::sync::RwLockWriteGuard<'a, T>,
        lock: &'a loom::sync::RwLock<T>,
    }
    impl<'a, T: 'a> Deref for RwLockWriteGuard<'a, T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            self.inner.deref()
        }
    }
    impl<'a, T: 'a> DerefMut for RwLockWriteGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.inner.deref_mut()
        }
    }

    pub struct RwLockReadGuard<'a, T: 'a> {
        inner: loom::sync::RwLockReadGuard<'a, T>,
        lock: &'a loom::sync::RwLock<T>,
    }
    impl<'a, T: 'a> Deref for RwLockReadGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.inner.deref()
        }
    }

    impl<T> GenericMutex<T, LoomBackend> for loom::sync::Mutex<T> {
        #[inline]
        fn lock(&self) -> loom::sync::MutexGuard<'_, T> {
            loom::sync::Mutex::lock(self).unwrap()
        }
        #[inline]
        fn new(t: T) -> Self {
            loom::sync::Mutex::new(t)
        }
        #[inline]
        fn into_inner(self) -> T {
            loom::sync::Mutex::into_inner(self).unwrap()
        }
    }

    impl<T> GenericRwLock<T, LoomBackend> for loom::sync::RwLock<T> {
        #[inline]
        #[track_caller]
        fn lock_read(&self) -> RwLockReadGuard<'_, T> {
            RwLockReadGuard {
                inner: loom::sync::RwLock::read(self).unwrap(),
                lock: self,
            }
        }
        #[inline]
        #[track_caller]
        fn lock_write(&self) -> RwLockWriteGuard<'_, T> {
            RwLockWriteGuard {
                inner: loom::sync::RwLock::write(self).unwrap(),
                lock: &self,
            }
        }
        #[inline]
        #[track_caller]
        fn new(t: T) -> Self {
            loom::sync::RwLock::new(t)
        }
        #[inline]
        #[track_caller]
        fn downgrade_writer<'a>(mut guard: RwLockWriteGuard<'a, T>) -> RwLockReadGuard<'a, T>
        where
            T: 'a,
        {
            loom::stop_exploring();
            std::mem::drop(guard.inner);
            let new_lock = guard.lock.read().unwrap();
            loom::explore();
            RwLockReadGuard {
                inner: new_lock,
                lock: guard.lock,
            }
        }
        #[inline]
        #[track_caller]
        fn reader_unlocked<R, F: FnOnce() -> R>(guard: &mut RwLockReadGuard<'_, T>, f: F) -> R {
            unsafe {
                let inner: &mut MaybeUninit<loom::sync::RwLockReadGuard<T>> =
                    std::mem::transmute(&mut guard.inner);
                inner.assume_init_drop();
                defer! {
                    inner.write(guard.lock.read().unwrap());
                };
                let result = f();
                result
            }
        }
        #[inline]
        #[track_caller]
        fn writer_unlocked<R, F: FnOnce() -> R>(guard: &mut RwLockWriteGuard<'_, T>, f: F) -> R {
            unsafe {
                let inner: &mut MaybeUninit<loom::sync::RwLockWriteGuard<T>> =
                    std::mem::transmute(&mut guard.inner);
                inner.assume_init_drop();
                defer! {
                    inner.write(guard.lock.write().unwrap());
                };
                let result = f();
                result
            }
        }
        #[inline]
        #[track_caller]
        fn bump_read(guard: &mut RwLockReadGuard<'_, T>) {
            Self::reader_unlocked(guard, || {})
        }
        #[inline]
        #[track_caller]
        fn bump_write(guard: &mut RwLockWriteGuard<'_, T>) {
            Self::writer_unlocked(guard, || {})
        }
        #[inline]
        #[track_caller]
        fn into_inner(self) -> T {
            loom::sync::RwLock::into_inner(self).unwrap()
        }
        #[inline]
        #[track_caller]
        fn read_recursive(&self) -> RwLockReadGuard<'_, T> {
            // loom doesn't quite support read_recursive
            RwLockReadGuard {
                inner: loom::sync::RwLock::read(self).unwrap(),
                lock: self,
            }
        }
    }

    impl GenericCondvar<LoomBackend> for loom::sync::Condvar {
        fn new() -> Self {
            loom::sync::Condvar::new()
        }
        #[inline]
        #[track_caller]
        fn wait<T>(&self, guard: &mut loom::sync::MutexGuard<'_, T>) {
            // Safety: This is used in test-only code, and further safety is described below.
            // This is rather unfortunate, because if we inlined `Condvar::wait` we could see that
            // it doesn't do anything to the guard, just returns it.
            unsafe {
                // Safety: Safe to transmute as long as we don't leak out the uninit
                let guard: &mut MaybeUninit<loom::sync::MutexGuard<T>> = std::mem::transmute(guard);
                // Safety: guard is initialized. We can take, leaving it uninitialized
                let taken_guard =
                    std::mem::replace(&mut *guard, MaybeUninit::uninit()).assume_init();
                // This will initialize the pointee of guard
                guard.write(
                    loom::sync::Condvar::wait(self, taken_guard).unwrap_or_else(|_| {
                        eprintln!("loom condvar wait returned Err");
                        abort();
                    }),
                );
            }
        }
        #[inline]
        fn notify_one(&self) {
            loom::sync::Condvar::notify_one(self);
        }
        #[inline]
        fn notify_all(&self) {
            loom::sync::Condvar::notify_all(self);
        }
    }

    impl SyncBackend for LoomBackend {
        type Mutex<T: Send + Sync> = loom::sync::Mutex<T>;
        type Guard<'a, T: 'a> = loom::sync::MutexGuard<'a, T>;
        type Condvar<T> = loom::sync::Condvar;
        type RwLock<T: Send + Sync> = loom::sync::RwLock<T>;
        type ReadGuard<'a, T: 'a> = RwLockReadGuard<'a, T>;
        type WriteGuard<'a, T: 'a> = RwLockWriteGuard<'a, T>;
    }
}

pub use loom::LoomBackend as TestonlyLoomBackend;
pub use parking_lot::ParkingLotBackend as DefaultSyncBackend;
