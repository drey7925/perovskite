// Copyright 2023 drey7925
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
pub mod rocksdb;

#[cfg(feature = "db_failure_injection")]
pub(crate) mod failure_injection {
    use crate::database::GameDatabase;
    use anyhow::bail;
    use rand::Rng;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    pub(crate) static FAILURE_CHANCE_OVER_256: AtomicU32 = AtomicU32::new(0);
    pub(crate) static HANG_CHANCE_OVER_256: AtomicU32 = AtomicU32::new(0);
    pub(crate) struct FailureInjectedDbWrapper<T> {
        inner: T,
    }

    impl<T: GameDatabase> FailureInjectedDbWrapper<T> {
        pub(crate) fn new(inner: T) -> FailureInjectedDbWrapper<T> {
            FailureInjectedDbWrapper { inner }
        }
    }

    #[inline(never)]
    fn db_hang() {
        std::thread::sleep(Duration::MAX);
    }
    fn maybe_fail() -> anyhow::Result<()> {
        let fail_chance = FAILURE_CHANCE_OVER_256.load(Ordering::Relaxed);
        let hang_chance = HANG_CHANCE_OVER_256.load(Ordering::Relaxed);
        if fail_chance == 0 && hang_chance == 0 {
            return Ok(());
        }

        let fail_chance = ((fail_chance as f64) / 256.0).clamp(0.0, 1.0);
        let hang_chance = ((hang_chance as f64) / 256.0).clamp(0.0, 1.0);

        let mut rng = rand::thread_rng();
        if rng.gen_bool(fail_chance) {
            bail!("Injected error");
        }
        if rng.gen_bool(hang_chance) {
            tracing::error!("Hanging a database thread");
            db_hang();
        }

        Ok(())
    }
    impl<T: GameDatabase> GameDatabase for FailureInjectedDbWrapper<T> {
        fn get(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
            maybe_fail()?;
            self.inner.get(key)
        }

        fn put(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
            maybe_fail()?;
            self.inner.put(key, value)
        }

        fn delete(&self, key: &[u8]) -> anyhow::Result<()> {
            maybe_fail()?;
            self.inner.delete(key)
        }

        fn flush(&self) -> anyhow::Result<()> {
            maybe_fail()?;
            self.inner.flush()
        }

        fn read_prefix(
            &self,
            prefix: &[u8],
            callback: &mut dyn FnMut(&[u8], &[u8]) -> anyhow::Result<()>,
        ) -> anyhow::Result<()> {
            maybe_fail()?;
            self.inner.read_prefix(prefix, callback)
        }
    }
}

use anyhow::Result;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

pub(crate) enum KeySpace {
    /// Core metadata for the game state, e.g. the block type list.
    /// Should generally contain only hardcoded keys.
    Metadata,
    /// Map chunks, keyed by the chunk coordinate
    MapchunkData,
    /// Plugin key-value storage
    Plugin,
    /// Inventory storage (may eventually become transactional,
    /// but at the moment it's possible for an "atomic" change to be partially committed
    /// with an unluckily-timed crash)
    Inventory,
    /// Player data (posiition, inventory, etc)
    Player,
    /// User metadata (login, etc)
    UserMeta,
    /// Entities
    Entity,
    /// Disaster recovery, write-only during normal gameplay, read only by recovery paths, manual
    /// tools, etc.
    DisasterRecovery,
}
impl KeySpace {
    pub(crate) fn make_key(&self, key: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(key.len() + 1);
        result.push(self.identifier());
        result.extend_from_slice(key);
        result
    }

    fn identifier(&self) -> u8 {
        match self {
            KeySpace::Metadata => b'0',
            KeySpace::MapchunkData => b'm',
            KeySpace::Plugin => b'p',
            KeySpace::Inventory => b'i',
            KeySpace::UserMeta => b'u',
            KeySpace::Player => b'P',
            KeySpace::Entity => b'e',
            KeySpace::DisasterRecovery => b'd',
        }
    }
}

pub trait GameDatabase: Send + Sync {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    /// Same as get, but does not keep the value cached in memory.
    ///
    /// Default impl will just call get, ignoring the cache hint.
    fn get_nontemporal(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get(key)
    }
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn flush(&self) -> Result<()>;

    fn read_prefix(
        &self,
        prefix: &[u8],
        callback: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()>;
}

/// Test-only game database
pub struct InMemGameDatabase {
    data: Mutex<FxHashMap<Vec<u8>, Vec<u8>>>,
}
impl InMemGameDatabase {
    pub fn new() -> InMemGameDatabase {
        InMemGameDatabase {
            data: HashMap::default().into(),
        }
    }
}
impl GameDatabase for InMemGameDatabase {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.data.lock().get(key).cloned())
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.data.lock().insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.data.lock().remove(key);
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn read_prefix(
        &self,
        prefix: &[u8],
        callback: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        for (key, value) in self.data.lock().iter() {
            if key.starts_with(prefix) {
                callback(key, value)?;
            }
        }
        Ok(())
    }
}
