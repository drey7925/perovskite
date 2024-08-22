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

pub(crate) mod database_engine;
pub(crate) mod rocksdb;

#[cfg(feature = "db_failure_injection")]
pub(crate) mod failure_injection {
    use crate::database::database_engine::GameDatabase;
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
