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

use std::path::Path;

use anyhow::{Context, Result};
use rocksdb::{Options, ReadOptions, DB};
use tracy_client::span;

use super::database_engine::GameDatabase;

pub(crate) struct RocksDbBackend {
    db: rocksdb::DB,
}
impl RocksDbBackend {
    pub(crate) fn new<P: AsRef<Path>>(path: P) -> Result<RocksDbBackend> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.optimize_for_point_lookup(128);
        let db = DB::open(&opts, path.as_ref())?;
        tracing::info!("Opened DB at {:?}", path.as_ref());
        tracing::info!(
            "db stats: \n{}\n{}\ntotal size: {}",
            db.property_value("rocksdb.stats").unwrap().unwrap(),
            db.property_value("rocksdb.levelstats").unwrap().unwrap(),
            db.property_value("rocksdb.total-sst-files-size")
                .unwrap()
                .unwrap()
        );
        Ok(RocksDbBackend { db })
    }
}
impl Drop for RocksDbBackend {
    fn drop(&mut self) {
        tracing::info!("Closing DB");
        match self.db.flush() {
            Ok(_) => {}
            Err(e) => tracing::error!("Failed to flush DB: {}", e),
        }
        tracing::info!(
            "db stats: \n{}\n{}\ntotal size: {}",
            self.db.property_value("rocksdb.stats").unwrap().unwrap(),
            self.db
                .property_value("rocksdb.levelstats")
                .unwrap()
                .unwrap(),
            self.db
                .property_value("rocksdb.total-sst-files-size")
                .unwrap()
                .unwrap()
        );
    }
}
impl GameDatabase for RocksDbBackend {
    fn get(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        let _span = span!("db get");
        self.db.get(key).with_context(|| "RocksDB get failed")
    }

    fn get_nontemporal(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let _span = span!("db get nontemporal");
        let mut opts = ReadOptions::default();
        opts.fill_cache(false);
        self.db
            .get_opt(key, &opts)
            .with_context(|| "RocksDB get failed")
    }

    fn put(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
        let _span = span!("db put");
        self.db
            .put(key, value)
            .with_context(|| "RocksDB put failed")
    }

    fn delete(&self, key: &[u8]) -> anyhow::Result<()> {
        let _span = span!("db delete");
        self.db.delete(key).with_context(|| "RocksDB delete failed")
    }

    fn flush(&self) -> Result<()> {
        self.db.flush().with_context(|| "RocksDB flush failed")
    }

    fn read_prefix(
        &self,
        prefix: &[u8],
        callback: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let _span = span!("db read prefix");
        let mut opts = ReadOptions::default();
        opts.fill_cache(false);
        for x in self.db.prefix_iterator(prefix) {
            let (k, v) = x?;
            callback(&k, &v)?;
        }
        Ok(())
    }
}
