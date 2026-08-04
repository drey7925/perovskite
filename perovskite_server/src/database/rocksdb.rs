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

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rocksdb::{ReadOptions, DB};
use tracy_client::span;

use crate::database::KeySpace;
pub(crate) use rocksdb::Options;

use super::{DbKey, GameDatabase};

pub(crate) struct RocksDbBackend {
    main_db: DB,
    auth_db: DB,
    dr_db: DB,
}
impl RocksDbBackend {
    pub(crate) fn new<P: AsRef<Path>>(
        path: P,
        options: rocksdb::Options,
    ) -> Result<RocksDbBackend> {
        let main_path = PathBuf::from(path.as_ref()).join("main");
        let auth_path = PathBuf::from(path.as_ref()).join("auth");
        let dr_path = PathBuf::from(path.as_ref()).join("dr");

        let mut secondary_options = rocksdb::Options::default();
        secondary_options.create_if_missing(true);
        secondary_options.optimize_for_point_lookup(1);
        secondary_options.set_max_open_files(4);

        let db = DB::open(&options, &main_path)?;
        tracing::info!("Opened DB at {:?}", main_path);
        tracing::info!(
            "db stats: \n{}\n{}\ntotal size: {},\nest. live data size: {}, est. num-keys: {}",
            db.property_value("rocksdb.stats")?
                .unwrap_or_else(|| String::from("???")),
            db.property_value("rocksdb.levelstats")?
                .unwrap_or_else(|| String::from("???")),
            db.property_value("rocksdb.total-sst-files-size")?
                .unwrap_or_else(|| String::from("???")),
            db.property_value("rocksdb.estimate-live-data-size")?
                .unwrap_or_else(|| String::from("???")),
            db.property_value("rocksdb.estimate-num-keys")?
                .unwrap_or_else(|| String::from("???"))
        );

        let auth_db = DB::open(&secondary_options, &auth_path)?;
        let dr_db = DB::open(&secondary_options, &dr_path)?;
        tracing::info!("Opened secondary dbs");

        Ok(RocksDbBackend {
            main_db: db,
            auth_db,
            dr_db,
        })
    }

    fn db(&self, key_space: KeySpace) -> &DB {
        // Warning: changing the routing of entries to keyspaces is a breaking
        // change unless it's a new keyspace.
        match key_space {
            KeySpace::UserAuth => &self.auth_db,
            KeySpace::DisasterRecovery => &self.dr_db,
            _ => &self.main_db,
        }
    }
}
impl Drop for RocksDbBackend {
    fn drop(&mut self) {
        // Avoid panicking in drop, which causes an abort.
        fn safe_unwrap<T>(x: Result<Option<String>, T>) -> String {
            x.unwrap_or_else(|_| Some(String::from("<Err>")))
                .unwrap_or_else(|| String::from("???"))
        }

        tracing::info!("Closing DB");
        match self.main_db.flush() {
            Ok(_) => {}
            Err(e) => tracing::error!("Failed to flush DB: {}", e),
        }
        tracing::info!(
            "db stats: \n{}\n{}\ntotal size: {}\nbackground errors: {}\nestimated live data size: {}\nestimated num-keys: {}",
            safe_unwrap(self.main_db.property_value("rocksdb.stats")),
            safe_unwrap(self.main_db.property_value("rocksdb.levelstats")),
            safe_unwrap(self.main_db.property_value("rocksdb.total-sst-files-size")),
            safe_unwrap(self.main_db.property_value("rocksdb.background-errors")),
            safe_unwrap(self.main_db.property_value("rocksdb.estimate-live-data-size")),
            safe_unwrap(self.main_db.property_value("rocksdb.estimate-num-keys"))
        );
    }
}
impl GameDatabase for RocksDbBackend {
    fn get(&self, key: &DbKey) -> Result<Option<Vec<u8>>> {
        let _span = span!("db get");
        self.db(key.space)
            .get(key.to_db_key())
            .with_context(|| "RocksDB get failed")
    }

    fn get_nontemporal(&self, key: &DbKey) -> Result<Option<Vec<u8>>> {
        let _span = span!("db get nontemporal");
        let mut opts = ReadOptions::default();
        opts.fill_cache(false);
        self.db(key.space)
            .get_opt(key.to_db_key(), &opts)
            .with_context(|| "RocksDB get failed")
    }

    fn put(&self, key: &DbKey, value: &[u8]) -> Result<()> {
        let _span = span!("db put");
        self.db(key.space)
            .put(key.to_db_key(), value)
            .with_context(|| "RocksDB put failed")
    }

    fn delete(&self, key: &DbKey) -> Result<()> {
        let _span = span!("db delete");
        self.db(key.space)
            .delete(key.to_db_key())
            .with_context(|| "RocksDB delete failed")
    }

    fn flush(&self) -> Result<()> {
        self.main_db
            .flush()
            .with_context(|| "RocksDB flush failed")?;
        self.auth_db
            .flush()
            .with_context(|| "RocksDB flush failed")?;
        self.dr_db.flush().with_context(|| "RocksDB flush failed")?;
        Ok(())
    }

    fn read_prefix(
        &self,
        prefix: &DbKey,
        callback: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let _span = span!("db read prefix");
        let mut opts = ReadOptions::default();
        opts.fill_cache(false);
        for x in self.db(prefix.space).prefix_iterator(prefix.to_db_key()) {
            let (k, v) = x?;
            callback(&k, &v)?;
        }
        Ok(())
    }
}
