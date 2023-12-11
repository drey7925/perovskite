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

use anyhow::Result;
use parking_lot::Mutex;
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
    /// with an unluckily-timed crash
    Inventory,
    /// Player data (posiition, inventory, etc)
    Player,
    /// User metadata (login, etc)
    UserMeta,
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
        }
    }
}

pub(crate) trait GameDatabase: Send + Sync {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    /// Same as get, but does not keep the value cached in memory.
    /// 
    /// Default impl will just call get
    fn get_nontemporal(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get(key)
    }
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn flush(&self) -> Result<()>;

    fn read_prefix(&self, prefix: &[u8], callback: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>) -> Result<()>;
}

/// Test-only game database
pub(crate) struct InMemGameDabase {
    data: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
}
impl InMemGameDabase {
    pub(crate) fn new() -> InMemGameDabase {
        InMemGameDabase {
            data: HashMap::new().into(),
        }
    }
}
impl GameDatabase for InMemGameDabase {
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

    fn read_prefix(&self, prefix: &[u8], callback: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>) -> Result<()> {
        for (key, value) in self.data.lock().iter() {
            if key.starts_with(prefix) {
                callback(key, value)?;
            }
        }
        Ok(())
    }
}
