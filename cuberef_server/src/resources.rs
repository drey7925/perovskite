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

use anyhow::{bail, Result};
use sha2::Digest;
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(unused)]
pub enum ResourceError {
    #[error("Resource `{0}` already exists")]
    ResourceAlreadyExists(String),
}

enum ResourceData {
    Owned(Vec<u8>),
    Path(PathBuf),
}
impl ResourceData {
    pub(crate) fn get(&self) -> Result<Vec<u8>> {
        match self {
            ResourceData::Owned(x) => Ok(x.clone()),
            ResourceData::Path(p) => {
                let mut buf = Vec::new();
                File::open(p)?.read_to_end(&mut buf)?;
                Ok(buf)
            }
        }
    }
}

pub struct Resource {
    data: ResourceData,
    hash: Vec<u8>,
}
impl Resource {
    pub(crate) fn hash(&self) -> &[u8] {
        &self.hash
    }
    pub(crate) fn data(&self) -> Result<Vec<u8>> {
        self.data.get()
    }
}

pub struct MediaManager {
    resources: HashMap<String, Resource>,
}
impl MediaManager {
    pub fn register_from_memory(&mut self, key: &str, data: &[u8]) -> Result<()> {
        let key = key.to_string();
        match self.resources.entry(key.clone()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                bail!(ResourceError::ResourceAlreadyExists(key))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let hash = sha2::Sha256::digest(data).to_vec();
                entry.insert(Resource {
                    data: ResourceData::Owned(data.to_vec()),
                    hash,
                });
                log::info!("Registered resource {} ({} bytes)", key, data.len());
                Ok(())
            }
        }
    }
    pub fn register_from_file<P: AsRef<Path>>(&mut self, key: String, path: P) -> Result<()> {
        let path_buf = PathBuf::from(path.as_ref());
        match self.resources.entry(key.clone()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                bail!(ResourceError::ResourceAlreadyExists(key))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let mut data = Vec::new();
                File::open(&path_buf)?.read_to_end(&mut data)?;
                let hash = sha2::Sha256::digest(data).to_vec();
                entry.insert(Resource {
                    data: ResourceData::Path(path_buf),
                    hash,
                });
                Ok(())
            }
        }
    }
    pub fn get(&self, key: &str) -> Option<&Resource> {
        self.resources.get(key)
    }

    pub(crate) fn new() -> MediaManager {
        MediaManager {
            resources: HashMap::new(),
        }
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&String, &Resource)> {
        self.resources.iter()
    }
}
