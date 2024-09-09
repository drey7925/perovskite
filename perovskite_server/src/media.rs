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

use anyhow::{bail, Context, Result};
use perovskite_core::protocol::audio::SampledSound;
use sha2::Digest;
use std::io::Cursor;
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

/// An token identifying a sampled sound
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SoundKey(pub u32);

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

struct SampledAudioDetails {
    key: u32,
    filename: String,
}

pub struct MediaManager {
    resources: HashMap<String, Resource>,
    sampled_audio: Vec<SampledAudioDetails>,
}

impl MediaManager {
    pub fn register_from_memory(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let key = name.to_string();
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
    pub fn register_from_file<P: AsRef<Path>>(&mut self, name: &str, path: P) -> Result<()> {
        let path_buf = PathBuf::from(path.as_ref());
        let key = name.to_string();
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
            sampled_audio: Vec::new(),
        }
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&String, &Resource)> {
        self.resources.iter()
    }

    /// Registers the given file to be used as sampled. The file data must be a valid
    /// WAV file with header.
    ///
    /// Errors:
    ///   * If no file has been registered with that key
    pub fn register_file_for_sampled_audio(&mut self, name: &str) -> Result<SoundKey> {
        if !self.resources.contains_key(name) {
            bail!("File with name {name} not found");
        }
        // Index 0 is reserved for no-sound
        let index = self.sampled_audio.len() + 1;
        if index > (u32::MAX as usize) {
            bail!("Audio resource count overflowed 2^32")
        }
        self.sampled_audio.push(SampledAudioDetails {
            key: index as u32,
            filename: name.to_string(),
        });

        log::info!("Registered sound {} with id {}", name, index);
        Ok(SoundKey(index as u32))
    }

    pub(crate) fn sampled_sound_client_protos(&self) -> Vec<SampledSound> {
        self.sampled_audio
            .iter()
            .map(|x| SampledSound {
                sound_id: x.key,
                media_filename: x.filename.clone(),
            })
            .collect()
    }
}
