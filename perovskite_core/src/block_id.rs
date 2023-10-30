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

use anyhow::{ensure, Result};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(unused)]
pub enum BlockError {
    #[error("Block type `{0}` already exists")]
    NameAlreadyExists(String),
    #[error("Block ID 0x{0:x} not found")]
    IdNotFound(u32),
    #[error("Block ID 0x{0:x} lowest 12 bits (variant) not zero")]
    VariantBitsNonzero(u32),
    #[error(
        "Block ID 0x{0:x} already present (short name `{1}`), and is not a placeholder block."
    )]
    IdAlreadyExists(u32, String),
    #[error("Too many block types are already registered")]
    TooManyBlocks,
    #[error(
        "BlockTypeRef/BlockTypeName came from the wrong BlockTypeManager. Ours: {0}, ref: {1}"
    )]
    WrongManager(usize, usize),
    #[error("Variant {0:x} is out of range (max is 0xfff")]
    VariantOutOfRange(u16),
    #[error("This BlockType ({0}) object is not registered with a BlockTypeManager")]
    BlockNotRegistered(String),
}
pub const BLOCK_VARIANT_MASK: u32 = 0xfff;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BlockId(pub u32);
impl BlockId {
    pub fn base_id(&self) -> u32 {
        self.0 & !BLOCK_VARIANT_MASK
    }
    pub fn index(&self) -> usize {
        (self.0 & !BLOCK_VARIANT_MASK) as usize >> 12
    }
    pub fn variant(&self) -> u16 {
        (self.0 & BLOCK_VARIANT_MASK) as u16
    }
    pub fn with_variant(self, variant: u16) -> Result<BlockId> {
        ensure!(
            variant & (BLOCK_VARIANT_MASK as u16) == variant,
            BlockError::VariantOutOfRange(variant)
        );
        Ok(BlockId(self.base_id() | (variant as u32)))
    }
    pub fn new(base: u32, variant: u16) -> Result<BlockId> {
        ensure!(
            base & BLOCK_VARIANT_MASK == 0,
            BlockError::VariantBitsNonzero(base)
        );
        ensure!(
            variant & (BLOCK_VARIANT_MASK as u16) == variant,
            BlockError::VariantOutOfRange(variant)
        );
        Ok(BlockId(base | (variant as u32)))
    }
    #[inline]
    pub fn equals_ignore_variant(&self, other: BlockId) -> bool {
        self.base_id() == other.base_id()
    }
    #[deprecated]
    pub fn id(&self) -> BlockId {
        *self
    }
}

impl From<u32> for BlockId {
    fn from(value: u32) -> Self {
        BlockId(value)
    }
}
impl From<BlockId> for u32 {
    fn from(value: BlockId) -> Self {
        value.0
    }
}
impl Debug for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("block:0x{:x}", self.0))
    }
}
