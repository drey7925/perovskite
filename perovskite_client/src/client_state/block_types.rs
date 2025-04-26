use anyhow::{ensure, Context, Result};
use perovskite_core::block_id::BlockId;
use std::num::NonZeroU32;

use crate::client_state::make_fallback_blockdef;
use bitvec::prelude as bv;
use perovskite_core::protocol::blocks::block_type_def::RenderInfo;
use perovskite_core::protocol::blocks::{
    BlockTypeDef, CubeRenderInfo, CubeRenderMode, CubeVariantEffect,
};
use rustc_hash::FxHashMap;

pub(crate) struct ClientBlockTypeManager {
    block_defs: Vec<Option<BlockTypeDef>>,
    fallback_block_def: BlockTypeDef,
    light_propagators: bv::BitVec,
    light_emitters: Vec<u8>,
    solid_opaque_blocks: bv::BitVec,
    variant_controls_extent: bv::BitVec,
    check_neighbor_variant: bv::BitVec,
    transparent_render_blocks: bv::BitVec,
    translucent_render_blocks: bv::BitVec,
    name_to_id: FxHashMap<String, BlockId>,
    audio_emitters: Vec<Option<(NonZeroU32, f32)>>,
}
impl ClientBlockTypeManager {
    pub(crate) fn new(server_defs: Vec<BlockTypeDef>) -> Result<ClientBlockTypeManager> {
        let max_id = server_defs
            .iter()
            .map(|x| x.id)
            .max()
            .with_context(|| "Server defs were empty")?;

        let mut block_defs = Vec::new();
        block_defs.resize_with(BlockId(max_id).index() + 1, || None);

        let mut light_propagators = bv::BitVec::new();
        light_propagators.resize(BlockId(max_id).index() + 1, false);

        let mut solid_opaque_blocks = bv::BitVec::new();
        solid_opaque_blocks.resize(BlockId(max_id).index() + 1, false);
        let mut variant_controls_extent = bv::BitVec::new();
        variant_controls_extent.resize(BlockId(max_id).index() + 1, false);
        let mut check_neighbor_variant = bv::BitVec::new();
        check_neighbor_variant.resize(BlockId(max_id).index() + 1, false);

        let mut transparent_render_blocks = bv::BitVec::new();
        transparent_render_blocks.resize(BlockId(max_id).index() + 1, false);

        let mut translucent_render_blocks = bv::BitVec::new();
        translucent_render_blocks.resize(BlockId(max_id).index() + 1, false);

        let mut light_emitters = Vec::new();
        light_emitters.resize(BlockId(max_id).index() + 1, 0);
        let mut audio_emitters = Vec::new();
        audio_emitters.resize(BlockId(max_id).index() + 1, None);

        let mut name_to_id = FxHashMap::default();

        for def in server_defs {
            let id = BlockId(def.id);
            ensure!(id.variant() == 0);
            ensure!(block_defs[id.index()].is_none());
            name_to_id.insert(def.short_name.clone(), id);
            if def.allow_light_propagation {
                light_propagators.set(id.index(), true);
            }
            let light_emission = if def.light_emission > 15 {
                log::warn!(
                    "Clamping light emission of {} from {} to 15",
                    def.short_name,
                    def.light_emission
                );
                15
            } else {
                def.light_emission as u8
            };
            if let Some(RenderInfo::Cube(render_info)) = &def.render_info {
                if render_info.render_mode() == CubeRenderMode::SolidOpaque {
                    solid_opaque_blocks.set(id.index(), true);
                }
                if render_info.variant_effect == CubeVariantEffect::Liquid.into()
                    || render_info.variant_effect == CubeVariantEffect::CubeVariantHeight.into()
                {
                    variant_controls_extent.set(id.index(), true);
                }
                // Only liquids blend smoothly to their neighbors
                if render_info.variant_effect == CubeVariantEffect::CubeVariantHeight.into() {
                    check_neighbor_variant.set(id.index(), true);
                }
            }

            let is_transparent = match &def.render_info {
                Some(RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. })) => {
                    *x == CubeRenderMode::Transparent.into()
                }
                // plant-like blocks always go into the transparent pass
                Some(RenderInfo::PlantLike(_)) => true,
                // axis-aligned boxes always go into the transparent pass
                Some(RenderInfo::AxisAlignedBoxes(_)) => true,
                Some(_) | None => false,
            };

            if is_transparent {
                transparent_render_blocks.set(id.index(), true);
            }

            let is_translucent = def.render_info.as_ref().is_some_and(|x| match x {
                RenderInfo::Cube(CubeRenderInfo { render_mode: x, .. }) => {
                    *x == CubeRenderMode::Translucent.into()
                }
                _ => false,
            });

            if is_translucent {
                translucent_render_blocks.set(id.index(), true);
            }

            if let Some(sound_id) = NonZeroU32::new(def.sound_id) {
                audio_emitters[id.index()] = Some((sound_id, def.sound_volume));
            }

            light_emitters[id.index()] = light_emission;
            block_defs[id.index()] = Some(def);
        }

        Ok(ClientBlockTypeManager {
            block_defs,
            fallback_block_def: make_fallback_blockdef(),
            light_propagators,
            light_emitters,
            solid_opaque_blocks,
            variant_controls_extent,
            check_neighbor_variant,
            transparent_render_blocks,
            translucent_render_blocks,
            name_to_id,
            audio_emitters,
        })
    }

    pub(crate) fn all_block_defs(&self) -> impl Iterator<Item = &BlockTypeDef> {
        self.block_defs.iter().flatten()
    }

    pub(crate) fn get_fallback_blockdef(&self) -> &BlockTypeDef {
        &self.fallback_block_def
    }
    pub(crate) fn get_blockdef(&self, id: BlockId) -> Option<&BlockTypeDef> {
        match self.block_defs.get(id.index()) {
            // none if get() failed due to bounds check
            None => None,
            Some(x) => {
                // Still an option, since we might be missing block defs from the server
                x.as_ref()
            }
        }
    }

    pub(crate) fn get_block_by_name(&self, name: &str) -> Option<BlockId> {
        self.name_to_id.get(name).copied()
    }

    /// Determines whether this block propagates light
    #[inline]
    pub(crate) fn propagates_light(&self, id: BlockId) -> bool {
        // Optimization note: This is called from the neighbor propagation code, which is single-threaded for
        // consistency. Unlike mesh generation, which can be parallelized to as many cores as are available, this
        // is *much* more important to speed up. Therefore, these functions get special implementations rather than
        // going through the usual get_blockdef() path. In particular, going through get_blockdef() involves large structs, hence cache pressure
        //
        // Todo: actually benchmark and optimize this. (this comment only provides rationale, but the code hasn't been optimized yet)
        // A typical cacheline on x86 has a size of 64 bytes == 512 bits. A typical l1d cache size is on the order of double-digit KiB -> triple digit kilobits
        // This is plenty of space to fit the bitvecs, although there will still be some cache misses since the chunk won't fit in a typical L1$.
        // Hence, a bitvec is the initial, and likely the fastest, approach. I can't imagine that bloom filtering would be cheap compared to bit reads, even if
        // we do have some caches misses along the way.
        //
        // n.b. the working set for the light propagation algorithm is at least 110 KiB.
        if id.index() < self.light_propagators.len() {
            self.light_propagators[id.index()]
        } else {
            // unknown blocks don't propagate light
            false
        }
    }
    #[inline]
    pub(crate) fn light_emission(&self, id: BlockId) -> u8 {
        if id.index() < self.light_emitters.len() {
            self.light_emitters[id.index()]
        } else {
            0
        }
    }
    #[inline]
    pub(crate) fn is_solid_opaque(&self, id: BlockId) -> bool {
        if id.index() < self.solid_opaque_blocks.len() {
            self.solid_opaque_blocks[id.index()]
        } else {
            // unknown blocks are solid opaque
            true
        }
    }
    #[inline]
    pub(crate) fn is_extent_controlled_by_variant(&self, id: BlockId) -> bool {
        if id.index() < self.variant_controls_extent.len() {
            self.variant_controls_extent[id.index()]
        } else {
            // unknown blocks are solid opaque
            false
        }
    }

    #[inline]
    pub(crate) fn check_neighbor_variant_for_face_suppression(&self, id: BlockId) -> bool {
        if id.index() < self.check_neighbor_variant.len() {
            self.check_neighbor_variant[id.index()]
        } else {
            // unknown blocks are solid opaque
            false
        }
    }
    #[inline]
    pub(crate) fn is_transparent_render(&self, id: BlockId) -> bool {
        if id.index() < self.transparent_render_blocks.len() {
            self.transparent_render_blocks[id.index()]
        } else {
            // unknown blocks are solid opaque
            false
        }
    }

    #[inline]
    pub(crate) fn is_translucent_render(&self, id: BlockId) -> bool {
        if id.index() < self.translucent_render_blocks.len() {
            self.translucent_render_blocks[id.index()]
        } else {
            // unknown blocks are solid opaque
            false
        }
    }

    #[inline]
    pub(crate) fn block_sound(&self, id: BlockId) -> Option<(NonZeroU32, f32)> {
        if id.index() < self.audio_emitters.len() {
            self.audio_emitters[id.index()]
        } else {
            None
        }
    }

    pub(crate) fn block_defs(&self) -> &[Option<BlockTypeDef>] {
        &self.block_defs
    }
}
