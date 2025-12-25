use crate::client_state::ChunkMap;
use crate::vulkan::gpu_chunk_table::{
    build_chunk_hashtable, gpu_table_lookup, CHUNK_LEN, CHUNK_LIGHTS_LEN, CHUNK_LIGHTS_OFFSET,
    CHUNK_STRIDE,
};
use crate::vulkan::shaders::raytracer::ChunkMapHeader;
use crate::vulkan::{ReclaimType, ReclaimableBuffer, VulkanContext};
use anyhow::Result;
use bytemuck::cast_slice;
use parking_lot::Mutex;
use perovskite_core::coordinates::ChunkCoordinate;
use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;
use std::sync::Arc;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::DeviceSize;
// Old abandoned approach: https://gist.github.com/drey7925/d75e3b665943e2332f7f3dd0db70b76a

// 7328 ints per entry, 286 entries, to get us just under 8 MiB. Round up to 8 MiB to eventually
// allow for metadata updates
const INCREMENTAL_STAGING_BUFFER_CAPACITY: DeviceSize = 2097152;
// 7328 ints per entry, 2289 entries, 64 MiB. We will leisurely copy this buffer off the render
// thread
const LOCAL_CATCHUP_STAGING_BUFFER_CAPACITY: DeviceSize = 16567782;

struct RaytraceBufferState {
    // The current set of changes, expected to be sent in the next frame
    incremental: UpdateBuilder,
    // If there is a rebuild in progress, the set of incremental changes that will need to be caught
    // up after that rebuild is done
    catchup: Option<UpdateBuilder>,
    rebuild_in_progress: bool,
    new_buffer: Option<RaytraceBuffer>,
}

/// Actions that the render thread should take. In principle, multiple of these actions can happen
/// in order; this will typically be returned in a tinyvec/smallvec to allow multiple actions to be
/// queued up. The render thread can make a simplifying assumption
pub(crate) enum RenderThreadAction {
    /// A buffer copy should be done, from the staging buffer to the render buffer, using the
    /// provided region list
    Incremental {
        staging_buffer: ReclaimableBuffer<u32>,
        scatter_gather_list: SmallVec<[BufferCopy; 8]>,
    },
}

pub(crate) struct RaytraceBuffer {
    /// buffer, in GPU memory, appropriate for SSBO binding
    pub(crate) data: ReclaimableBuffer<u32>,
    /// buffer, in GPU memory, appropriate for uniform binding
    pub(crate) header: Subbuffer<ChunkMapHeader>,
}

pub(crate) struct RtFrameData {
    pub(crate) new_buffer: Option<RaytraceBuffer>,
    pub(crate) update_steps: SmallVec<[RenderThreadAction; 1]>,
}

pub(crate) struct RaytraceBufferManager {
    state: Mutex<RaytraceBufferState>,
    vk_ctx: Arc<VulkanContext>,
}
impl RaytraceBufferManager {
    pub(crate) fn new(vk_ctx: Arc<VulkanContext>) -> Result<Self> {
        let incremental =
            UpdateBuilder::with_capacity(&vk_ctx, INCREMENTAL_STAGING_BUFFER_CAPACITY, None)?;
        Ok(Self {
            state: Mutex::new(RaytraceBufferState {
                incremental,
                catchup: None,
                rebuild_in_progress: false,
                new_buffer: None,
            }),
            vk_ctx,
        })
    }

    pub(crate) fn do_rebuild(&self, chunks: ChunkMap) -> Result<()> {
        // Catchup works as:
        // * Steady-state, only incremental buffer, no catchup buffer. During each frame we accumulate
        //     chunk updates into the incremental buffer, and on each acquire() we drain them to the render
        //     thread to apply them to the main render buffer.
        // * A rebuild is requested. Large catchup buffer enabled. Incoming chunk deltas are written to both
        //     the incremental buffer (drained on each frame as before) and the catchup buffer.
        // * IF the catchup buffer has more 1.5x incremental buffer worth of work (or if force-enabled for
        //     testing) [NOT IMPLEMENTED YET, TODO EVALUATE IF NEEDED]
        //     * The rebuild completes, the catchup buffer is taken and replaced with a small catchup buffer
        //       in its place. Incoming chunk deltas are written to the incremental buffer as before,
        //       applied by render thread on each frame as before. During this time, we again accumulate
        //       into both buffers. The preceding large catchup buffer is applied to the nascent GPU buffer
        //       on the catchup thread (and on the bulk transfer queue)
        // * The incremental buffer continues recording changes not yet in the nascent buffer. When the
        //     render thread calls acquire(), it will receive both the new buffer and an incremental buffer
        //     that it must apply.
        {
            let mut state = self.state.lock();
            if state.rebuild_in_progress {
                return Ok(());
            }
            state.rebuild_in_progress = true;
            assert!(
                state.catchup.is_none(),
                "catchup buffer present unexpectedly"
            );
            state.catchup = Some(UpdateBuilder::with_capacity(
                &self.vk_ctx,
                LOCAL_CATCHUP_STAGING_BUFFER_CAPACITY,
                // We haven't built the hashtable, so we don't know how coordinates will be hashed
                None,
            )?);
        }

        let (table, header) = {
            let _span = span!("rebuild hashtable");
            build_chunk_hashtable(chunks, 30, 3)
        };
        let table_control = table[0..(4 * (header.n_minus_one as usize + 1))].to_vec();
        let data_len = table.len() as DeviceSize;
        let table_gpubuf = self
            .vk_ctx
            .iter_to_device_via_staging_with_reclaim_and_flush(
                table.into_iter(),
                ReclaimType::GpuSsboTransferDst,
                self.vk_ctx.u32_reclaimer().clone(),
                data_len,
            )?;
        let header_gpubuf = Buffer::from_data(
            self.vk_ctx.clone_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            header.clone(),
        )?;

        {
            let mut state = self.state.lock();
            state.new_buffer = Some(RaytraceBuffer {
                data: table_gpubuf,
                header: header_gpubuf,
            });
            state.incremental = state.catchup.take().unwrap();
            state.incremental.inject_lookup_data(header, table_control);
            state.rebuild_in_progress = false;
        }
        Ok(())
    }

    pub(crate) fn push_chunk(
        &self,
        coord: ChunkCoordinate,
        blocks: Option<&[u32; 5832]>,
        lights: Option<&[u8; 5832]>,
    ) -> Result<()> {
        let mut state = self.state.lock();

        if let Some(catchup) = state.catchup.as_mut() {
            catchup.append_chunk(coord, blocks, lights)?;
        }

        state.incremental.append_chunk(coord, blocks, lights)?;

        Ok(())
    }

    pub(crate) fn acquire(&self) -> Result<RtFrameData> {
        let _span = span!("raytrace acquire");
        let mut state = self.state.lock();
        let mut update_steps = SmallVec::new();

        if let Some(step) = state
            .incremental
            .reset_with_capacity(&self.vk_ctx, INCREMENTAL_STAGING_BUFFER_CAPACITY)?
        {
            update_steps.push(step);
        }
        Ok(RtFrameData {
            new_buffer: state.new_buffer.take(),
            update_steps,
        })
    }
}

#[derive(Copy, Clone, Debug)]
struct UnresolvedCopy {
    src_pos: usize,
    len: usize,
    coord: ChunkCoordinate,
    lights_only: bool,
}

struct UpdateBuilder {
    staging_buffer: ReclaimableBuffer<u32>,
    resolved_sg_list: SmallVec<[BufferCopy; 8]>,
    // src pos (bytes), len (bytes), coordinate (used to compute dst pos)
    unresolved_sg_list: SmallVec<[UnresolvedCopy; 8]>,

    lookup_data: Option<(ChunkMapHeader, Vec<u32>)>,

    // Offset, in u32s, of the next write to land in the staging buffer
    append_pos: usize,
    capacity: usize,
    // Used to deduplicate writes to the staging buffer
    staging_buffer_locations_blocks: FxHashMap<ChunkCoordinate, usize>,
    staging_buffer_locations_lights: FxHashMap<ChunkCoordinate, usize>,
}

impl UpdateBuilder {
    pub(crate) fn append_chunk(
        &mut self,
        coord: ChunkCoordinate,
        blocks: Option<&[u32; 5832]>,
        lights: Option<&[u8; 5832]>,
    ) -> Result<()> {
        let _span = span!("UpdateBuilder append_chunk");

        let mut needed_len = 0;
        let blocks_pos = match self.staging_buffer_locations_blocks.get(&coord).copied() {
            None => {
                needed_len += CHUNK_LIGHTS_OFFSET;
                None
            }
            Some(x) => Some(x),
        };
        let lights_pos = match self.staging_buffer_locations_lights.get(&coord).copied() {
            None => {
                needed_len += CHUNK_LIGHTS_LEN;
                None
            }
            Some(x) => Some(x),
        };
        if self.append_pos + needed_len > self.capacity {
            // TODO: This should be handled by smartly prioritizing nearby chunks until the next
            // rebuild when space runs low
            // bail!(
            //     "Over capacity, need {}, have {}",
            //     self.append_pos + needed_len,
            //     self.capacity
            // );
            return Ok(());
        }
        let (blocks_pos, blocks_copy_len) = match blocks_pos {
            Some(x) => (x, None),
            None => {
                if blocks.is_some() {
                    let current_pos = self.append_pos;
                    // Extra 24 bytes of wasted padding data in cases where lights aren't being
                    // updated. The bookkeeping cost makes it not worth it to chase these 24 bytes
                    self.append_pos += CHUNK_LIGHTS_OFFSET;
                    self.staging_buffer_locations_blocks
                        .insert(coord, current_pos);
                    (current_pos, Some(CHUNK_LIGHTS_OFFSET))
                } else {
                    (0, None)
                }
            }
        };
        let (lights_pos, lights_copy_len) = match lights_pos {
            Some(x) => (x, None),
            None => {
                if lights.is_some() {
                    let current_pos = self.append_pos;
                    self.append_pos += CHUNK_LIGHTS_LEN;
                    self.staging_buffer_locations_lights
                        .insert(coord, current_pos);
                    (current_pos, Some(CHUNK_LIGHTS_LEN))
                } else {
                    (0, None)
                }
            }
        };

        {
            let mut guard = self.staging_buffer.write()?;
            if let Some(blocks) = blocks {
                guard[blocks_pos..blocks_pos + CHUNK_LEN].copy_from_slice(blocks);
            }
            if let Some(lights) = lights {
                guard[lights_pos..lights_pos + CHUNK_LIGHTS_LEN]
                    .copy_from_slice(cast_slice(lights));
            }
        }

        let unresolved = match (blocks_copy_len, lights_copy_len) {
            (Some(block_len), Some(lights_len)) => {
                assert_eq!(lights_pos, blocks_pos + block_len);
                Some(UnresolvedCopy {
                    src_pos: blocks_pos * 4,
                    len: (block_len + lights_len) * 4,
                    coord,
                    lights_only: false,
                })
            }
            (Some(block_len), None) => Some(UnresolvedCopy {
                src_pos: blocks_pos * 4,
                len: block_len * 4,
                coord,
                lights_only: false,
            }),
            (None, Some(lights_len)) => Some(UnresolvedCopy {
                src_pos: lights_pos * 4,
                len: lights_len * 4,
                coord,
                lights_only: true,
            }),
            (None, None) => None,
        };
        if let Some(unresolved) = unresolved {
            if let Some((header, control_block)) = &self.lookup_data {
                if let Some(entry) = Self::make_sg_entry(header, control_block, unresolved) {
                    self.resolved_sg_list.push(entry)
                }
            } else {
                self.unresolved_sg_list.push(unresolved);
            }
        }

        Ok(())
    }
    pub(crate) fn inject_lookup_data(&mut self, header: ChunkMapHeader, control_block: Vec<u32>) {
        for unresolved in self.unresolved_sg_list.drain(..) {
            if let Some(entry) = Self::make_sg_entry(&header, &control_block, unresolved) {
                self.resolved_sg_list.push(entry);
            }
        }

        self.lookup_data = Some((header, control_block));
    }

    fn make_sg_entry(
        header: &ChunkMapHeader,
        control_block: &Vec<u32>,
        unresolved_copy: UnresolvedCopy,
    ) -> Option<BufferCopy> {
        let slot = gpu_table_lookup(&control_block, &header, unresolved_copy.coord);
        if slot == u32::MAX {
            None
        } else {
            let mut dst_offset_ints =
                ((header.n_minus_one as usize + 1) * 4) + (slot as usize * CHUNK_STRIDE);
            if unresolved_copy.lights_only {
                dst_offset_ints += CHUNK_LIGHTS_OFFSET;
            }
            Some(BufferCopy {
                src_offset: unresolved_copy.src_pos as DeviceSize,
                dst_offset: (dst_offset_ints * 4) as DeviceSize,
                size: unresolved_copy.len as DeviceSize,
                ..Default::default()
            })
        }
    }
    fn with_capacity(
        vk_ctx: &VulkanContext,
        capacity: DeviceSize,
        lookup_data: Option<(ChunkMapHeader, Vec<u32>)>,
    ) -> Result<UpdateBuilder> {
        let staging_buffer = vk_ctx.u32_reclaimer.take_or_create_slice(
            &vk_ctx,
            ReclaimType::CpuTransferSrc,
            capacity,
        )?;
        Ok(UpdateBuilder {
            staging_buffer,
            resolved_sg_list: smallvec![],
            unresolved_sg_list: smallvec![],
            append_pos: 0,
            capacity: capacity as usize,

            staging_buffer_locations_blocks: Default::default(),

            staging_buffer_locations_lights: Default::default(),
            lookup_data,
        })
    }

    fn reset_with_capacity(
        &mut self,
        vk_ctx: &VulkanContext,
        capacity: DeviceSize,
    ) -> Result<Option<RenderThreadAction>> {
        if !self.unresolved_sg_list.is_empty() {
            return Ok(None);
        }
        let staging_buffer = std::mem::replace(
            &mut self.staging_buffer,
            vk_ctx.u32_reclaimer.take_or_create_slice(
                &vk_ctx,
                ReclaimType::CpuTransferSrc,
                capacity,
            )?,
        );
        let sg_list = std::mem::replace(&mut self.resolved_sg_list, smallvec![]);
        self.capacity = capacity as usize;
        self.append_pos = 0;
        self.staging_buffer_locations_blocks.clear();
        self.staging_buffer_locations_lights.clear();
        if sg_list.is_empty() {
            Ok(None)
        } else {
            Ok(Some(RenderThreadAction::Incremental {
                staging_buffer,
                scatter_gather_list: sg_list,
            }))
        }
    }
}
