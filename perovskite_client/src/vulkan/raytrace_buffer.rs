use crate::client_state::chunk::ChunkDataView;
use crate::client_state::{ChunkManager, ChunkManagerView};
use crate::vulkan::block_renderer::VkChunkRaytraceData;
use crate::vulkan::gpu_chunk_table::{
    gpu_table_lookup, phash, ChunkHashtableBuilder, CHUNK_LEN, CHUNK_LIGHTS_LEN,
    CHUNK_LIGHTS_OFFSET, CHUNK_STRIDE,
};
use crate::vulkan::shaders::raytracer::ChunkMapHeader;
use crate::vulkan::shaders::vert_3d::UniformData;
use crate::vulkan::{gpu_chunk_table, VulkanContext};
use anyhow::Result;
use bytemuck::cast_slice;
use parking_lot::Mutex;
use perovskite_core::coordinates::ChunkCoordinate;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracy_client::span;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::DeviceSize;

#[derive(Clone)]
pub(crate) struct RaytraceSlot {
    pub(crate) header: Subbuffer<ChunkMapHeader>,
    pub(crate) data: Subbuffer<[u32]>,
    serial: u64,
}
impl RaytraceSlot {
    fn build_gpu_slot(
        ctx: &VulkanContext,
        data_len: DeviceSize,
        serial: u64,
    ) -> Result<RaytraceSlot> {
        Ok(RaytraceSlot {
            header: Buffer::new_sized(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )?,
            data: Buffer::new_slice(
                ctx.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                data_len,
            )?,
            serial,
        })
    }
}

pub(crate) struct RaytraceBufferManager {
    prep: Mutex<PrepState>,
    render: Mutex<RenderState>,
    serial: AtomicU64,
    ctx: Arc<VulkanContext>,
}

pub(crate) enum UpdateResult {
    Ok,
    NeedsRebuild,
}

impl RaytraceBufferManager {
    pub(crate) fn new(ctx: Arc<VulkanContext>) -> Result<RaytraceBufferManager> {
        let (empty_data, empty_header) = gpu_chunk_table::build_empty()?;

        let data_len = empty_data.len();
        let gpu_slot0 = RaytraceSlot::build_gpu_slot(&ctx, data_len as DeviceSize, 0)?;
        let gpu_slot1 = RaytraceSlot::build_gpu_slot(&ctx, data_len as DeviceSize, 1)?;

        let cpu_data = Self::make_cpu_subbuffer(&ctx, empty_data)?;

        Ok(RaytraceBufferManager {
            prep: PrepState {
                cpu_data,
                cpu_header: empty_header,
                gpu_slot: None,
                needs_structural_rebuild: true,
                needs_copy: false,
            }
            .into(),
            render: RenderState {
                gpu_slots: [gpu_slot0.into(), gpu_slot1.into()],
                render_slot: 0,
                prepare_slot: 1,
                in_flight_slots: vec![usize::MAX; ctx.swapchain_len],
                chunk_data: FxHashMap::default(),
                prep_slot_dirty_range: (0, data_len),
                render_slot_dirty_range: (0, data_len),
            }
            .into(),
            serial: AtomicU64::new(2),
            ctx,
        })
    }

    fn make_cpu_subbuffer(ctx: &VulkanContext, data: Vec<u32>) -> Result<Subbuffer<[u32]>> {
        let _span = span!("make_cpu_subbuffer");
        let buffer = Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            data.iter().copied(),
        )?;
        Ok(buffer)
    }

    /// Call when done with a batch of changes, doing a rebuild if necessary, and flushing to GPU
    /// if possible. This should be called frequently (from a general tokio thread), cheap if no work
    /// is to be done
    pub(crate) fn batch_done(&self, chunks: &ChunkManager, force_rebuild: bool) -> Result<()> {
        let _span = span!("raytrace batch_done");
        // Lock order: prep -> render
        //
        // Push updates: lock prep, lock render, check in_flight_slots and copy it into our own slot if
        //   safe, then unlock render.
        //   If a rebuild is requested, update the CPU slot.
        // If there's a gpu_slot, copy into it, then lock
        //   render, update render_slot and prepare_slot to reflect the new slot.
        //
        // Rendering: do not lock prep, only lock render. Update in-flight slots based on what
        //   render_slot we gave to the current frame.
        //
        // Safety remarks: all GPU slot accounting happens under the render lock, and we only flip the
        //  render_slot and prepare_slot when holding both the prep lock + render lock. Once a GPU slot
        //  is given to prep, render won't touch it until we flip the slot IDs under both locks, so it's
        //  safe to copy into it.
        //  We'll only give a GPU slot to prep once all swapchain frames that were using it are done.
        //  However, this means that a change (under prep) might only make it to the CPU buffer but not
        //  to a GPU buffer; therefore, we need to schedule a copy/swap operation (preferably not on the
        //  render thread) once the prepare slot is clear to use.
        let mut prep = self.prep.lock();
        let prep = prep.deref_mut();

        if prep.gpu_slot.is_none() {
            let _span = span!("batch_done grab GPU slot");
            // Try to grab a GPU slot from the render state
            let render_lock = self.render.lock();
            let render = render_lock.deref();
            if !render
                .in_flight_slots
                .iter()
                .any(|x| *x == render.prepare_slot)
            {
                prep.gpu_slot = Some(render.gpu_slots[render.prepare_slot].clone());
            }
            drop(render_lock);
        }

        let dirty_view = chunks.take_rt_dirty_chunks();

        let mut dirty_min = usize::MAX;
        let mut dirty_max = usize::MIN;

        if !prep.needs_structural_rebuild {
            let _span = span!("raytrace incremental rebuild");
            // If we're doing a structural rebuild, don't bother with incremental rebuilds
            let mut cpu_buffer = prep
                .cpu_data
                .write()
                .expect("Expected data to be uncontended under the prep lock");
            for (coord, chunk) in dirty_view.iter() {
                let slot = gpu_table_lookup(&cpu_buffer, &prep.cpu_header, *coord);
                let mut chunk_data = chunk.chunk_data_mut();
                if chunk_data.raytrace_data().is_some() {
                    if slot == u32::MAX {
                        // Chunk needs to be inserted
                        prep.needs_structural_rebuild = true;
                        break;
                    } else {
                        let (blocks, lights) = chunk_data
                            .copy_rt_data()
                            .expect("raytrace_data was Some but copy_rt_data wasn't");
                        let n = (prep.cpu_header.n_minus_one + 1) as usize;
                        let data_base = 4 * n + CHUNK_STRIDE * (slot as usize);
                        let light_base =
                            4 * n + CHUNK_STRIDE * (slot as usize) + CHUNK_LIGHTS_OFFSET;
                        cpu_buffer[data_base..data_base + CHUNK_LEN]
                            .copy_from_slice(blocks.deref());
                        cpu_buffer[light_base..light_base + CHUNK_LIGHTS_LEN]
                            .copy_from_slice(cast_slice(&lights));
                        dirty_min = dirty_min.min(data_base);
                        dirty_max = dirty_max.max(light_base + CHUNK_LIGHTS_LEN);
                        prep.needs_copy = true;
                    }
                } else {
                    if slot != u32::MAX {
                        // Chunk needs to be removed
                        prep.needs_structural_rebuild = true;
                        break;
                    } else {
                        // empty chunk, not in hashtable. Nothing to do.
                    }
                }
            }
            drop(cpu_buffer);
        }

        if prep.needs_structural_rebuild || force_rebuild {
            prep.needs_structural_rebuild = false;
            prep.needs_copy = true;
            let view = chunks.renderable_chunks_cloned_view();

            let _span = span!("batch_done structural rebuild");
            let mut builder = ChunkHashtableBuilder::new();
            for (coord, chunk) in view.iter() {
                let mut data = chunk.chunk_data_mut();
                if let Some((blocks, lights)) = data.copy_rt_data() {
                    builder.add_chunk(*coord, blocks, lights);
                }
            }

            let (table, header) = builder.build(20, 3)?;
            prep.cpu_header = header;

            dirty_min = 0;
            dirty_max = table.len();
            if (prep.cpu_data.len() as usize) >= table.len() {
                let _span = span!("Copy into existing buffer");
                // Consider using the whole buffer to reduce collisions; this requires coordinating with the builder.
                let mut cpu_buffer = prep
                    .cpu_data
                    .write()
                    .expect("Expected data to be uncontended under the prep lock");
                cpu_buffer[0..table.len()].copy_from_slice(&table);
            } else {
                prep.cpu_data = Self::make_cpu_subbuffer(&self.ctx, table)?;
            }
        }

        if prep.needs_copy {
            let mut render = self.render.lock();
            let render = render.deref_mut();
            render.prep_slot_dirty_range.0 = render.prep_slot_dirty_range.0.min(dirty_min);
            render.prep_slot_dirty_range.1 = render.prep_slot_dirty_range.1.max(dirty_max);
            render.render_slot_dirty_range.0 = render.render_slot_dirty_range.0.min(dirty_min);
            render.render_slot_dirty_range.1 = render.render_slot_dirty_range.1.max(dirty_max);
            if let Some(mut gpu_slot) = prep.gpu_slot.take() {
                prep.needs_copy = false;
                let _span = span!("batch_done copy");

                assert_eq!(
                    gpu_slot.serial,
                    render.gpu_slots[render.prepare_slot].serial
                );
                let new_len = prep.cpu_data.len();
                if new_len > gpu_slot.data.len() {
                    let _span = span!("Resize GPU buffer");
                    // Do we have a way of easily freeing the previous buffer before building
                    // the new one?
                    gpu_slot = RaytraceSlot::build_gpu_slot(
                        &self.ctx,
                        new_len,
                        self.serial.fetch_add(1, Ordering::Relaxed),
                    )?;
                    render.gpu_slots[render.prepare_slot] = gpu_slot.clone();
                }
                let copy_min = render.prep_slot_dirty_range.0 as DeviceSize;
                let copy_max = render.prep_slot_dirty_range.1 as DeviceSize;
                self.ctx.copy_to_device(
                    prep.cpu_data.clone().slice(copy_min..copy_max),
                    gpu_slot.data.clone().slice(copy_min..copy_max),
                )?;
                *gpu_slot.header.write().expect("Write-lock for header") = prep.cpu_header;
                // And swap the buffers!
                std::mem::swap(&mut render.render_slot, &mut render.prepare_slot);
                render.prep_slot_dirty_range = render.render_slot_dirty_range;
                render.render_slot_dirty_range = (0, new_len as usize);
            }
        }

        Ok(())
    }

    pub(crate) fn acquire(&self, image_i: usize) -> Option<RaytraceSlot> {
        let _span = span!("raytrace acquire");
        let mut render = self.render.lock();
        let render = render.deref_mut();
        let result = render.gpu_slots[render.render_slot].clone();
        render.in_flight_slots[image_i] = render.render_slot;
        Some(result)
    }
}

struct PrepState {
    // A prepared set of CPU buffers that we'll copy into one of the GPU slots
    cpu_data: Subbuffer<[u32]>,
    cpu_header: ChunkMapHeader,
    // If we have a slot to copy into, the slot
    gpu_slot: Option<RaytraceSlot>,
    // We need to rebuild the whole structure from the chunk set
    needs_structural_rebuild: bool,
    // Our staging buffer is good, we just need to copy it
    needs_copy: bool,
}

struct RenderState {
    // The GPU-side slots that we will write into and the shader will read from
    gpu_slots: [RaytraceSlot; 2],
    // The slot that we'll give to running frames
    render_slot: usize,
    // The slot that we'll write to, once safe and enough writes have been coalesced
    prepare_slot: usize,
    // What we know the alternate slot needs to have copied in (we couldn't copy it now)
    render_slot_dirty_range: (usize, usize),
    prep_slot_dirty_range: (usize, usize),
    // Slots that we've given to each frame in the swapchain; used to track when
    // a slot becomes free for write. usize::MAX is used for undefined/none (i.e. at startup)
    in_flight_slots: Vec<usize>,
    // the actual chunk data - a redundant copy, maybe worth optimizing later
    chunk_data: FxHashMap<ChunkCoordinate, VkChunkRaytraceData>,
}
