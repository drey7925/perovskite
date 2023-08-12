use std::{
    ops::{ControlFlow, DerefMut},
    sync::Arc,
};

use crate::{
    block_renderer::{VkChunkPass, VkChunkVertexData},
    game_state::chunk::ClientChunk,
    vulkan::shaders::cube_geometry::CubeGeometryVertex,
};
use anyhow::Result;
use cgmath::{vec3, ElementWise, Matrix4, Vector3};
use cuberef_core::coordinates::ChunkCoordinate;
use futures::future::ready;
use parking_lot::{Mutex, MutexGuard, RwLock};
use rustc_hash::FxHashMap;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{DrawIndexedIndirectCommand, DrawIndirectCommand},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
    },
};

use super::shaders::cube_geometry::{BlockRenderPass, CubeGeometryDrawCall};

/// An immutable on-GPU buffer for vertex data
pub(crate) struct ManagedBuffer {
    anchor_position: cgmath::Vector3<f64>,
    // The actual vertex data sent to the GPU
    vertices: Subbuffer<[CubeGeometryVertex]>,
    // The indices for draw_indexed
    indices: Subbuffer<[u32]>,
    // The chunks that are backing this buffer
    backing_chunks: Vec<Arc<ClientChunk>>,
}

struct ManagedBufferBuilder {
    vertices: Vec<CubeGeometryVertex>,
    indices: Vec<u32>,
    backing_chunks: Vec<Arc<ClientChunk>>,
    anchor_position: Option<cgmath::Vector3<f64>>,
    allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
    // (offset, len) pairs
    chunk_slices: FxHashMap<ChunkCoordinate, (usize, usize)>,
    // This builder will go into the indicated slot in the buffer list
    upcoming_buffer_id: usize,
}

impl ManagedBufferBuilder {
    fn new(
        allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
        upcoming_id: usize,
    ) -> ManagedBufferBuilder {
        ManagedBufferBuilder {
            vertices: Vec::new(),
            indices: Vec::new(),
            backing_chunks: Vec::new(),
            anchor_position: None,
            allocator,
            chunk_slices: FxHashMap::default(),
            upcoming_buffer_id: upcoming_id,
        }
    }
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn build(self) -> Result<ManagedBuffer> {
        let vertices = Buffer::from_iter(
            &self.allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            self.vertices.into_iter(),
        )?;
        let indices = Buffer::from_iter(
            &self.allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            self.indices.into_iter(),
        )?;
        Ok(ManagedBuffer {
            anchor_position: self.anchor_position.unwrap(),
            vertices,
            indices,
            backing_chunks: self.backing_chunks,
        })
    }

    fn push(
        &mut self,
        pass_data: &crate::block_renderer::IndexedVertexBuffer,
        chunk: Arc<ClientChunk>,
    ) {
        if self.anchor_position.is_none() {
            self.anchor_position = Some(chunk.origin());
        }

        let position_adjustment: Vector3<f32> = (chunk.origin() - self.anchor_position.unwrap())
            .cast()
            .unwrap();
        self.vertices
            .extend(pass_data.vtx.iter().map(|v| CubeGeometryVertex {
                position: [
                    v.position[0] + position_adjustment.x,
                    v.position[1] + position_adjustment.y,
                    v.position[2] + position_adjustment.z,
                ],
                ..*v
            }));
        let index_offset = self.vertices.len() as u32;
        self.indices
            .extend(pass_data.idx.iter().map(|x| x + index_offset));

        self.backing_chunks.push(chunk);
    }

    fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.backing_chunks.clear();
        self.anchor_position = None;
        self.chunk_slices.clear();
    }
}

pub(crate) struct BufferManager {
    allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
    builder: Mutex<ManagedBufferBuilder>,
    // lock orders: ready_buffers can be read-locked at any time
    // To write, first lock the builder, then write-lock ready_buffers
    ready_buffers: RwLock<Vec<Option<Arc<ManagedBuffer>>>>,
    pass: BlockRenderPass,
}
impl BufferManager {
    pub(crate) fn new(
        allocator: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
        pass: BlockRenderPass,
    ) -> Self {
        BufferManager {
            allocator: allocator.clone(),
            builder: Mutex::new(ManagedBufferBuilder::new(allocator, 0)),
            ready_buffers: RwLock::new(Vec::new()),
            pass,
        }
    }

    pub(crate) fn get_for_rendering(
        &self,
        player_position: cgmath::Vector3<f64>,
    ) -> Vec<CubeGeometryDrawCall> {
        self.ready_buffers
            .read()
            .iter()
            .filter_map(|x| {
                x.as_ref().map(|x| {
                    let offset =
                        x.anchor_position - (player_position.mul_element_wise(vec3(1., -1., 1.)));
                    CubeGeometryDrawCall {
                        models: VkChunkPass {
                            vtx: x.vertices.clone(),
                            idx: x.indices.clone(),
                        },
                        model_matrix: Matrix4::from_translation(offset.cast().unwrap()),
                    }
                })
            })
            .collect()
    }

    pub(crate) fn add_chunk(&self, chunk: &Arc<ClientChunk>) -> Result<()> {
        let mut builder = self.builder.lock();
        let buffer_id = chunk
            .assigned_buffer
            .load(std::sync::atomic::Ordering::SeqCst);
        // log::info!("chunk {:?} had buffer {}", chunk.coord(), buffer_id as isize);
        match buffer_id {
            usize::MAX => {
                // The chunk isn't in a buffer yet
                self.add_chunk_locked(chunk, &mut builder)
            }
            x if x == builder.upcoming_buffer_id => {
                // log::info!("Builder at {} is being drained", builder.upcoming_buffer_id);
                // the chunk is in the current builder. Delete and rebuild the whole builder.
                let chunks: Vec<_> = builder.backing_chunks.drain(..).collect();
                builder.clear();
                assert!(chunks.iter().find(|x| x.coord() == chunk.coord()).is_some());
                for chunk in chunks {
                    self.add_chunk_locked(&chunk, &mut builder);
                }
            }
            x => {
                // This chunk was already in a buffer, and we need to clean up that prior buffer
                // TODO: see if we can reuse the buffer it was in before (if the size matches)
                // For now, we need to create a completely new buffer, and find a new home for the other
                // chunks in the buffer
                //
                // If this unwrap panics, an invariant is broken - a chunk's buffer_id should always point to a live buffer
                // log::info!("buffer at {} is being drained", x as isize);
                let mut ready_buffers = self.ready_buffers.write();
                // log::info!("debug bitmap before taking: {}", debug_bitmap(&ready_buffers));
                let removed = ready_buffers[x].take().unwrap();
                assert!(removed
                    .backing_chunks
                    .iter()
                    .find(|x| x.coord() == chunk.coord())
                    .is_some());
                for chunk in removed.backing_chunks.iter() {
                    // Every chunk, including the one we're currently dealing with, needs to be added to the builder
                    self.add_chunk_locked(chunk, &mut builder);
                }
            }
        }
        if builder.len() > BUILDER_SIZE_THRESHOLD {
            self.promote_builder_locked(&mut builder)?;
        }
        Ok(())
    }

    pub(crate) fn flush(&self) -> Result<()> {
        let mut builder = self.builder.lock();
        self.promote_builder_locked(&mut builder)?;
        Ok(())
    }

    fn add_chunk_locked(
        &self,
        chunk: &Arc<ClientChunk>,
        builder: &mut MutexGuard<'_, ManagedBufferBuilder>,
    ) {
        let mut mesh_data = chunk.cached_vertex_data.lock();
        let data = match mesh_data.as_mut() {
            Some(data) => data,
            None => {
                
                chunk.assigned_buffer.store(
                    // This chunk has no data, so it won't go to a buffer
                    usize::MAX,
                    std::sync::atomic::Ordering::SeqCst,
                );
                return;
            },
        };
        let pass_data = match self.pass {
            BlockRenderPass::Opaque => &data.solid_opaque,
            BlockRenderPass::Transparent => &data.transparent,
            BlockRenderPass::Translucent => &data.translucent,
        };
        let pass_data = match pass_data {
            Some(pass_data) => pass_data,
            None => {
                chunk.assigned_buffer.store(
                    // This chunk has no data, so it won't go to a buffer
                    usize::MAX,
                    std::sync::atomic::Ordering::SeqCst,
                );
                return;
            },
        };
        builder.push(pass_data, chunk.clone());
        // log::info!("chunk {:?} added to buffer {}", chunk.coord(), builder.upcoming_buffer_id);
        chunk.assigned_buffer.store(
            builder.upcoming_buffer_id,
            std::sync::atomic::Ordering::SeqCst,
        );
    }


    fn promote_builder_locked(
        &self,
        builder_lock: &mut MutexGuard<'_, ManagedBufferBuilder>,
    ) -> Result<()> {
        if builder_lock.len() == 0 {
            return Ok(());
        }
        // log::info!("promoting buffer {}", builder_lock.upcoming_buffer_id);
        // log::info!("Debug bitmap before promotion: {}", debug_bitmap(&self.ready_buffers.read()));
        let mut ready_buffers = self.ready_buffers.write();
        // scan to find the slot that we'll use for the *next* buffer
        let fallback_next_slot = if builder_lock.upcoming_buffer_id == ready_buffers.len() {
            // The current buffer will append; the next buffer should append beyond that
            ready_buffers.len() + 1
        } else {
            // The current buffer is using an existing empty slot; if the next buffer will get appended, it should
            // be appended based on the *current* length of the list, which will not change.
            ready_buffers.len()
        };
        let next_slot = ready_buffers
            .iter()
            .enumerate()
            .find(|(i, x)| *i != builder_lock.upcoming_buffer_id && x.is_none())
            .map_or(fallback_next_slot, |(i, _)| i);
        // log::info!("Next slot will be {}", next_slot);
        let mut builder = ManagedBufferBuilder::new(self.allocator.clone(), next_slot);
        std::mem::swap(builder_lock.deref_mut(), &mut builder);
        let current_slot = builder.upcoming_buffer_id;
        let buffer = Arc::new(builder.build()?);
        if current_slot == ready_buffers.len() {
            // log::info!("Appending new buffer");
            ready_buffers.push(Some(buffer));
        } else {
            // log::info!("Replacing empty buffer at slot {}", current_slot);
            assert!(ready_buffers[current_slot].is_none());
            ready_buffers[current_slot] = Some(buffer);
        }
        
        // log::info!("Debug bitmap after promotion: {}", debug_bitmap(&ready_buffers));
        Ok(())
    }

    pub(crate) fn remove(&self, removed_chunk: &Arc<ClientChunk>) -> Result<()> {
        let mut builder = self.builder.lock();
        let buffer_id = removed_chunk
            .assigned_buffer
            .load(std::sync::atomic::Ordering::SeqCst);
        match buffer_id {
            usize::MAX => {
                // nothing to do
            }
            x if x == self.ready_buffers.read().len() => {
                // the chunk is in the current builder. Delete and rebuild the whole builder.
                let chunks: Vec<_> = builder.backing_chunks.drain(..).collect();
                builder.clear();
                assert!(chunks
                    .iter()
                    .find(|x| x.coord() == removed_chunk.coord())
                    .is_some());
                for chunk in chunks {
                    if chunk.coord() != removed_chunk.coord() {
                        self.add_chunk_locked(&chunk, &mut builder);
                    }
                }
            }
            x => {
                // This chunk was already in a buffer, and we need to clean up that prior buffer
                // We need to create a completely new buffer, and find a new home for the other
                // chunks in the buffer
                //
                // If this unwrap panics, an invariant is broken - a chunk's buffer_id should always point to a live buffer
                let removed_buffer = self.ready_buffers.write()[x].take().unwrap();
                assert!(removed_buffer
                    .backing_chunks
                    .iter()
                    .find(|x| x.coord() == removed_chunk.coord())
                    .is_some());
                for chunk in removed_buffer.backing_chunks.iter() {
                    // Every other chunk needs to be added to the builder
                    if chunk.coord() != removed_chunk.coord() {
                        self.add_chunk_locked(chunk, &mut builder);
                    }
                }
            }
        }
        Ok(())
    }
}


fn debug_bitmap<T>(contents: &[Option<T>]) -> String {
    "[".to_string() + contents.iter().map(|x| {
        if x.is_some() {
            'X'
        } else {
            ' '
        }
    }).collect::<String>().as_str() + "]"
}

const BUILDER_SIZE_THRESHOLD: usize = 32768;
