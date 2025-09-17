
use crate::prelude::*;
use struct_iterable::Iterable;

#[derive(Debug, Clone)]
pub struct Mesh<V: BufferVecItem, E: BufferVecItem, I: BufferVecItem, C: BufferVecItem> {
	pub primitive_type: VkPrimitiveTopology,
	pub vertices: BufferVec<V>,
	pub indices: Option<BufferVec<E>>,
	pub instances: Option<BufferVec<I>>,
	pub commands: Option<BufferVec<C>>,
}

#[derive(Default, Debug, Clone, Copy, Iterable)]
pub struct UnusedBufferItem {}

pub type UnusedBufferType = BufferVec<UnusedBufferItem>;

pub fn buffer_unused() -> Option<UnusedBufferType> {
	None
}

impl<V, E, I, C> Mesh<V, E, I, C>
where
	V: BufferVecItem,
	E: BufferVecItem,
	I: BufferVecItem,
	C: BufferVecItem {
	pub fn new(primitive_type: VkPrimitiveTopology, vertices: BufferVec<V>, indices: Option<BufferVec<E>>, instances: Option<BufferVec<I>>, commands: Option<BufferVec<C>>) -> Self {
		Self {
			primitive_type,
			vertices,
			indices,
			instances,
			commands,
		}
	}

	pub fn strip_vertices_staging_buffer(&mut self) -> Result<(), VulkanError> {
		unsafe {self.vertices.change_capacity(0)}
	}
}
