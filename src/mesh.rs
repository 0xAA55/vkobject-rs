
use crate::prelude::*;
use std::{
	any::{Any, TypeId},
	ffi::c_void,
	fmt::Debug,
	marker::PhantomData,
	mem::size_of,
	sync::Arc,
	vec::IntoIter,
};
use struct_iterable::Iterable;

/// The type that could be the item of the `BufferVec`
pub trait BufferVecStructItem: Copy + Clone + Sized + Default + Debug + Iterable {}
impl<T> BufferVecStructItem for T where T: Copy + Clone + Sized + Default + Debug + Iterable {}

/// A wrapper for `Buffer`
#[derive(Debug, Clone)]
pub struct BufferWithType<T>
where
	T: BufferVecItem {
	/// The buffer
	buffer: Buffer,

	/// The phantom data to hold the type
	_phantom: PhantomData<T>,
}

impl<T> BufferWithType<T>
where
	T: BufferVecItem {
	/// Create the `BufferWithType<T>`
	pub fn new(device: Arc<VulkanDevice>, data: &[T], cmdbuf: VkCommandBuffer, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let ret = Self {
			buffer: Buffer::new(device, (data.len() * size_of::<T>()) as VkDeviceSize, Some(data.as_ptr() as *const c_void), usage)?,
			_phantom: PhantomData,
		};
		ret.upload_staging_buffer(cmdbuf)?;
		Ok(ret)
	}

	/// Create the `BufferWithType<T>`
	pub fn new_empty(device: Arc<VulkanDevice>, size: usize, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		Ok(Self {
			buffer: Buffer::new(device, (size * size_of::<T>()) as VkDeviceSize, None, usage)?,
			_phantom: PhantomData,
		})
	}

	/// Create the `BufferWithType<T>`
	pub fn new_from_buffer(buffer: Buffer) -> Self {
		Self {
			buffer,
			_phantom: PhantomData,
		}
	}

	/// Create staging buffer for the `BufferWithType<T>`
	pub fn ensure_staging_buffer(&mut self) -> Result<(), VulkanError> {
		self.buffer.ensure_staging_buffer()
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&mut self) {
		self.buffer.discard_staging_buffer();
	}

	/// Get data by an index
	pub fn get_data(&self, index: usize) -> Option<T> {
		if let Some(staging_buffer) = &self.buffer.staging_buffer {
			let mut ret = T::default();
			staging_buffer.get_data(&mut ret as *mut T as *mut c_void, (index * size_of::<T>()) as VkDeviceSize, size_of::<T>()).ok()?;
			Some(ret)
		} else {
			None
		}
	}

	/// Set data
	pub fn set_data(&mut self, index: usize, data: T) -> Result<(), VulkanError> {
		self.buffer.set_staging_data(&data as *const T as *const c_void, (index * size_of::<T>()) as VkDeviceSize, size_of::<T>())
	}

	/// Upload staging buffer data to buffer
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.buffer.get_size())
	}

	/// Get the count of the data
	pub fn len(&self) -> usize {
		self.buffer.get_size() as usize / size_of::<T>()
	}

	/// Get if the buffer is empty
	pub fn is_empty(&self) -> bool {
		self.buffer.get_size() == 0
	}

	/// Get the inner buffer
	pub fn into_inner(self) -> Buffer {
		self.buffer
	}
}

/// The trait for the mesh to hold buffers
pub trait BufferForDraw<T>: Debug + Clone
where
	T: BufferVecItem {
	/// Must be able to get the `VkBuffer` handle
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Must be able to flush
	fn flush(&mut self, _cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		Ok(())
	}

	/// Convert to `BufferVec<T>`
	fn convert_to_buffer_vec(self) -> BufferVec<T>;

	/// Convert to `BufferWithType<T>`
	fn convert_to_buffer_with_type(self) -> BufferWithType<T>;
}

impl<T> BufferForDraw<T> for BufferVec<T>
where
	T: BufferVecItem {
	fn get_vk_buffer(&self) -> VkBuffer {
		self.get_vk_buffer()
	}

	fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.flush(cmdbuf)
	}

	fn convert_to_buffer_vec(self) -> BufferVec<T> {
		self
	}

	fn convert_to_buffer_with_type(self) -> BufferWithType<T> {
		BufferWithType::new_from_buffer(self.into_inner())
	}
}

impl<T> BufferForDraw<T> for BufferWithType<T>
where
	T: BufferVecItem {
	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.upload_staging_buffer(cmdbuf)?;
		self.discard_staging_buffer();
		Ok(())
	}

	fn convert_to_buffer_vec(self) -> BufferVec<T> {
		let len = self.len();
		unsafe {BufferVec::from_raw_parts(self.into_inner(), len).unwrap()}
	}

	fn convert_to_buffer_with_type(self) -> BufferWithType<T> {
		self
	}
}

#[derive(Debug, Clone)]
pub struct Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecStructItem,
	E: BufferVecItem + 'static,
	I: BufferVecStructItem,
	C: BufferVecStructItem {
	pub primitive_type: VkPrimitiveTopology,
	pub vertices: BV,
	pub indices: Option<BE>,
	pub instances: Option<BI>,
	pub commands: Option<BC>,
	vertex_type: V,
	element_type: E,
	instance_type: I,
	command_type: C,
}

/// If a buffer you don't need, use this for your buffer item type
#[derive(Default, Debug, Clone, Copy, Iterable)]
pub struct UnusedBufferItem {}

/// If a buffer you don't need, use this for your buffer type
pub type UnusedBufferType = BufferWithType<UnusedBufferItem>;

/// Use this function to create an unused buffer type
pub fn buffer_unused() -> Option<UnusedBufferType> {
	None
}

impl<BV, V, BE, E, BI, I, BC, C> Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecStructItem,
	E: BufferVecItem + 'static,
	I: BufferVecStructItem,
	C: BufferVecStructItem {
	pub fn new(primitive_type: VkPrimitiveTopology, vertices: BV, indices: Option<BE>, instances: Option<BI>, commands: Option<BC>) -> Self {
		Self {
			primitive_type,
			vertices,
			indices,
			instances,
			commands,
			vertex_type: V::default(),
			element_type: E::default(),
			instance_type: I::default(),
			command_type: C::default(),
		}
	}

	pub fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.vertices.flush(cmdbuf)?;
		if let Some(ref mut indices) = self.indices {indices.flush(cmdbuf)?;}
		if let Some(ref mut instances) = self.instances {instances.flush(cmdbuf)?;}
		if let Some(ref mut commands) = self.commands {commands.flush(cmdbuf)?;}
		Ok(())
	}
}

/// The most typical static mesh type: use `BufferWithType` for vertices and elements(indices), use `BufferVec` for instances and draw commands
pub type StaticMesh<V, E, I, C> = Mesh<BufferWithType<V>, V, BufferWithType<E>, E, BufferVec<I>, I, BufferVec<C>, C>;

/// The dynamic mesh type: use `BufferVec` for all buffers
pub type DynamicMesh<V, E, I, C> = Mesh<BufferVec<V>, V, BufferVec<E>, E, BufferVec<I>, I, BufferVec<C>, C>;

/// The trait for a mesh
pub trait GenericMesh: Debug {
	/// Get the vertex buffer
	fn get_vk_vertex_buffer(&self) -> VkBuffer;

	/// Get the index buffer
	fn get_vk_index_buffer(&self) -> Option<VkBuffer>;

	/// Get the instance buffer
	fn get_vk_instance_buffer(&self) -> Option<VkBuffer>;

	/// Get the command buffer
	fn get_vk_command_buffer(&self) -> Option<VkBuffer>;

	/// Get the primitive type
	fn get_primitive_type(&self) -> VkPrimitiveTopology;

	/// Get the iterator for the vertex buffer item structure
	fn iter_vertex_buffer_struct_members(&self) -> IntoIter<(&'static str, &(dyn Any + 'static))>;

	/// Get the TypeId of the index buffer item
	fn get_index_type_id(&self) -> Option<TypeId>;

	/// Get the iterator for the vertex buffer item structure
	fn iter_instance_buffer_struct_members(&self) -> Option<IntoIter<(&'static str, &(dyn Any + 'static))>>;

	/// Get the iterator for the vertex buffer item structure
	fn iter_command_buffer_struct_members(&self) -> Option<IntoIter<(&'static str, &(dyn Any + 'static))>>;

	/// Get the stride of the vertex buffer
	fn get_vertex_stride(&self) -> usize;

	/// Get the stride of the index buffer
	fn get_index_stride(&self) -> usize;

	/// Get the stride of the instance buffer
	fn get_instance_stride(&self) -> usize;

	/// Get the stride of the command buffer
	fn get_command_stride(&self) -> usize;
}

impl<BV, V, BE, E, BI, I, BC, C> GenericMesh for Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecStructItem,
	E: BufferVecItem + 'static,
	I: BufferVecStructItem,
	C: BufferVecStructItem {
	fn get_vk_vertex_buffer(&self) -> VkBuffer {
		self.vertices.get_vk_buffer()
	}

	fn get_vk_index_buffer(&self) -> Option<VkBuffer> {
		self.indices.as_ref().map(|b|b.get_vk_buffer())
	}

	fn get_vk_instance_buffer(&self) -> Option<VkBuffer> {
		self.instances.as_ref().map(|b|b.get_vk_buffer())
	}

	fn get_vk_command_buffer(&self) -> Option<VkBuffer> {
		self.commands.as_ref().map(|b|b.get_vk_buffer())
	}

	fn get_primitive_type(&self) -> VkPrimitiveTopology {
		self.primitive_type
	}

	fn iter_vertex_buffer_struct_members(&self) -> IntoIter<(&'static str, &(dyn Any + 'static))> {
		self.vertex_type.iter()
	}

	fn get_index_type_id(&self) -> Option<TypeId> {
		self.indices.as_ref().map(|_|self.element_type.type_id())
	}

	fn iter_instance_buffer_struct_members(&self) -> Option<IntoIter<(&'static str, &(dyn Any + 'static))>> {
		self.instances.as_ref().map(|_|self.instance_type.iter())
	}

	fn iter_command_buffer_struct_members(&self) -> Option<IntoIter<(&'static str, &(dyn Any + 'static))>> {
		self.commands.as_ref().map(|_|self.command_type.iter())
	}

	fn get_vertex_stride(&self) -> usize {
		size_of::<V>()
	}

	fn get_index_stride(&self) -> usize {
		size_of::<E>()
	}

	fn get_instance_stride(&self) -> usize {
		size_of::<I>()
	}

	fn get_command_stride(&self) -> usize {
		size_of::<C>()
	}
}

/// A Mesh with a material
#[derive(Debug)]
pub struct GenericMeshWithMaterial {
	/// The mesh
	pub mesh: Box<dyn GenericMesh>,

	/// The material
	pub material: Box<dyn Material>,
}

impl GenericMeshWithMaterial {
	/// Create an instance for the `GenericMeshWithMaterial`
	pub fn new(mesh: Box<dyn GenericMesh>, material: Box<dyn Material>) -> Self {
		Self {
			mesh,
			material,
		}
	}
}

