
use crate::prelude::*;
use std::{
	any::{Any, TypeId, type_name},
	collections::BTreeMap,
	ffi::c_void,
	fmt::Debug,
	marker::PhantomData,
	mem::{size_of, size_of_val},
	path::Path,
	ptr::{copy, null},
	slice,
	sync::{Arc, RwLock, RwLockWriteGuard},
	vec::IntoIter,
};
use struct_iterable::Iterable;

/// The type that could be the item of the `BufferVec`
pub trait BufferVecStructItem: Clone + Copy + Sized + Default + Debug + Iterable + Any + 'static {}
impl<T> BufferVecStructItem for T where T: Clone + Copy + Sized + Default + Debug + Iterable + Any + 'static {}

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
	/// * `cmdbuf`: Could be `null`, thus the staging buffer would not be uploaded immediately.
	pub fn new(device: Arc<VulkanDevice>, data: &[T], cmdbuf: VkCommandBuffer, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let ret = Self {
			buffer: Buffer::new(device, size_of_val(data) as VkDeviceSize, Some(data.as_ptr() as *const c_void), usage)?,
			_phantom: PhantomData,
		};
		if !cmdbuf.is_null() {
			ret.upload_staging_buffer(cmdbuf)?;
		}
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

	/// Get the device
	pub fn get_device(&self) -> Arc<VulkanDevice> {
		self.buffer.device.clone()
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
	pub fn get_item(&mut self, index: usize) -> Option<T> {
		if let Some(ref mut staging_buffer) = self.buffer.staging_buffer {
			if index >= staging_buffer.get_size() as usize / size_of::<T>() {
				None
			} else {
				let mut ret = T::default();
				staging_buffer.get_data(&mut ret as *mut T as *mut c_void, (index * size_of::<T>()) as VkDeviceSize, size_of::<T>()).ok()?;
				Some(ret)
			}
		} else {
			None
		}
	}

	/// Set data
	pub fn set_item(&mut self, index: usize, data: T) -> Result<(), VulkanError> {
		if index >= self.len() {
			panic!("The index is {index}, and the size of the buffer is {}", self.len());
		}
		unsafe {self.buffer.set_staging_data(&data as *const T as *const c_void, (index * size_of::<T>()) as VkDeviceSize, size_of::<T>())}
	}

	/// Get all data
	pub fn get_data(&self) -> Option<&[T]> {
		if let Some(ref staging_buffer) = self.buffer.staging_buffer {
			Some(unsafe {slice::from_raw_parts(staging_buffer.get_address() as *const T, self.len())})
		} else {
			None
		}
	}

	/// Set all data
	/// **NOTE** The buffer will be resized to the exact size of the data, causes the return value of `get_vk_buffer` to be changed to a new buffer.
	pub fn set_data(&mut self, data: &[T]) -> Result<(), VulkanError> {
		if self.len() != data.len() {
			self.buffer = Buffer::new(self.buffer.device.clone(), size_of_val(data) as VkDeviceSize, Some(data.as_ptr() as *const c_void), self.buffer.usage)?;
		}
		unsafe {self.buffer.set_staging_data(data.as_ptr() as *const c_void, 0, size_of_val(data))}
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
pub trait BufferForDraw<T>: Debug + Clone + Any
where
	T: BufferVecItem {
	/// Used to create the buffer
	fn create(device: Arc<VulkanDevice>, data: &[T], cmdbuf: VkCommandBuffer, usage: VkBufferUsageFlags) -> Result<Self, VulkanError>;

	/// Must be able to get the `VkBuffer` handle
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Get the device
	fn get_device(&self) -> Arc<VulkanDevice>;

	/// Set data to be flushed
	fn set_data(&mut self, data: &[T]) -> Result<(), VulkanError>;

	/// Flush staging buffer data to GPU
	fn flush(&mut self, _cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		Ok(())
	}

	/// Discard staging buffer if the buffer's staging buffer is discardable
	fn discard_staging_buffer(&mut self) {}

	/// Get the number of the items in the buffer
	fn len(&self) -> usize;

	/// Check if the buffer is empty
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Convert to `BufferVec<T>`
	fn convert_to_buffer_vec(self) -> BufferVec<T>;

	/// Convert to `BufferWithType<T>`
	fn convert_to_buffer_with_type(self) -> BufferWithType<T>;
}

impl<T> BufferForDraw<T> for BufferVec<T>
where
	T: BufferVecItem {
	fn create(device: Arc<VulkanDevice>, data: &[T], cmdbuf: VkCommandBuffer, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		BufferVec::from(device, data, cmdbuf, usage)
	}

	fn get_vk_buffer(&self) -> VkBuffer {
		self.get_vk_buffer()
	}

	fn get_device(&self) -> Arc<VulkanDevice> {
		self.get_device()
	}

	fn set_data(&mut self, data: &[T]) -> Result<(), VulkanError> {
		self.resize(data.len(), T::default())?;
		let s: &mut [T] = &mut self[..];
		unsafe {copy(data.as_ptr(), s.as_mut_ptr(), s.len())};
		Ok(())
	}

	fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.flush(cmdbuf)
	}

	fn len(&self) -> usize {
		self.len()
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
	fn create(device: Arc<VulkanDevice>, data: &[T], cmdbuf: VkCommandBuffer, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		BufferWithType::new(device, data, cmdbuf, usage)
	}

	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	fn get_device(&self) -> Arc<VulkanDevice> {
		self.get_device()
	}

	fn set_data(&mut self, data: &[T]) -> Result<(), VulkanError> {
		self.set_data(data)
	}

	fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.upload_staging_buffer(cmdbuf)?;
		Ok(())
	}

	fn discard_staging_buffer(&mut self) {
		self.discard_staging_buffer()
	}

	fn len(&self) -> usize {
		self.len()
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
	pub vertices: Arc<RwLock<BV>>,
	pub indices: Option<Arc<RwLock<BE>>>,
	pub instances: Option<Arc<RwLock<BI>>>,
	pub commands: Option<Arc<RwLock<BC>>>,
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
pub fn buffer_unused() -> Option<Arc<RwLock<UnusedBufferType>>> {
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
	/// Create the mesh from the buffers
	pub fn new(primitive_type: VkPrimitiveTopology, vertices: Arc<RwLock<BV>>, indices: Option<Arc<RwLock<BE>>>, instances: Option<Arc<RwLock<BI>>>, commands: Option<Arc<RwLock<BC>>>) -> Self {
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

	/// Create the index buffer
	pub fn create_index_buffer(&mut self, cmdbuf: VkCommandBuffer, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		let device = self.vertices.read().unwrap().get_device();
		let data_slice = unsafe {slice::from_raw_parts(data as *const E, size / size_of::<E>())};
		self.indices = Some(Arc::new(RwLock::new(BE::create(device, data_slice, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_INDEX_BUFFER_BIT as VkBufferUsageFlags)?)));
		Ok(())
	}

	/// Create the instance buffer
	pub fn create_instance_buffer(&mut self, cmdbuf: VkCommandBuffer, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		let device = self.vertices.read().unwrap().get_device();
		let data_slice = unsafe {slice::from_raw_parts(data as *const I, size / size_of::<I>())};
		self.instances = Some(Arc::new(RwLock::new(BI::create(device, data_slice, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?)));
		Ok(())
	}

	/// Create the command buffer
	pub fn create_command_buffer(&mut self, cmdbuf: VkCommandBuffer, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		let device = self.vertices.read().unwrap().get_device();
		let data_slice = unsafe {slice::from_raw_parts(data as *const C, size / size_of::<C>())};
		self.commands = Some(Arc::new(RwLock::new(BC::create(device, data_slice, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT as VkBufferUsageFlags)?)));
		Ok(())
	}

	/// Upload staging buffers to GPU
	pub fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		filter_no_staging_buffer(self.vertices.write().unwrap().flush(cmdbuf))?;
		if let Some(ref indices) = self.indices {filter_no_staging_buffer(indices.write().unwrap().flush(cmdbuf))?;}
		if let Some(ref instances) = self.instances {filter_no_staging_buffer(instances.write().unwrap().flush(cmdbuf))?;}
		if let Some(ref commands) = self.commands {filter_no_staging_buffer(commands.write().unwrap().flush(cmdbuf))?;}
		Ok(())
	}

	/// Discard staging buffers if the data will never be modified.
	pub fn discard_staging_buffers(&self) {
		self.vertices.write().unwrap().discard_staging_buffer();
		if let Some(ref indices) = self.indices {indices.write().unwrap().discard_staging_buffer();}
		if let Some(ref instances) = self.instances {instances.write().unwrap().discard_staging_buffer();}
		if let Some(ref commands) = self.commands {commands.write().unwrap().discard_staging_buffer();}
	}
}

/// The most typical static mesh type: use `BufferWithType` for vertices and elements(indices), use `BufferVec` for instances and draw commands
pub type StaticMesh<V, E, I, C> = Mesh<BufferWithType<V>, V, BufferWithType<E>, E, BufferVec<I>, I, BufferVec<C>, C>;

/// The dynamic mesh type: use `BufferVec` for all buffers
pub type DynamicMesh<V, E, I, C> = Mesh<BufferVec<V>, V, BufferVec<E>, E, BufferVec<I>, I, BufferVec<C>, C>;

/// The trait for a mesh
pub trait GenericMesh: Debug + Any {
	/// Clone the mesh
	fn clone(&self) -> Box<dyn GenericMesh>;

	/// Get the vertex buffer
	fn get_vk_vertex_buffer(&self) -> VkBuffer;

	/// Get the index buffer
	fn get_vk_index_buffer(&self) -> Option<VkBuffer>;

	/// Get the instance buffer
	fn get_vk_instance_buffer(&self) -> Option<VkBuffer>;

	/// Get the command buffer
	fn get_vk_command_buffer(&self) -> Option<VkBuffer>;

	/// Create the index buffer
	fn create_index_buffer(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Create the instance buffer
	fn create_instance_buffer(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Create the command buffer
	fn create_command_buffer(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Get the vertex buffer
	fn set_vertex_buffer_data(&self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Get the index buffer
	fn set_index_buffer_data(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Get the instance buffer
	fn set_instance_buffer_data(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Get the command buffer
	fn set_command_buffer_data(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError>;

	/// Get the primitive type
	fn get_primitive_type(&self) -> VkPrimitiveTopology;

	/// Get vertex count
	fn get_vertex_count(&self) -> usize;

	/// Get index count
	fn get_index_count(&self) -> usize;

	/// Get instance count
	fn get_instance_count(&self) -> usize;

	/// Get command count
	fn get_command_count(&self) -> usize;

	/// Get the iterator for the vertex buffer item structure
	fn iter_vertex_buffer_struct_members(&self) -> IntoIter<(&'static str, &(dyn Any + 'static))>;

	/// Get the TypeId of the index buffer item
	fn get_index_type_id(&self) -> Option<TypeId>;

	/// Get the iterator for the vertex buffer item structure
	fn iter_instance_buffer_struct_members(&self) -> Option<IntoIter<(&'static str, &(dyn Any + 'static))>>;

	/// Get the iterator for the vertex buffer item structure
	fn iter_command_buffer_struct_members(&self) -> Option<IntoIter<(&'static str, &(dyn Any + 'static))>>;

	/// Get the stride of the vertex buffer
	fn get_vertex_type_name(&self) -> &'static str;

	/// Get the stride of the index buffer
	fn get_index_type_name(&self) -> &'static str;

	/// Get the stride of the instance buffer
	fn get_instance_type_name(&self) -> &'static str;

	/// Get the stride of the command buffer
	fn get_command_type_name(&self) -> &'static str;

	/// Get the stride of the vertex buffer
	fn get_vertex_stride(&self) -> usize;

	/// Get the stride of the index buffer
	fn get_index_stride(&self) -> usize;

	/// Get the stride of the instance buffer
	fn get_instance_stride(&self) -> usize;

	/// Get the stride of the command buffer
	fn get_command_stride(&self) -> usize;

	/// Flush all buffers that needs to be flushed to use
	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError>;

	/// Discard staging buffers if the data will never be modified.
	fn discard_staging_buffers(&self);

	/// Get the index type
	fn get_index_type(&self) -> Option<VkIndexType> {
		match self.get_index_stride() {
			0 => None,
			1 => Some(VkIndexType::VK_INDEX_TYPE_UINT8),
			2 => Some(VkIndexType::VK_INDEX_TYPE_UINT16),
			4 => Some(VkIndexType::VK_INDEX_TYPE_UINT32),
			_ => panic!("Unsupported index type: {}", self.get_index_type_name()),
		}
	}
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
	fn clone(&self) -> Box<dyn GenericMesh> {
		Box::new(Clone::clone(self))
	}

	fn get_vk_vertex_buffer(&self) -> VkBuffer {
		self.vertices.read().unwrap().get_vk_buffer()
	}

	fn get_vk_index_buffer(&self) -> Option<VkBuffer> {
		self.indices.as_ref().map(|b|b.read().unwrap().get_vk_buffer())
	}

	fn get_vk_instance_buffer(&self) -> Option<VkBuffer> {
		self.instances.as_ref().map(|b|b.read().unwrap().get_vk_buffer())
	}

	fn get_vk_command_buffer(&self) -> Option<VkBuffer> {
		self.commands.as_ref().map(|b|b.read().unwrap().get_vk_buffer())
	}

	fn create_index_buffer(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		self.create_index_buffer(null(), data, size)
	}

	fn create_instance_buffer(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		self.create_instance_buffer(null(), data, size)
	}

	fn create_command_buffer(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		self.create_command_buffer(null(), data, size)
	}

	fn set_vertex_buffer_data(&self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		self.vertices.write().unwrap().set_data(unsafe {slice::from_raw_parts(data as *const V, size / size_of::<V>())})
	}

	fn set_index_buffer_data(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		if let Some(ref mut eb) = self.indices {
			eb.write().unwrap().set_data(unsafe {slice::from_raw_parts(data as *const E, size / size_of::<E>())})?;
		} else {
			self.create_index_buffer(null(), data, size)?;
		}
		Ok(())
	}

	fn set_instance_buffer_data(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		if let Some(ref mut ib) = self.instances {
			ib.write().unwrap().set_data(unsafe {slice::from_raw_parts(data as *const I, size / size_of::<I>())})?;
		} else {
			self.create_instance_buffer(null(), data, size)?;
		}
		Ok(())
	}

	fn set_command_buffer_data(&mut self, data: *const c_void, size: usize) -> Result<(), VulkanError> {
		if let Some(ref mut cb) = self.commands {
			cb.write().unwrap().set_data(unsafe {slice::from_raw_parts(data as *const C, size / size_of::<C>())})?;
		} else {
			self.create_command_buffer(null(), data, size)?;
		}
		Ok(())
	}

	fn get_primitive_type(&self) -> VkPrimitiveTopology {
		self.primitive_type
	}

	fn get_vertex_count(&self) -> usize {
		self.vertices.read().unwrap().len()
	}

	fn get_index_count(&self) -> usize {
		if let Some(indices) = &self.indices {
			indices.read().unwrap().len()
		} else {
			0
		}
	}

	fn get_instance_count(&self) -> usize {
		if let Some(instances) = &self.instances {
			instances.read().unwrap().len()
		} else {
			1
		}
	}

	fn get_command_count(&self) -> usize {
		if let Some(commands) = &self.commands {
			commands.read().unwrap().len()
		} else {
			0
		}
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

	fn get_vertex_type_name(&self) -> &'static str {
		type_name::<V>()
	}

	fn get_index_type_name(&self) -> &'static str {
		type_name::<E>()
	}

	fn get_instance_type_name(&self) -> &'static str {
		type_name::<I>()
	}

	fn get_command_type_name(&self) -> &'static str {
		type_name::<C>()
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

	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.flush(cmdbuf)
	}

	fn discard_staging_buffers(&self) {
		self.discard_staging_buffers()
	}
}

/// Obj file parse error
#[derive(Debug, Clone)]
pub struct ObjParseError {
	pub line_number: usize,
	pub message: String,
}

/// The struct for OBJ vertices with position data only
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionOnly {
	pub position: Vec3,
}

/// The struct for OBJ vertices with position data and normal data
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionNormal {
	pub position: Vec3,
	pub normal: Vec3,
}

/// The struct for OBJ vertices with position data and and texcoord data (2D)
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord2D {
	pub position: Vec3,
	pub texcoord: Vec2,
}

/// The struct for OBJ vertices with position data and texcoord data (3D)
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord3D {
	pub position: Vec3,
	pub texcoord: Vec3,
}

/// The struct for OBJ vertices with position data, texcoord data (2D) and normal data
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord2DNormal {
	pub position: Vec3,
	pub texcoord: Vec2,
	pub normal: Vec3,
}

/// The struct for OBJ vertices with position data, texcoord data (3D) and normal data
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord3DNormal {
	pub position: Vec3,
	pub texcoord: Vec3,
	pub normal: Vec3,
}

/// The struct for OBJ vertices with position data only
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionOnlyDouble {
	pub position: DVec3,
}

/// The struct for OBJ vertices with position data and normal data
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionNormalDouble {
	pub position: DVec3,
	pub normal: DVec3,
}

/// The struct for OBJ vertices with position data and and texcoord data (2D)
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord2DDouble {
	pub position: DVec3,
	pub texcoord: DVec2,
}

/// The struct for OBJ vertices with position data and texcoord data (3D)
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord3DDouble {
	pub position: DVec3,
	pub texcoord: DVec3,
}

/// The struct for OBJ vertices with position data, texcoord data (2D) and normal data
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord2DNormalDouble {
	pub position: DVec3,
	pub texcoord: DVec2,
	pub normal: DVec3,
}

/// The struct for OBJ vertices with position data, texcoord data (3D) and normal data
#[derive(Iterable, Default, Debug, Clone, Copy)]
pub struct ObjVertPositionTexcoord3DNormalDouble {
	pub position: DVec3,
	pub texcoord: DVec3,
	pub normal: DVec3,
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionOnly
where
	F: ObjMeshVecCompType, f32: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: Vec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionNormal
where
	F: ObjMeshVecCompType, f32: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: Vec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			normal: if let Some(normal) = f.normal {
				Vec3::new(normal.x.into(), normal.y.into(), normal.z.into())
			} else {
				Vec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord2D
where
	F: ObjMeshVecCompType, f32: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: Vec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				Vec2::new(texcoord.x.into(), texcoord.y.into())
			} else {
				Vec2::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord3D
where
	F: ObjMeshVecCompType, f32: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: Vec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				Vec3::new(texcoord.x.into(), texcoord.y.into(), texcoord.z.into())
			} else {
				Vec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord2DNormal
where
	F: ObjMeshVecCompType, f32: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: Vec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				Vec2::new(texcoord.x.into(), texcoord.y.into())
			} else {
				Vec2::default()
			},
			normal: if let Some(normal) = f.normal {
				Vec3::new(normal.x.into(), normal.y.into(), normal.z.into())
			} else {
				Vec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord3DNormal
where
	F: ObjMeshVecCompType, f32: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: Vec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				Vec3::new(texcoord.x.into(), texcoord.y.into(), texcoord.z.into())
			} else {
				Vec3::default()
			},
			normal: if let Some(normal) = f.normal {
				Vec3::new(normal.x.into(), normal.y.into(), normal.z.into())
			} else {
				Vec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionOnlyDouble
where
	F: ObjMeshVecCompType, f64: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: DVec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionNormalDouble
where
	F: ObjMeshVecCompType, f64: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: DVec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			normal: if let Some(normal) = f.normal {
				DVec3::new(normal.x.into(), normal.y.into(), normal.z.into())
			} else {
				DVec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord2DDouble
where
	F: ObjMeshVecCompType, f64: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: DVec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				DVec2::new(texcoord.x.into(), texcoord.y.into())
			} else {
				DVec2::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord3DDouble
where
	F: ObjMeshVecCompType, f64: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: DVec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				DVec3::new(texcoord.x.into(), texcoord.y.into(), texcoord.z.into())
			} else {
				DVec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord2DNormalDouble
where
	F: ObjMeshVecCompType, f64: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: DVec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				DVec2::new(texcoord.x.into(), texcoord.y.into())
			} else {
				DVec2::default()
			},
			normal: if let Some(normal) = f.normal {
				DVec3::new(normal.x.into(), normal.y.into(), normal.z.into())
			} else {
				DVec3::default()
			},
		}
	}
}

impl<F> From<ObjIndexedVertices<F>> for ObjVertPositionTexcoord3DNormalDouble
where
	F: ObjMeshVecCompType, f64: From<F> {
	fn from(f: ObjIndexedVertices<F>) -> Self {
		Self {
			position: DVec3::new(f.position.x.into(), f.position.y.into(), f.position.z.into()),
			texcoord: if let Some(texcoord) = f.texcoord {
				DVec3::new(texcoord.x.into(), texcoord.y.into(), texcoord.z.into())
			} else {
				DVec3::default()
			},
			normal: if let Some(normal) = f.normal {
				DVec3::new(normal.x.into(), normal.y.into(), normal.z.into())
			} else {
				DVec3::default()
			},
		}
	}
}

/// A Mesh with a material
#[derive(Debug, Clone)]
pub struct GenericMeshWithMaterial {
	/// The mesh
	pub geometry: Arc<dyn GenericMesh>,

	/// The material
	pub material: Option<Arc<dyn Material>>,
}

impl GenericMeshWithMaterial {
	/// Create an instance for the `GenericMeshWithMaterial`
	pub fn new(geometry: Arc<dyn GenericMesh>, material: Option<Arc<dyn Material>>) -> Self {
		Self {
			geometry,
			material,
		}
	}

	/// Discard staging buffers for the mesh
	pub fn discard_staging_buffers(&self) {
		self.geometry.discard_staging_buffers();
	}
}

/// The mesh set
#[derive(Debug, Clone)]
pub struct GenericMeshSet<I>
where
	I: BufferVecStructItem {
	/// The meshset
	pub meshset: BTreeMap<String, Arc<GenericMeshWithMaterial>>,

	/// The instance buffer of the meshset, modify this instance buffer equals to modify each meshes' instance buffer
	instances: Option<Arc<RwLock<BufferVec<I>>>>,
}

impl<I> GenericMeshSet<I>
where
	I: BufferVecStructItem {
	/// Load the `obj` file and create the meshset, all the materials were also loaded.
	pub fn create_meshset_from_obj_file<F, P>(device: Arc<VulkanDevice>, path: P, cmdbuf: VkCommandBuffer, instances_data: Option<&[I]>) -> Result<Self, VulkanError>
	where
		P: AsRef<Path>,
		F: ObjMeshVecCompType,
		f32: From<F>,
		f64: From<F> {
		let obj = ObjMesh::<F>::from_file(path)?;
		Self::create_meshset_from_obj(device, &obj, cmdbuf, instances_data)
	}
	/// Load the `obj` file and create the meshset, all the materials were also loaded.
	pub fn create_meshset_from_obj<F>(device: Arc<VulkanDevice>, obj: &ObjMesh::<F>, cmdbuf: VkCommandBuffer, instances_data: Option<&[I]>) -> Result<Self, VulkanError>
	where
		F: ObjMeshVecCompType,
		f32: From<F>,
		f64: From<F> {
		let obj_mesh_set: ObjIndexedMeshSet<F, u32> = obj.convert_to_indexed_meshes()?;
		let (pdim, tdim, ndim) = obj_mesh_set.get_vert_dims();
		let template_mesh;
		let instances;
		macro_rules! vert_conv {
			($type:ty, $src:ident) => {
				{let vertices: Vec<$type> = $src.iter().map(|v|<$type>::from(*v)).collect(); vertices}
			}
		}
		macro_rules! mesh_create {
			($vb:ident) => {
				{
					instances = if let Some(id) = instances_data {
						Some(Arc::new(RwLock::new(BufferVec::from(device.clone(), id, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?)))
					} else {
						None
					};
					let mesh: Box<dyn GenericMesh> = Box::new(Mesh::new(VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, Arc::new(RwLock::new($vb)), Option::<Arc::<RwLock::<BufferWithType::<u32>>>>::None, instances.clone(), buffer_unused()));
					mesh
				}
			}
		}
		#[allow(non_camel_case_types)] type ObjV___F = ObjVertPositionOnly;
		#[allow(non_camel_case_types)] type ObjV__NF = ObjVertPositionNormal;
		#[allow(non_camel_case_types)] type ObjV2D_F = ObjVertPositionTexcoord2D;
		#[allow(non_camel_case_types)] type ObjV3D_F = ObjVertPositionTexcoord3D;
		#[allow(non_camel_case_types)] type ObjV2DNF = ObjVertPositionTexcoord2DNormal;
		#[allow(non_camel_case_types)] type ObjV3DNF = ObjVertPositionTexcoord3DNormal;
		#[allow(non_camel_case_types)] type ObjV___D = ObjVertPositionOnlyDouble;
		#[allow(non_camel_case_types)] type ObjV__ND = ObjVertPositionNormalDouble;
		#[allow(non_camel_case_types)] type ObjV2D_D = ObjVertPositionTexcoord2DDouble;
		#[allow(non_camel_case_types)] type ObjV3D_D = ObjVertPositionTexcoord3DDouble;
		#[allow(non_camel_case_types)] type ObjV2DND = ObjVertPositionTexcoord2DNormalDouble;
		#[allow(non_camel_case_types)] type ObjV3DND = ObjVertPositionTexcoord3DNormalDouble;
		if TypeId::of::<F>() == TypeId::of::<f32>() {
			match (pdim, tdim, ndim) {
				(_, 0, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV___F, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 1, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2D_F, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 2, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2D_F, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 3, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV3D_F, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 0, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV__NF, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 1, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2DNF, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 2, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2DNF, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 3, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV3DNF, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, _, _) => panic!("Unknown VTN dimensions: V({pdim}), T({tdim}), N({ndim})"),
			}
		} else if TypeId::of::<F>() == TypeId::of::<f64>() {
			match (pdim, tdim, ndim) {
				(_, 0, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV___D, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 1, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2D_D, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 2, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2D_D, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 3, 0) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV3D_D, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 0, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV__ND, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 1, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2DND, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 2, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV2DND, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, 3, _) => {let fv = obj_mesh_set.face_vertices; let vertices = vert_conv!(ObjV3DND, fv); let vb = BufferWithType::new(device.clone(), &vertices, cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?; template_mesh = mesh_create!(vb)}
				(_, _, _) => panic!("Unknown VTN dimensions: V({pdim}), T({tdim}), N({ndim})"),
			}
		} else {
			let name_of_f = type_name::<F>();
			panic!("Unsupported generic type `{name_of_f}`, couldn't match its type id to neither `f32` nor `f64`");
		}
		let mut materials: BTreeMap<String, Arc<dyn Material>> = BTreeMap::new();
		for (matname, matdata) in obj.materials.iter() {
			materials.insert(matname.clone(), create_material_from_obj_material(device.clone(), cmdbuf, &*matdata.read().unwrap())?);
		}
		let mut meshset = BTreeMap::new();
		for objmesh in obj_mesh_set.meshes.iter() {
			let object_name = &objmesh.object_name;
			let group_name = &objmesh.group_name;
			let material_name = &objmesh.material_name;
			let smooth_group = objmesh.smooth_group;
			let mut indices: Vec<u32> = Vec::with_capacity(objmesh.face_indices.len() * 3);
			let mut mesh = template_mesh.clone();
			for (v1, v2, v3) in objmesh.face_indices.iter() {
				indices.extend([v1, v2, v3]);
			}
			mesh.create_index_buffer(indices.as_ptr() as *const c_void, size_of_val(&indices))?;
			mesh.flush(cmdbuf)?;
			meshset.insert(format!("{object_name}_{group_name}_{material_name}_{smooth_group}"), Arc::new(GenericMeshWithMaterial{
				geometry: Arc::from(mesh),
				material: materials.get(material_name).cloned(),
			}));
		}
		Ok(Self {
			meshset,
			instances,
		})
	}

	/// Edit the instance buffer
	pub fn edit_instances<'a>(&'a self) -> Option<RwLockWriteGuard<'a, BufferVec<I>>> {
		self.instances.as_ref().map(|ib|ib.write().unwrap())
	}

	/// Discard staging buffers for all meshes
	pub fn discard_staging_buffers(&self) {
		for mesh in self.meshset.values() {
			mesh.discard_staging_buffers();
		}
	}
}
