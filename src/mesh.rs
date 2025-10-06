
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
	sync::{Arc, Mutex},
	slice,
	vec::IntoIter,
};
use struct_iterable::Iterable;

/// The type that could be the item of the `BufferVec`
pub trait BufferVecStructItem: Copy + Clone + Sized + Default + Debug + Iterable + Any + 'static {}
impl<T> BufferVecStructItem for T where T: Copy + Clone + Sized + Default + Debug + Iterable + Any + 'static {}

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
	pub vertices: Arc<Mutex<BV>>,
	pub indices: Option<Arc<Mutex<BE>>>,
	pub instances: Option<Arc<Mutex<BI>>>,
	pub commands: Option<Arc<Mutex<BC>>>,
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
pub fn buffer_unused() -> Option<Arc<Mutex<UnusedBufferType>>> {
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
	pub fn new(primitive_type: VkPrimitiveTopology, vertices: Arc<Mutex<BV>>, indices: Option<Arc<Mutex<BE>>>, instances: Option<Arc<Mutex<BI>>>, commands: Option<Arc<Mutex<BC>>>) -> Self {
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

	/// Upload staging buffers to GPU
	pub fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		filter_no_staging_buffer(self.vertices.lock().unwrap().flush(cmdbuf))?;
		if let Some(ref indices) = self.indices {filter_no_staging_buffer(indices.lock().unwrap().flush(cmdbuf))?;}
		if let Some(ref instances) = self.instances {filter_no_staging_buffer(instances.lock().unwrap().flush(cmdbuf))?;}
		if let Some(ref commands) = self.commands {filter_no_staging_buffer(commands.lock().unwrap().flush(cmdbuf))?;}
		Ok(())
	}

	/// Discard staging buffers if the data will never be modified.
	pub fn discard_staging_buffers(&self) {
		self.vertices.lock().unwrap().discard_staging_buffer();
		if let Some(ref indices) = self.indices {indices.lock().unwrap().discard_staging_buffer();}
		if let Some(ref instances) = self.instances {instances.lock().unwrap().discard_staging_buffer();}
		if let Some(ref commands) = self.commands {commands.lock().unwrap().discard_staging_buffer();}
	}
}

/// The most typical static mesh type: use `BufferWithType` for vertices and elements(indices), use `BufferVec` for instances and draw commands
pub type StaticMesh<V, E, I, C> = Mesh<BufferWithType<V>, V, BufferWithType<E>, E, BufferVec<I>, I, BufferVec<C>, C>;

/// The dynamic mesh type: use `BufferVec` for all buffers
pub type DynamicMesh<V, E, I, C> = Mesh<BufferVec<V>, V, BufferVec<E>, E, BufferVec<I>, I, BufferVec<C>, C>;

/// The trait for a mesh
pub trait GenericMesh: Debug + Any {
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
	fn get_vk_vertex_buffer(&self) -> VkBuffer {
		self.vertices.lock().unwrap().get_vk_buffer()
	}

	fn get_vk_index_buffer(&self) -> Option<VkBuffer> {
		self.indices.as_ref().map(|b|b.lock().unwrap().get_vk_buffer())
	}

	fn get_vk_instance_buffer(&self) -> Option<VkBuffer> {
		self.instances.as_ref().map(|b|b.lock().unwrap().get_vk_buffer())
	}

	fn get_vk_command_buffer(&self) -> Option<VkBuffer> {
		self.commands.as_ref().map(|b|b.lock().unwrap().get_vk_buffer())
	}

	fn get_primitive_type(&self) -> VkPrimitiveTopology {
		self.primitive_type
	}

	fn get_vertex_count(&self) -> usize {
		self.vertices.lock().unwrap().len()
	}

	fn get_index_count(&self) -> usize {
		if let Some(indices) = &self.indices {
			indices.lock().unwrap().len()
		} else {
			0
		}
	}

	fn get_instance_count(&self) -> usize {
		if let Some(instances) = &self.instances {
			instances.lock().unwrap().len()
		} else {
			1
		}
	}

	fn get_command_count(&self) -> usize {
		if let Some(commands) = &self.commands {
			commands.lock().unwrap().len()
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

/// A Mesh with a material
#[derive(Debug)]
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

	/// Load the `obj` file and create the meshset, all the materials were also loaded.
	pub fn create_meshset_from_obj<P: AsRef<Path>>(device: Arc<VulkanDevice>, path: P, cmdbuf: VkCommandBuffer) -> Result<BTreeMap<String, Self>, VulkanError> {
		let obj = ObjMesh::<f32>::from_file(path);
		let ret = BTreeMap::new();
		Ok(ret)
	}
}

#[test]
fn test_obj() {
	let path = "assets/testobj/avocado.obj";
	let obj = ObjMesh::<f32>::from_file(path).unwrap();
	dbg!(&obj.materials);
}
