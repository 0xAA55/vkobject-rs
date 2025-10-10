
use crate::prelude::*;
use std::{
	collections::HashMap,
	sync::{Arc, RwLock},
};

/// The properties for the descriptor set
#[derive(Debug)]
pub enum DescriptorProp {
	/// The props for the samplers
	Samplers(Vec<Arc<VulkanSampler>>),

	/// The props for the image
	Images(Vec<TextureForSample>),

	/// The props for the storage buffer
	StorageBuffers(Vec<Arc<dyn GenericStorageBuffer>>),

	/// The props for the uniform buffers
	UniformBuffers(Vec<Arc<dyn GenericUniformBuffer>>),

	/// The props for the storage texel buffer
	StorageTexelBuffers(Vec<Arc<VulkanBufferView>>),

	/// The props for the uniform texel buffers
	UniformTexelBuffers(Vec<Arc<VulkanBufferView>>),
}

impl DescriptorProp {
	/// Create as a sampler
	pub fn new_sampler(sampler: Arc<VulkanSampler>) -> Self {
		Self::Samplers(vec![sampler])
	}

	/// Create as samplers
	pub fn new_samplers(samplers: Vec<Arc<VulkanSampler>>) -> Self {
		Self::Samplers(samplers)
	}

	/// Create as a texture
	pub fn new_texture(texture: TextureForSample) -> Self {
		Self::Images(vec![texture])
	}

	/// Create as textures
	pub fn new_textures(textures: Vec<TextureForSample>) -> Self {
		Self::Images(textures)
	}

	/// Create as a storage buffer
	pub fn new_storage_buffer(storage_buffer: Arc<dyn GenericStorageBuffer>) -> Self {
		Self::StorageBuffers(vec![storage_buffer])
	}

	/// Create as storage buffers
	pub fn new_storage_buffers(storage_buffers: Vec<Arc<dyn GenericStorageBuffer>>) -> Self {
		Self::StorageBuffers(storage_buffers)
	}

	/// Create as a uniform buffer
	pub fn new_uniform_buffer(uniform_buffer: Arc<dyn GenericUniformBuffer>) -> Self {
		Self::UniformBuffers(vec![uniform_buffer])
	}

	/// Create as uniform buffers
	pub fn new_uniform_buffers(uniform_buffers: Vec<Arc<dyn GenericUniformBuffer>>) -> Self {
		Self::UniformBuffers(uniform_buffers)
	}

	/// Create as a storage texel buffer
	pub fn new_storage_texel_buffer(storage_texel_buffer: Arc<VulkanBufferView>) -> Self {
		Self::StorageTexelBuffers(vec![storage_texel_buffer])
	}

	/// Create as storage texel buffers
	pub fn new_storage_texel_buffers(storage_texel_buffers: Vec<Arc<VulkanBufferView>>) -> Self {
		Self::StorageTexelBuffers(storage_texel_buffers)
	}

	/// Create as a uniform texel buffer
	pub fn new_uniform_texel_buffer(uniform_texel_buffer: Arc<VulkanBufferView>) -> Self {
		Self::UniformTexelBuffers(vec![uniform_texel_buffer])
	}

	/// Create as uniform texel buffers
	pub fn new_uniform_texel_buffers(uniform_texel_buffers: Vec<Arc<VulkanBufferView>>) -> Self {
		Self::UniformTexelBuffers(uniform_texel_buffers)
	}

	/// Get samplers
	pub fn get_samplers(&self) -> Result<&[Arc<VulkanSampler>], VulkanError> {
		if let Self::Samplers(samplers) = self {
			Ok(samplers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::Samplers`, got {self:?}")))
		}
	}

	/// Get images
	pub fn get_images(&self) -> Result<&[TextureForSample], VulkanError> {
		if let Self::Images(images) = self {
			Ok(images)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::Images`, got {self:?}")))
		}
	}

	/// Get uniform buffers
	pub fn get_uniform_buffers(&self) -> Result<&[Arc<dyn GenericUniformBuffer>], VulkanError> {
		if let Self::UniformBuffers(uniform_buffers) = self {
			Ok(uniform_buffers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::UniformBuffers`, got {self:?}")))
		}
	}

	/// Get storage buffers
	pub fn get_storage_buffers(&self) -> Result<&[Arc<dyn GenericStorageBuffer>], VulkanError> {
		if let Self::StorageBuffers(uniform_buffers) = self {
			Ok(uniform_buffers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::StorageBuffers`, got {self:?}")))
		}
	}

	/// Get uniform texel buffers
	pub fn get_uniform_texel_buffers(&self) -> Result<&[Arc<VulkanBufferView>], VulkanError> {
		if let Self::UniformTexelBuffers(uniform_buffers) = self {
			Ok(uniform_buffers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::UniformTexelBuffers`, got {self:?}")))
		}
	}

	/// Get storage texel buffers
	pub fn get_storage_texel_buffers(&self) -> Result<&[Arc<VulkanBufferView>], VulkanError> {
		if let Self::StorageTexelBuffers(uniform_buffers) = self {
			Ok(uniform_buffers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::StorageTexelBuffers`, got {self:?}")))
		}
	}

	/// Unwrap for samplers
	pub fn unwrap_samplers(&self) -> &[Arc<VulkanSampler>] {
		if let Self::Samplers(samplers) = self {
			samplers
		} else {
			panic!("Expected `DescriptorProp::Samplers`, got {self:?}")
		}
	}

	/// Unwrap for images
	pub fn unwrap_images(&self) -> &[TextureForSample] {
		if let Self::Images(images) = self {
			images
		} else {
			panic!("Expected `DescriptorProp::Images`, got {self:?}")
		}
	}

	/// Unwrap for uniform buffers
	pub fn unwrap_uniform_buffers(&self) -> &[Arc<dyn GenericUniformBuffer>] {
		if let Self::UniformBuffers(uniform_buffers) = self {
			uniform_buffers
		} else {
			panic!("Expected `DescriptorProp::UniformBuffers`, got {self:?}")
		}
	}

	/// Unwrap for storage buffers
	pub fn unwrap_storage_buffers(&self) -> &[Arc<dyn GenericStorageBuffer>] {
		if let Self::StorageBuffers(storage_buffers) = self {
			storage_buffers
		} else {
			panic!("Expected `DescriptorProp::StorageBuffers`, got {self:?}")
		}
	}

	/// Unwrap for uniform texel buffers
	pub fn unwrap_uniform_texel_buffers(&self) -> &[Arc<VulkanBufferView>] {
		if let Self::UniformTexelBuffers(uniform_texel_buffers) = self {
			uniform_texel_buffers
		} else {
			panic!("Expected `DescriptorProp::UniformTexelBuffers`, got {self:?}")
		}
	}

	/// Unwrap for storage texel buffers
	pub fn unwrap_storage_texel_buffers(&self) -> &[Arc<VulkanBufferView>] {
		if let Self::StorageTexelBuffers(storage_texel_buffers) = self {
			storage_texel_buffers
		} else {
			panic!("Expected `DescriptorProp::StorageTexelBuffers`, got {self:?}")
		}
	}

	/// Check if it is samplers
	pub fn is_samplers(&self) -> bool {
		matches!(self, Self::Samplers(_))
	}

	/// Check if it is images
	pub fn is_images(&self) -> bool {
		matches!(self, Self::Images(_))
	}

	/// Check if it is uniform buffers
	pub fn is_uniform_buffers(&self) -> bool {
		matches!(self, Self::UniformBuffers(_))
	}

	/// Check if it is storage buffers
	pub fn is_storage_buffers(&self) -> bool {
		matches!(self, Self::StorageBuffers(_))
	}

	/// Check if it is uniform texel buffers
	pub fn is_uniform_texel_buffers(&self) -> bool {
		matches!(self, Self::UniformTexelBuffers(_))
	}

	/// Check if it is storage texel buffers
	pub fn is_storage_texel_buffers(&self) -> bool {
		matches!(self, Self::StorageTexelBuffers(_))
	}
}

/// The descriptor set properties
#[derive(Default, Debug)]
pub struct DescriptorProps {
	/// The descriptor sets
	pub sets: RwLock<HashMap<u32 /* set */, HashMap<u32 /* binding */, Arc<DescriptorProp>>>>,
}

impl DescriptorProps {
	/// Create a new `DescriptorProps`
	pub fn new() -> Self {
		Self::default()
	}

	/// Insert a prop
	pub fn insert(&self, set: u32, binding: u32, prop: Arc<DescriptorProp>) -> Option<Arc<DescriptorProp>> {
		let mut w_lock = self.sets.write().unwrap();
		if let Some(bindings) = w_lock.get_mut(&set) {
			bindings.insert(binding, prop)
		} else {
			w_lock.insert(set, HashMap::new());
			w_lock.get_mut(&set).unwrap().insert(binding, prop)
		}
	}

	/// Get from the set
	pub fn get(&self, set: u32, binding: u32) -> Option<Arc<DescriptorProp>> {
		let r_lock = self.sets.read().unwrap();
		if let Some(bindings) = r_lock.get(&set) {
			bindings.get(&binding).cloned()
		} else {
			None
		}
	}

	/// Create as a sampler
	pub fn new_sampler(&self, set: u32, binding: u32, sampler: Arc<VulkanSampler>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_sampler(sampler)))
	}

	/// Create as samplers
	pub fn new_samplers(&self, set: u32, binding: u32, samplers: Vec<Arc<VulkanSampler>>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_samplers(samplers)))
	}

	/// Create as a texture
	pub fn new_texture(&self, set: u32, binding: u32, texture: TextureForSample) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_texture(texture)))
	}

	/// Create as textures
	pub fn new_textures(&self, set: u32, binding: u32, textures: Vec<TextureForSample>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_textures(textures)))
	}

	/// Create as a storage buffer
	pub fn new_storage_buffer(&self, set: u32, binding: u32, storage_buffer: Arc<dyn GenericStorageBuffer>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_storage_buffer(storage_buffer)))
	}

	/// Create as storage buffers
	pub fn new_storage_buffers(&self, set: u32, binding: u32, storage_buffers: Vec<Arc<dyn GenericStorageBuffer>>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_storage_buffers(storage_buffers)))
	}

	/// Create as a uniform buffer
	pub fn new_uniform_buffer(&self, set: u32, binding: u32, uniform_buffer: Arc<dyn GenericUniformBuffer>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_uniform_buffer(uniform_buffer)))
	}

	/// Create as uniform buffers
	pub fn new_uniform_buffers(&self, set: u32, binding: u32, uniform_buffers: Vec<Arc<dyn GenericUniformBuffer>>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_uniform_buffers(uniform_buffers)))
	}

	/// Create as a storage texel buffer
	pub fn new_storage_texel_buffer(&self, set: u32, binding: u32, storage_texel_buffer: Arc<VulkanBufferView>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_storage_texel_buffer(storage_texel_buffer)))
	}

	/// Create as storage texel buffers
	pub fn new_storage_texel_buffers(&self, set: u32, binding: u32, storage_texel_buffers: Vec<Arc<VulkanBufferView>>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_storage_texel_buffers(storage_texel_buffers)))
	}

	/// Create as a uniform texel buffer
	pub fn new_uniform_texel_buffer(&self, set: u32, binding: u32, uniform_texel_buffer: Arc<VulkanBufferView>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_uniform_texel_buffer(uniform_texel_buffer)))
	}

	/// Create as uniform texel buffers
	pub fn new_uniform_texel_buffers(&self, set: u32, binding: u32, uniform_texel_buffers: Vec<Arc<VulkanBufferView>>) -> Option<Arc<DescriptorProp>> {
		self.insert(set, binding, Arc::new(DescriptorProp::new_uniform_texel_buffers(uniform_texel_buffers)))
	}

	/// Get specific number of samplers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_samplers(&self, set: u32, binding: u32, desired_count: usize) -> Result<Vec<Arc<VulkanSampler>>, VulkanError> {
		let prop = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?;
		let samplers = prop.get_samplers()?;
		if samplers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} sampler(s) is needed for `layout(set = {set}, binding = {binding})`, but {} sampler(s) were provided.", samplers.len())));
		}
		Ok(samplers.to_vec())
	}

	/// Get specific number of textures from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_textures(&self, set: u32, binding: u32, desired_count: usize) -> Result<Vec<TextureForSample>, VulkanError> {
		let prop = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?;
		let textures = prop.get_images()?;
		if textures.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} texture(s) is needed for `layout(set = {set}, binding = {binding})`, but {} texture(s) were provided.", textures.len())));
		}
		Ok(textures.to_vec())
	}

	/// Get specific number of uniform buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_uniform_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<Vec<Arc<dyn GenericUniformBuffer>>, VulkanError> {
		let prop = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?;
		let uniform_buffers = prop.get_uniform_buffers()?;
		if uniform_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} uniform buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} uniform buffer(s) were provided.", uniform_buffers.len())));
		}
		Ok(uniform_buffers.to_vec())
	}

	/// Get specific number of uniform buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_storage_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<Vec<Arc<dyn GenericStorageBuffer>>, VulkanError> {
		let prop = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?;
		let storage_buffers = prop.get_storage_buffers()?;
		if storage_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} storage buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} storage buffer(s) were provided.", storage_buffers.len())));
		}
		Ok(storage_buffers.to_vec())
	}

	/// Get specific number of uniform texel buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_uniform_texel_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<Vec<Arc<VulkanBufferView>>, VulkanError> {
		let prop = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?;
		let uniform_texel_buffers = prop.get_uniform_texel_buffers()?;
		if uniform_texel_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} uniform buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} uniform buffer(s) were provided.", uniform_texel_buffers.len())));
		}
		Ok(uniform_texel_buffers.to_vec())
	}

	/// Get specific number of uniform texel buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_storage_texel_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<Vec<Arc<VulkanBufferView>>, VulkanError> {
		let prop = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?;
		let storage_texel_buffers = prop.get_storage_texel_buffers()?;
		if storage_texel_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} storage buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} storage buffer(s) were provided.", storage_texel_buffers.len())));
		}
		Ok(storage_texel_buffers.to_vec())
	}
}
