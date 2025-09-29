
use crate::prelude::*;
use std::{
	collections::HashMap,
	sync::Arc,
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
	StorageTexelBuffers(Vec<VulkanBufferView>),

	/// The props for the uniform texel buffers
	UniformTexelBuffers(Vec<VulkanBufferView>),
}

impl DescriptorProp {
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
	pub fn get_uniform_texel_buffers(&self) -> Result<&[VulkanBufferView], VulkanError> {
		if let Self::UniformTexelBuffers(uniform_buffers) = self {
			Ok(uniform_buffers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::UniformTexelBuffers`, got {self:?}")))
		}
	}

	/// Get storage texel buffers
	pub fn get_storage_texel_buffers(&self) -> Result<&[VulkanBufferView], VulkanError> {
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
	pub fn unwrap_uniform_texel_buffers(&self) -> &[VulkanBufferView] {
		if let Self::UniformTexelBuffers(uniform_texel_buffers) = self {
			uniform_texel_buffers
		} else {
			panic!("Expected `DescriptorProp::UniformTexelBuffers`, got {self:?}")
		}
	}

	/// Unwrap for storage texel buffers
	pub fn unwrap_storage_texel_buffers(&self) -> &[VulkanBufferView] {
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
#[derive(Default, Debug, Clone)]
pub struct DescriptorProps {
	/// The descriptor sets
	pub sets: HashMap<u32 /* set */, HashMap<u32 /* binding */, Arc<DescriptorProp>>>,
}

impl DescriptorProps {
	/// Create a new `DescriptorProps`
	pub fn new(sets: HashMap<u32, HashMap<u32, Arc<DescriptorProp>>>) -> Self {
		Self {
			sets,
		}
	}

	/// Insert a prop
	pub fn insert(&mut self, set: u32, binding: u32, prop: Arc<DescriptorProp>) -> Option<Arc<DescriptorProp>> {
		if let Some(bindings) = self.sets.get_mut(&set) {
			bindings.insert(binding, prop)
		} else {
			self.sets.insert(set, HashMap::new());
			self.sets.get_mut(&set).unwrap().insert(binding, prop)
		}
	}

	/// Get from the set
	pub fn get(&self, set: u32, binding: u32) -> Option<&Arc<DescriptorProp>> {
		if let Some(bindings) = self.sets.get(&set) {
			bindings.get(&binding)
		} else {
			None
		}
	}

	/// Get specific number of samplers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_samplers(&self, set: u32, binding: u32, desired_count: usize) -> Result<&[Arc<VulkanSampler>], VulkanError> {
		let samplers = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?.get_samplers()?;
		if samplers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} sampler(s) is needed for `layout(set = {set}, binding = {binding})`, but {} sampler(s) were provided.", samplers.len())));
		}
		Ok(samplers)
	}

	/// Get specific number of textures from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_textures(&self, set: u32, binding: u32, desired_count: usize) -> Result<&[TextureForSample], VulkanError> {
		let textures = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?.get_images()?;
		if textures.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} texture(s) is needed for `layout(set = {set}, binding = {binding})`, but {} texture(s) were provided.", textures.len())));
		}
		Ok(textures)
	}

	/// Get specific number of uniform buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_uniform_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<&[Arc<dyn GenericUniformBuffer>], VulkanError> {
		let uniform_buffers = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?.get_uniform_buffers()?;
		if uniform_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} uniform buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} uniform buffer(s) were provided.", uniform_buffers.len())));
		}
		Ok(uniform_buffers)
	}

	/// Get specific number of uniform buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_storage_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<&[Arc<dyn GenericStorageBuffer>], VulkanError> {
		let storage_buffers = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?.get_storage_buffers()?;
		if storage_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} storage buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} storage buffer(s) were provided.", storage_buffers.len())));
		}
		Ok(storage_buffers)
	}

	/// Get specific number of uniform texel buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_uniform_texel_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<&[VulkanBufferView], VulkanError> {
		let uniform_texel_buffers = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?.get_uniform_texel_buffers()?;
		if uniform_texel_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} uniform buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} uniform buffer(s) were provided.", uniform_texel_buffers.len())));
		}
		Ok(uniform_texel_buffers)
	}

	/// Get specific number of uniform texel buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_storage_texel_buffers(&self, set: u32, binding: u32, desired_count: usize) -> Result<&[VulkanBufferView], VulkanError> {
		let storage_texel_buffers = self.get(set, binding).ok_or(VulkanError::MissingShaderInputs(format!("layout(set = {set}, binding = {binding})")))?.get_storage_texel_buffers()?;
		if storage_texel_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} storage buffer(s) is needed for `layout(set = {set}, binding = {binding})`, but {} storage buffer(s) were provided.", storage_texel_buffers.len())));
		}
		Ok(storage_texel_buffers)
	}
}
