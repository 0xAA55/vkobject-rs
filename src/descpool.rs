
use crate::prelude::*;
use std::{
	collections::HashMap,
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::{Arc,
		atomic::{AtomicU32, Ordering},
	},
};

/// The size info for creating a descriptor pool
#[derive(Debug)]
pub struct DescriptorPoolSize {
	/// How many descriptor sets
	pub max_sets: u32,

	/// The detailed capacity for desciptor types
	pub typed_capacity: HashMap<VkDescriptorType, u32>,
}

impl DescriptorPoolSize {
	/// Create the `DescriptorPoolSize`
	pub fn new(max_sets: u32, typed_capacity: HashMap<VkDescriptorType, u32>) -> Self {
		Self {
			max_sets,
			typed_capacity,
		}
	}
}

impl Default for DescriptorPoolSize {
	fn default() -> Self {
		Self {
			max_sets: 1000,
			typed_capacity: [
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLER, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_TENSOR_ARM, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_MUTABLE_EXT, 1000),
				(VkDescriptorType::VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV, 1000),
			].into_iter().collect(),
		}
	}
}

/// The global descriptor pool
pub struct DescriptorPool {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The usage of the pool
	pub pool_capacity: Arc<HashMap<VkDescriptorType, u32>>,

	/// The usage of the pool
	pub pool_usage: Arc<HashMap<VkDescriptorType, AtomicU32>>,

	/// Max sets
	max_sets: u32,

	/// The pool
	pool: VkDescriptorPool,
}

impl DescriptorPool {
	/// Create a pool
	pub fn new(device: Arc<VulkanDevice>, capacity: DescriptorPoolSize) -> Result<Self, VulkanError> {
		let pool_sizes: Vec<VkDescriptorPoolSize> = capacity.typed_capacity.iter().map(|(&type_, &count)| VkDescriptorPoolSize {type_, descriptorCount: count}).collect();
		let pool_capacity: Arc<HashMap<VkDescriptorType, u32>> = Arc::new(capacity.typed_capacity.iter().map(|(&t, &c)|(t, c)).collect());
		let pool_usage: Arc<HashMap<VkDescriptorType, AtomicU32>> = Arc::new(capacity.typed_capacity.iter().map(|(&type_, _)|(type_, AtomicU32::new(0))).collect());
		let pool_ci = VkDescriptorPoolCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			pNext: null(),
			flags: 0,
			maxSets: capacity.max_sets,
			poolSizeCount: pool_sizes.len() as u32,
			pPoolSizes: pool_sizes.as_ptr(),
		};
		let mut pool = null();
		device.vkcore.vkCreateDescriptorPool(device.get_vk_device(), &pool_ci, null(), &mut pool)?;
		Ok(Self {
			device,
			pool_capacity,
			pool_usage,
			max_sets: capacity.max_sets,
			pool,
		})
	}

	/// Get the `VkDescriptorPool`
	pub(crate) fn get_vk_pool(&self) -> VkDescriptorPool {
		self.pool
	}

	/// Get max sets of this pool
	pub fn get_max_sets(&self) -> u32 {
		self.max_sets
	}

	/// Get the capacity of a descriptor type
	pub fn get_capacity(&self, key: VkDescriptorType) -> u32 {
		if let Some(capacity) = self.pool_capacity.get(&key) {
			*capacity
		} else {
			0
		}
	}

	/// Increase the usage of a descriptor type
	///
	/// # Panic
	///
	/// Panic if the `key` does not exist in the map.
	pub fn incr_usage(&mut self, key: VkDescriptorType, incr_count: u32) {
		if let Some(usage) = self.pool_usage.get(&key) {
			usage.fetch_add(incr_count, Ordering::Release);
		} else {
			panic!("[PANIC] You have not allocated such a type of descriptor `{key:?}` for the descriptor pool to allocate.");
		}
	}

	/// Decrease the usage of a descriptor type
	///
	/// # Panic
	///
	/// Panic if the `key` does not exist in the map.
	pub fn decr_usage(&mut self, key: VkDescriptorType, decr_count: u32) {
		if let Some(usage) = self.pool_usage.get(&key) {
			usage.fetch_sub(decr_count, Ordering::Release);
		} else {
			panic!("[PANIC] You have not allocated such a type of descriptor `{key:?}` for the descriptor pool to allocate.");
		}
	}

	/// Get the usage of a descriptor type
	pub fn get_usage(&self, key: VkDescriptorType) -> u32 {
		if let Some(ref mut usage) = self.pool_usage.get(&key) {
			usage.load(Ordering::Acquire)
		} else {
			0
		}
	}
}

impl Debug for DescriptorPool {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("DescriptorPool")
		.field("pool_capacity", &self.pool_capacity)
		.field("pool_usage", &self.pool_usage)
		.field("max_sets", &self.max_sets)
		.field("pool", &self.pool)
		.finish()
	}
}

impl Drop for DescriptorPool {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyDescriptorPool(self.device.get_vk_device(), self.pool, null()).unwrap();
	}
}
