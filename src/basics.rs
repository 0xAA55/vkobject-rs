
use crate::prelude::*;
use std::{
	ptr::{null, null_mut},
	rc::Rc,
};

pub struct VulkanGpuInfo {
	gpu: VkPhysicalDevice,
	queue_families: Vec<VkQueueFamilyProperties>,
}

impl VkQueueFamilyProperties {
	pub fn can_graphic(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT) == VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT
	}
	pub fn can_compute(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT) == VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT
	}
	pub fn can_transfer(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_TRANSFER_BIT) == VkQueueFlagBits::VK_QUEUE_TRANSFER_BIT
	}
	pub fn can_sparse_binding(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_SPARSE_BINDING_BIT) == VkQueueFlagBits::VK_QUEUE_SPARSE_BINDING_BIT
	}
	pub fn can_protected(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_PROTECTED_BIT) == VkQueueFlagBits::VK_QUEUE_PROTECTED_BIT
	}
	pub fn can_video_decode(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_VIDEO_DECODE_BIT_KHR) == VkQueueFlagBits::VK_QUEUE_VIDEO_DECODE_BIT_KHR
	}
	pub fn can_video_encode(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_VIDEO_ENCODE_BIT_KHR) == VkQueueFlagBits::VK_QUEUE_VIDEO_ENCODE_BIT_KHR
	}
	pub fn can_optical_flow(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_OPTICAL_FLOW_BIT_NV) == VkQueueFlagBits::VK_QUEUE_OPTICAL_FLOW_BIT_NV
	}
	pub fn can_data_graph_flow(&self) -> bool {
		(self.queueFlags & VkQueueFlagBits::VK_QUEUE_DATA_GRAPH_BIT_ARM) == VkQueueFlagBits::VK_QUEUE_DATA_GRAPH_BIT_ARM
	}
}

impl VulkanGpuInfo{
	pub fn get_gpu_info(vkcore: Rc<VkCore>) -> Result<Vec<VulkanGpuInfo>, VkError> {
		let mut gpu_count = 0u32;
		vkcore.vkEnumeratePhysicalDevices(vkcore.instance, &mut gpu_count, null_mut::<_>())?;
		let mut gpus = Vec::<VkPhysicalDevice>::with_capacity(gpu_count as usize);
		vkcore.vkEnumeratePhysicalDevices(vkcore.instance, &mut gpu_count, &mut gpus[0])?;
		unsafe {gpus.set_len(gpu_count as usize)};
		let mut ret = Vec::<VulkanGpuInfo>::with_capacity(gpu_count as usize);
		for gpu in gpus {
			let mut queue_family_count = 0u32;
			vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut queue_family_count, null())?;
			let mut queue_families = Vec::<VkQueueFamilyProperties>::with_capacity(queue_family_count as usize);
			vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut queue_family_count, &mut queue_families[0])?;
			unsafe {queue_families.set_len(queue_family_count as usize)};
			ret.push(VulkanGpuInfo {
				gpu,
				queue_families,
			});
		}
		ret
	}

	pub fn get_gpu(&self) -> VkPhysicalDevice {
		self.gpu
	}

	pub fn get_queue_families(&self) -> &[VkQueueFamilyProperties] {
		self.queue_families.as_ref()
	}

	pub fn get_queue_family_index(&self, queue_flag_match: u32) -> u32 {
		for i in 0..self.queue_families.len() {
			if (self.queue_families[i].queueFlags & queue_flag_match) == queue_flag_match {
				return i as u32;
			}
		}
		u32::MAX
	}
}

pub struct VulkanDevice {
	pub vkcore: Rc<VkCore>,
	pub device: VkDevice,
}

impl VulkanDevice {
	pub fn new(vkcore: Rc<VkCore>, gpu: VkPhysicalDevice, queue_family_index: u32) -> Result<Self, VkError> {
		let priorities = vec![1.0];
		let queue_create_info = VkDeviceQueueCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			queueFamilyIndex: queue_family_index,
			queueCount: 1,
			pQueuePriorities: priorities.as_ptr(),
		};

		let mut device: VkDevice = null();
		vkCreateDevice(gpu, &device_create_info, null(), &mut device)?;

		Self {
			vkcore,
			device,
		}
	}
}

impl Drop for VulkanDevice {
	fn drop(&mut self) {

	}
}
