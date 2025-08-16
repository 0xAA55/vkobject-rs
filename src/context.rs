
use crate::prelude::*;
use std::{
	ptr::null,
	rc::Rc,
};

pub struct VulkanContext {
	pub vkcore: Rc<VkCore>,


	pub vkcmdpool: Rc<VkCommandPool>,
	pub device: Rc<VkDevice>,
}

impl VulkanContext {
	pub fn new(vkcore: Rc<VkCore>) -> Self {









		let priorities = vec![1.0];
		let queue_create_info = VkDeviceQueueCreateInfo {
			sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			queueFamilyIndex: 0,
			queueCount: 1,
			pQueuePriorities: priorities.as_ptr(),
		};

		Self {
			vkcore,

		}
	}
}
