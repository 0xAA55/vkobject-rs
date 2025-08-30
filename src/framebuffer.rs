
use crate::prelude::*;
use std::{
	ptr::null,
	sync::{Mutex, Arc, Weak},
};

/// A framebuffer
#[derive(Debug)]
pub struct VulkanFramebuffer {
	/// The `VulkanContext` that helps to manage the resources of the swapchain image
	pub(crate) ctx: Weak<Mutex<VulkanContext>>,

	/// The size of the framebuffer
	size: VkExtent2D,

	/// The handle to the framebuffer
	framebuffer: VkFramebuffer,
}

impl VulkanFramebuffer {
	/// Create the `VulkanFramebuffer`
	pub fn new_(vkcore: &VkCore, device: VkDevice, image: VkImage, extent: &VkExtent2D) -> Result<Self, VulkanError> {
		let framebuffer_ci = VkFramebufferCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			renderPass: VkRenderPass,
			attachmentCount: u32,
			pAttachments: ,
			width: extent.width,
			height: extent.height,
			layers: 1,
		};
	}

	/// Set the `VulkanContext` if it hasn't provided previously
	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.ctx = ctx;
	}
}