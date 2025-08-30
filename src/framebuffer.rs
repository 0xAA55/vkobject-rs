
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::{Arc, Mutex},
};

/// A framebuffer
pub struct VulkanFramebuffer {
	/// The `VkCore` is the Vulkan driver
	vkcore: Arc<VkCore>,

	/// The `VulkanDevice` is the associated device
	device: Arc<VulkanDevice>,

	/// The size of the framebuffer
	size: VkExtent2D,

	/// The handle to the framebuffer
	framebuffer: VkFramebuffer,
}

impl VulkanFramebuffer {
	/// Create the `VulkanFramebuffer`
	pub fn new_(vkcore: Arc<VkCore>, device: Arc<VulkanDevice>, extent: &VkExtent2D, render_pass: VkRenderPass, attachments: &[VkImageView]) -> Result<Self, VulkanError> {
		let framebuffer_ci = VkFramebufferCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			renderPass: render_pass,
			attachmentCount: attachments.len() as u32,
			pAttachments: attachments.as_ptr(),
			width: extent.width,
			height: extent.height,
			layers: 1,
		};
		let mut framebuffer: VkFramebuffer = null();
		vkcore.vkCreateFramebuffer(device.get_vk_device(), &framebuffer_ci, null(), &mut framebuffer)?;
		Ok(Self {
			vkcore,
			device,
			size: *extent,
			framebuffer,
		})
	}

	/// Create the `VulkanFramebuffer`
	pub fn new(ctx: Arc<Mutex<VulkanContext>>, extent: &VkExtent2D, render_pass: VkRenderPass, attachments: &[VkImageView]) -> Result<Self, VulkanError> {
		let lock = ctx.lock().unwrap();
		Ok(Self::new_(lock.vkcore.clone(), lock.device.clone(), extent, render_pass, attachments)?)
	}
}

impl Debug for VulkanFramebuffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanFramebuffer")
		.field("size", &self.size)
		.field("framebuffer", &self.framebuffer)
		.finish()
	}
}

impl Drop for VulkanFramebuffer {
	fn drop(&mut self) {
		self.vkcore.vkDestroyFramebuffer(self.device.get_vk_device(), self.framebuffer, null()).unwrap();
	}
}
