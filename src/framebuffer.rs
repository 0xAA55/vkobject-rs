
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::Arc,
};

/// A framebuffer
pub struct VulkanFramebuffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The size of the framebuffer
	size: VkExtent2D,

	/// The handle to the framebuffer
	framebuffer: VkFramebuffer,
}

impl VulkanFramebuffer {
	/// Create the `VulkanFramebuffer`
	pub(crate) fn new(device: Arc<VulkanDevice>, extent: &VkExtent2D, renderpass: VkRenderPass, attachments: &[VkImageView]) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let framebuffer_ci = VkFramebufferCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			renderPass: renderpass,
			attachmentCount: attachments.len() as u32,
			pAttachments: attachments.as_ptr(),
			width: extent.width,
			height: extent.height,
			layers: 1,
		};
		let mut framebuffer: VkFramebuffer = null();
		vkcore.vkCreateFramebuffer(device.get_vk_device(), &framebuffer_ci, null(), &mut framebuffer)?;
		Ok(Self {
			device,
			size: *extent,
			framebuffer,
		})
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
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroyFramebuffer(self.device.get_vk_device(), self.framebuffer, null()).unwrap();
	}
}
