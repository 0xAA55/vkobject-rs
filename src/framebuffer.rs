
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
	pub(crate) fn new_from_views(device: Arc<VulkanDevice>, extent: &VkExtent2D, renderpass: VkRenderPass, attachments: &[VkImageView]) -> Result<Self, VulkanError> {
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

	/// Create the `VulkanFramebuffer` from a bunch of textures
	pub(crate) fn new(device: Arc<VulkanDevice>, extent: &VkExtent2D, renderpass: &VulkanRenderPass, attachments: &[Arc<VulkanTexture>]) -> Result<Self, VulkanError> {
		let attachments: Vec<VkImageView> = attachments.iter().map(|t|t.get_vk_image_view()).collect();
		Self::new_from_views(device, extent, renderpass.get_vk_renderpass(), &attachments)
	}
	/// Get the `VkFramebuffer`
	pub(crate) fn get_vk_framebuffer(&self) -> VkFramebuffer {
		self.framebuffer
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
