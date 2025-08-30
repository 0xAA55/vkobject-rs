
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
	pub fn new_(vkcore: &VkCore, device: VkDevice, extent: &VkExtent2D, render_pass: VkRenderPass, attachments: &[VkImageView]) -> Result<Self, VulkanError> {
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
		vkcore.vkCreateFramebuffer(device, &framebuffer_ci, null(), &mut framebuffer)?;
		Ok(Self {
			ctx: Weak::new(),
			size: *extent,
			framebuffer,
		})
	}

	/// Create the `VulkanFramebuffer`
	pub fn new(ctx: Arc<Mutex<VulkanContext>>, extent: &VkExtent2D, render_pass: VkRenderPass, attachments: &[VkImageView]) -> Result<Self, VulkanError> {
		let lock = ctx.lock().unwrap();
		let vkcore = lock.vkcore.clone();
		let device = lock.get_vk_device();
		drop(lock);
		let mut ret = Self::new_(&vkcore, device, extent, render_pass, attachments)?;
		ret.set_ctx(Arc::downgrade(&ctx));
		Ok(ret)
	}

	/// Set the `VulkanContext` if it hasn't provided previously
	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.ctx = ctx;
	}
}

impl Drop for VulkanFramebuffer {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			vkcore.vkDestroyFramebuffer(ctx.get_vk_device(), self.framebuffer, null()).unwrap();
		}
	}
}
