
use crate::prelude::*;
use std::{
	sync::{Arc, Mutex},
};

/// The resources for a texture to become a render-target texture
#[derive(Debug)]
pub struct RenderTargetProps {
	/// The render pass object
	pub renderpass: Arc<VulkanRenderPass>,

	/// The framebuffer object
	pub framebuffer: VulkanFramebuffer,

	/// The extent of the render target
	pub(crate) extent: VkExtent2D,

	/// The attachment textures
	pub attachments: Vec<Arc<VulkanTexture>>,

	/// The semaphore for waiting the render target to be ready
	pub acquire_semaphore: Arc<Mutex<VulkanSemaphore>>,

	/// The semaphore indicating that the render commands were runned
	pub release_semaphore: Arc<VulkanSemaphore>,
}

impl RenderTargetProps {
	pub fn new(device: Arc<VulkanDevice>, extent: &VkExtent2D, renderpass: Option<Arc<VulkanRenderPass>>, attachments: &[Arc<VulkanTexture>]) -> Result<Self, VulkanError> {
		let renderpass_attachments: Vec<VulkanRenderPassAttachment> = attachments.iter().map(|t| {
			VulkanRenderPassAttachment::new(t.format, t.type_size.is_depth_stencil())
		}).collect();
		let renderpass = if let Some(renderpass) = renderpass {
			renderpass.clone()
		} else {
			Arc::new(VulkanRenderPass::new(device.clone(), &renderpass_attachments)?)
		};
		let framebuffer = VulkanFramebuffer::new(device.clone(), extent, &renderpass, attachments)?;
		Ok(Self {
			renderpass,
			framebuffer,
			attachments: attachments.to_vec(),
			extent: *extent,
			acquire_semaphore: Arc::new(Mutex::new(VulkanSemaphore::new(device.clone())?)),
			release_semaphore: Arc::new(VulkanSemaphore::new(device.clone())?),
		})
	}

	pub fn get_extent(&self) -> &VkExtent2D {
		&self.extent
	}

	pub(crate) fn get_vk_renderpass(&self) -> VkRenderPass {
		self.renderpass.get_vk_renderpass()
	}

	pub(crate) fn get_vk_framebuffer(&self) -> VkFramebuffer {
		self.framebuffer.get_vk_framebuffer()
	}
}

impl Drop for RenderTargetProps {
	fn drop(&mut self) {
		let device = self.renderpass.device.clone();
		device.vkcore.vkQueueWaitIdle(device.get_vk_queue()).unwrap();
	}
}

unsafe impl Send for RenderTargetProps {}
unsafe impl Sync for RenderTargetProps {}
