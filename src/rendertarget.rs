
use crate::prelude::*;
use std::{
	sync::{Arc, Mutex},
};

/// The resources for a texture to become a render-target texture
#[derive(Debug)]
pub struct RenderTargetProps {
	/// The render pass object
	pub renderpass: VulkanRenderPass,

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

	/// The fence of submitting commands to a queue
	pub queue_submit_fence: Arc<VulkanFence>,
}

impl RenderTargetProps {
	pub fn new(device: Arc<VulkanDevice>, extent: &VkExtent2D, attachments: &[Arc<VulkanTexture>]) -> Result<Self, VulkanError> {
		let renderpass_attachments: Vec<VulkanRenderPassAttachment> = attachments.iter().map(|t| {
			VulkanRenderPassAttachment::new(t.format, t.type_size.is_depth_stencil())
		}).collect();
		let framebuffer_attachments: Vec<VkImageView> = attachments.iter().map(|t| {
			t.image_view
		}).collect();
		let renderpass = VulkanRenderPass::new(device.clone(), &renderpass_attachments)?;
		let framebuffer = VulkanFramebuffer::new(device.clone(), extent, renderpass.get_vk_renderpass(), &framebuffer_attachments)?;
		Ok(Self {
			renderpass,
			framebuffer,
			attachments: attachments.iter().map(|t|t.clone()).collect(),
			extent: *extent,
			acquire_semaphore: Arc::new(Mutex::new(VulkanSemaphore::new(device.clone())?)),
			release_semaphore: Arc::new(VulkanSemaphore::new(device.clone())?),
			queue_submit_fence: Arc::new(VulkanFence::new(device.clone())?),
		})
	}
}
