
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::Arc,
};

/// The renderpass attachment
#[derive(Debug, Clone, Copy)]
pub struct VulkanRenderPassAttachment {
	/// The format of the attachment
	pub format: VkFormat,

	/// Is this attachment for depth stencil?
	pub is_depth_stencil: bool,
}

impl VulkanRenderPassAttachment {
	/// Create a new `VulkanRenderPassAttachment`
	pub fn new(format: VkFormat, is_depth_stencil: bool) -> Self {
		Self {
			format,
			is_depth_stencil,
		}
	}

	/// Convert to `VkAttachmentDescription`
	pub(crate) fn to_attachment_desc(&self) -> VkAttachmentDescription {
		VkAttachmentDescription {
			flags: 0,
			format: self.format,
			samples: VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT,
			loadOp: VkAttachmentLoadOp::VK_ATTACHMENT_LOAD_OP_CLEAR,
			storeOp: VkAttachmentStoreOp::VK_ATTACHMENT_STORE_OP_STORE,
			stencilLoadOp: VkAttachmentLoadOp::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			stencilStoreOp: VkAttachmentStoreOp::VK_ATTACHMENT_STORE_OP_DONT_CARE,
			initialLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
			finalLayout: if !self.is_depth_stencil {VkImageLayout::VK_IMAGE_LAYOUT_PRESENT_SRC_KHR} else {VkImageLayout::VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
		}
	}
}

/// The wrapper for `VkRenderPass`
pub struct VulkanRenderPass {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The render pass attachments
	pub attachments: Vec<VulkanRenderPassAttachment>,

	/// The handle to the renderpass object
	renderpass: VkRenderPass,
}

impl VulkanRenderPass {
	/// Create the `VulkanRenderPass`
	pub fn new(device: Arc<VulkanDevice>, attachments: &[VulkanRenderPassAttachment]) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let attachment_descs: Vec<VkAttachmentDescription> = attachments.iter().map(|a|a.to_attachment_desc()).collect();
		let color_attachment_refs: Vec<VkAttachmentReference> = attachments.iter().enumerate().filter_map(|(i, a)|
			if !a.is_depth_stencil {
				Some(VkAttachmentReference {
					attachment: i as u32,
					layout: VkImageLayout::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
				})
			} else {
				None
			}
		).collect();
		let depth_attachment_ref: Option<VkAttachmentReference> = attachments.iter().enumerate().filter_map(|(i, a)|
			if a.is_depth_stencil {
				Some(VkAttachmentReference {
					attachment: i as u32,
					layout: VkImageLayout::VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
				})
			} else {
				None
			}
		).next();
		let subpass_desc = VkSubpassDescription {
			flags: 0,
			pipelineBindPoint: VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_GRAPHICS,
			inputAttachmentCount: 0,
			pInputAttachments: null(),
			colorAttachmentCount: color_attachment_refs.len() as u32,
			pColorAttachments: color_attachment_refs.as_ptr(),
			pResolveAttachments: null(),
			pDepthStencilAttachment: if let Some(depth) = &depth_attachment_ref {
				depth
			} else {
				null()
			},
			preserveAttachmentCount: 0,
			pPreserveAttachments: null(),
		};
		let dependencies: [VkSubpassDependency; 2] = [
			VkSubpassDependency {
				srcSubpass: VK_SUBPASS_EXTERNAL,
				dstSubpass: 0,
				srcStageMask:
					VkPipelineStageFlagBits::VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT as VkPipelineStageFlags |
					VkPipelineStageFlagBits::VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT as VkPipelineStageFlags,
				dstStageMask:
					VkPipelineStageFlagBits::VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT as VkPipelineStageFlags |
					VkPipelineStageFlagBits::VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT as VkPipelineStageFlags,
				srcAccessMask:
					VkAccessFlagBits::VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT as VkPipelineStageFlags,
				dstAccessMask:
					VkAccessFlagBits::VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT as VkPipelineStageFlags |
					VkAccessFlagBits::VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT as VkPipelineStageFlags,
				dependencyFlags: 0,
			},
			VkSubpassDependency {
				srcSubpass: VK_SUBPASS_EXTERNAL,
				dstSubpass: 0,
				srcStageMask: VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT as VkPipelineStageFlags,
				dstStageMask: VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT as VkPipelineStageFlags,
				srcAccessMask: 0,
				dstAccessMask:
					VkAccessFlagBits::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT as VkAccessFlags |
					VkAccessFlagBits::VK_ACCESS_COLOR_ATTACHMENT_READ_BIT as VkAccessFlags,
				dependencyFlags: 0,
			},
		];
		let renderpass_ci = VkRenderPassCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			pNext: null(),
			flags: 0,
			attachmentCount: attachment_descs.len() as u32,
			pAttachments: attachment_descs.as_ptr(),
			subpassCount: 1,
			pSubpasses: &subpass_desc,
			dependencyCount: dependencies.len() as u32,
			pDependencies: dependencies.as_ptr(),
		};
		let mut renderpass: VkRenderPass = null();
		vkcore.vkCreateRenderPass(device.get_vk_device(), &renderpass_ci, null(), &mut renderpass)?;
		Ok(Self {
			device,
			attachments: attachments.iter().collect(),
			renderpass,
		})
	}

	/// Get the `VkRenderPass`
	pub(crate) fn get_vk_renderpass(&self) -> VkRenderPass {
		self.renderpass
	}
}

impl Debug for VulkanRenderPass {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanRenderPass")
		.field("renderpass", &self.renderpass)
		.finish()
	}
}

impl Drop for VulkanRenderPass {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroyRenderPass(self.device.get_vk_device(), self.renderpass, null()).unwrap();
	}
}
