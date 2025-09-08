
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	fs::read,
	mem::transmute,
	path::Path,
	ptr::null,
	sync::Arc,
};

/// The wrapper for `VkShaderModule`
pub struct VulkanShader {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the shader
	shader: VkShaderModule,
}

impl VulkanShader {
	/// Create the `VulkanShader` from the shader code, it should be aligned to 32-bits
	pub fn new(device: Arc<VulkanDevice>, shader_code: &[u32]) -> Result<Self, VulkanError> {
		let shader_module_ci = VkShaderModuleCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			codeSize: shader_code.len() * 4,
			pCode: shader_code.as_ptr(),
		};
		let mut shader: VkShaderModule = null();
		device.vkcore.vkCreateShaderModule(device.get_vk_device(), &shader_module_ci, null(), &mut shader)?;
		Ok(Self {
			device,
			shader,
		})
	}

	/// Create the `VulkanShader` from file
	pub fn new_from_file(device: Arc<VulkanDevice>, shader_file: &Path) -> Result<Self, VulkanError> {
		let mut shader_code: Vec<u32> = unsafe {transmute::<Vec<u8>, Vec<u32>>(read(shader_file)?)};
		unsafe {shader_code.set_len(shader_code.len() >> 2)};
		Self::new(device, &shader_code)
	}
}

impl Debug for VulkanShader {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanShader")
		.field("shader", &self.shader)
		.finish()
	}
}

impl Drop for VulkanShader {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyShaderModule(self.device.get_vk_device(), self.shader, null()).unwrap();
	}
}
