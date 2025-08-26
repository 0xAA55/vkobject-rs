
#[cfg(any(feature = "glfw", test))]
pub mod init_from_glfw {
	use crate::prelude::*;
	use vkcore_rs::*;
	use std::{
		ffi::{c_void, CString},
		ptr::null,
		sync::{Arc, Mutex},
	};

	unsafe extern "C" {
		fn glfwGetInstanceProcAddress(instance: VkInstance, procname: *const i8) -> *const c_void;
	}

	/// Create a `VkCore` from GLFW
	pub fn create_vkcore_from_glfw(app_name: &str, engine_name: &str, app_version: u32, engine_version: u32, api_version: u32) -> Result<VkCore, VulkanError> {
		let app_name = CString::new(app_name).unwrap();
		let engine_name = CString::new(engine_name).unwrap();
		let app_info = VkApplicationInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_APPLICATION_INFO,
			pNext: null(),
			pApplicationName: app_name.as_ptr(),
			applicationVersion: app_version,
			pEngineName: engine_name.as_ptr(),
			engineVersion: engine_version,
			apiVersion: api_version,
		};
		Ok(VkCore::new(app_info, |instance, proc_name|unsafe {glfwGetInstanceProcAddress(instance, CString::new(proc_name).unwrap().as_ptr())})?)
	}

	/// Create a Vulkan context
	pub fn create_vulkan_context(window: &glfw::PWindow, vsync: bool, max_concurrent_frames: usize, is_vr: bool) -> Result<Arc<Mutex<VulkanContext>>, VulkanError> {
		let vkcore = Arc::new(create_vkcore_from_glfw("VkObject-test", "VkObject-rs", vk_make_version(1, 0, 0), vk_make_version(1, 0, 0), vk_make_api_version(0, 1, 4, 0))?);
		let device = VulkanDevice::choose_gpu_with_graphics(vkcore.clone())?;
		let context_ci = VulkanContextCreateInfo {
			vkcore,
			device,
			window,
			vsync,
			max_concurrent_frames,
			is_vr,
		};
		Ok(VulkanContext::new(context_ci)?)
	}
}

#[cfg(any(feature = "glfw", test))]
pub use init_from_glfw::*;
