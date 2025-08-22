
#[cfg(any(feature = "glfw", test))]
pub mod init_from_glfw {
	use vkcore_rs::*;
	use std::{
		ffi::{c_void, CString},
		ptr::null,
	};

	unsafe extern "C" {
		fn glfwGetInstanceProcAddress(instance: VkInstance, procname: *const i8) -> *const c_void;
	}

	/// Create a `VkCore` from GLFW
	pub fn create_vkcore_from_glfw(app_name: &str, engine_name: &str, app_version: u32, engine_version: u32, api_version: u32) -> Result<VkCore, VkError> {
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
		VkCore::new(app_info, |instance, proc_name|unsafe {glfwGetInstanceProcAddress(instance, CString::new(proc_name).unwrap().as_ptr())})
	}
}

#[cfg(any(feature = "glfw", test))]
pub use init_from_glfw::*;
