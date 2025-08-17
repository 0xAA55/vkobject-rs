
#![allow(unused_imports)]

/// The common helper library
pub mod common;

/// The VkCore initializer
pub mod init;

/// The Vulkan basics
pub mod basics;

/// The buffer object
pub mod buffer;

/// The common things for you to use
pub mod prelude {
	pub use vkcore_rs::*;
	pub use crate::common::*;
	pub use crate::init::*;
	pub use crate::basics::*;
	pub use crate::buffer::*;
}

#[cfg(test)]
mod tests {
	use std::{
		rc::Rc,
	};
	use glfw::*;
	use crate::prelude::*;

	const TEST_TIME: f64 = 10.0;

	#[test]
	fn test() {
		let test_time: Option<f64> = Some(TEST_TIME);
		let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
		glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi);
		let (mut window, events) = glfw.create_window(1024, 768, "GLFW Window", glfw::WindowMode::Windowed).expect("Failed to create VKFW window.");

		window.set_key_polling(true);
		window.make_current();
		glfw.set_swap_interval(SwapInterval::Adaptive);

		let vkcore = Rc::new(create_vkcore_from_glfw("VkObject-test", "VkObject-rs", vk_make_version(1, 0, 0), vk_make_version(1, 0, 0), vk_make_api_version(0, 1, 3, 0)));
		dbg!(VulkanGpuInfo::get_gpu_info(vkcore.clone()).unwrap());

		let start_time = glfw.get_time();
		while !window.should_close() {
			let cur_frame_time = glfw.get_time();

			window.swap_buffers();
			glfw.poll_events();
			for (_, event) in glfw::flush_messages(&events) {
				match event {
					glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
						window.set_should_close(true)
					}
					_ => {}
				}
			}
			if let Some(test_time) = test_time {
				if cur_frame_time - start_time >= test_time {
					window.set_should_close(true)
				}
			}
		}
	}
}
