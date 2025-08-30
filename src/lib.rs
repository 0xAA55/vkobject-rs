
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

/// The common helper library
pub mod common;

/// The VkCore initializer
pub mod init;

/// The Vulkan basics
pub mod basics;

/// The Vulkan device
pub mod device;

/// The Vulkan surface
pub mod surface;

/// The Vulkan swapchain
pub mod swapchain;

/// The Vulkan framebuffer
pub mod framebuffer;

/// The Vulkan command pool
pub mod cmdpool;

/// The Vulkan context
pub mod context;

/// The buffer object
pub mod buffer;

/// The common things for you to use
pub mod prelude {
	pub use vkcore_rs::*;
	pub use crate::common::*;
	pub use crate::init::*;
	pub use crate::basics::*;
	pub use crate::device::*;
	pub use crate::surface::*;
	pub use crate::swapchain::*;
	pub use crate::framebuffer::*;
	pub use crate::cmdpool::*;
	pub use crate::context::*;
	pub use crate::buffer::*;
}

#[cfg(test)]
mod tests {
	use glfw::*;
	use crate::prelude::*;

	const TEST_TIME: f64 = 10.0;

	#[test]
	fn test() {
		let test_time: Option<f64> = Some(TEST_TIME);
		let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
		glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
		let (mut window, events) = glfw.create_window(1024, 768, "GLFW Window", glfw::WindowMode::Windowed).expect("Failed to create VKFW window.");

		window.set_key_polling(true);

		let ctx = create_vulkan_context(&window, true, 3, false).unwrap();
		dbg!(ctx);

		let start_time = glfw.get_time();
		while !window.should_close() {
			let cur_frame_time = glfw.get_time();

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
