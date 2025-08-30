
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

/// The Vulkan renderpass
pub mod renderpass;

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
	pub use crate::renderpass::*;
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

		let mut ctx = create_vulkan_context(&window, true, 3, false).unwrap();

		let start_time = glfw.get_time();
		let mut num_frames: u64 = 0;
		let mut time_in_sec: u64 = 0;
		while !window.should_close() {
			let cur_frame_time = glfw.get_time();
			let run_time = cur_frame_time - start_time;
			ctx.on_resize().unwrap();
			let frame = ctx.begin_frame(true).unwrap();

			drop(frame);
			num_frames += 1;
			let new_time_in_sec = run_time.floor() as u64;
			if new_time_in_sec > time_in_sec {
				let fps = num_frames as f64 / run_time;
				println!("Avr FPS: {fps}\tat {new_time_in_sec}\r");
				time_in_sec = new_time_in_sec;
			}

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
				if run_time >= test_time {
					window.set_should_close(true)
				}
			}
		}
	}
}
