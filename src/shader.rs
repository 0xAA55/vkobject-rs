
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	fs::read,
	mem::forget,
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

/// The shader source
#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
	VertexShader(&'a str),
	GeometryShader(&'a str),
	FragmentShader(&'a str),
	ComputeShader(&'a str)
}

/// The shader source path
#[derive(Debug, Clone, Copy)]
pub enum ShaderSourcePath<'a> {
	VertexShader(&'a Path),
	GeometryShader(&'a Path),
	FragmentShader(&'a Path),
	ComputeShader(&'a Path)
}

/// The optimization level for shaderc
#[cfg(feature = "shaderc")]
pub use shaderc::OptimizationLevel;

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
		let mut shader_bytes = read(shader_file)?;
		shader_bytes.resize(((shader_bytes.len() - 1) / 4 + 1) * 4, 0);
		let shader_code = unsafe {
			let ptr = shader_bytes.as_mut_ptr() as *mut u32;
			let len = shader_bytes.len() >> 2;
			let cap = shader_bytes.capacity() >> 2;
			forget(shader_bytes);
			Vec::from_raw_parts(ptr, len, cap)
		};
		Self::new(device, &shader_code)
	}

	/// Create the `VulkanShader` from source code
	#[cfg(feature = "shaderc")]
	pub fn new_from_source(device: Arc<VulkanDevice>, code: ShaderSource, filename: &str, entry_point: &str, level: OptimizationLevel, debug_info: bool, warning_as_error: bool) -> Result<Self, VulkanError> {
		use shaderc::*;
		use ShaderSource::*;
		let compiler = Compiler::new()?;
		let mut options = CompileOptions::new()?;
		options.set_optimization_level(level);
		if debug_info {options.set_generate_debug_info()}
		if warning_as_error {options.set_warnings_as_errors()}
		options.set_target_env(TargetEnv::Vulkan, device.vkcore.get_app_info().apiVersion);
		let artifact = match code {
			VertexShader(source) => compiler.compile_into_spirv(source, ShaderKind::Vertex, filename, entry_point, Some(&options))?,
			GeometryShader(source) => compiler.compile_into_spirv(source, ShaderKind::Geometry, filename, entry_point, Some(&options))?,
			FragmentShader(source) => compiler.compile_into_spirv(source, ShaderKind::Fragment, filename, entry_point, Some(&options))?,
			ComputeShader(source) => compiler.compile_into_spirv(source, ShaderKind::Compute, filename, entry_point, Some(&options))?,
		};
		Self::new(device, artifact.as_binary())
	}

	/// Create the `VulkanShader` from source code
	#[cfg(feature = "shaderc")]
	pub fn new_from_source_file(device: Arc<VulkanDevice>, code_path: ShaderSourcePath, entry_point: &str, level: OptimizationLevel, debug_info: bool, warning_as_error: bool) -> Result<Self, VulkanError> {
		use ShaderSourcePath::*;
		match code_path {
			VertexShader(path) =>   {let bytes = read(path)?; let source = unsafe {str::from_utf8_unchecked(&bytes)}; Self::new_from_source(device, ShaderSource::VertexShader(source),   &path.file_name().unwrap().to_string_lossy().to_owned(), entry_point, level, debug_info, warning_as_error)}
			GeometryShader(path) => {let bytes = read(path)?; let source = unsafe {str::from_utf8_unchecked(&bytes)}; Self::new_from_source(device, ShaderSource::GeometryShader(source), &path.file_name().unwrap().to_string_lossy().to_owned(), entry_point, level, debug_info, warning_as_error)}
			FragmentShader(path) => {let bytes = read(path)?; let source = unsafe {str::from_utf8_unchecked(&bytes)}; Self::new_from_source(device, ShaderSource::FragmentShader(source), &path.file_name().unwrap().to_string_lossy().to_owned(), entry_point, level, debug_info, warning_as_error)}
			ComputeShader(path) =>  {let bytes = read(path)?; let source = unsafe {str::from_utf8_unchecked(&bytes)}; Self::new_from_source(device, ShaderSource::ComputeShader(source),  &path.file_name().unwrap().to_string_lossy().to_owned(), entry_point, level, debug_info, warning_as_error)}
		}
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
