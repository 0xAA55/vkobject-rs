
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	fs::read,
	mem::forget,
	path::Path,
	ptr::null,
	sync::Arc,
};

/// The optimization level for shaderc
#[cfg(feature = "shaderc")]
pub use shaderc::OptimizationLevel;

use rspirv::{
	dr::Module,
	spirv::{Decoration, Op, StorageClass, Word},
};

/// The input and output of the shader
#[derive(Debug, Clone)]
pub struct ShaderVariable {
	/// The type of the variable
	pub var_type: String,

	/// The name of the variable
	pub var_name: Option<String>,

	/// The storage class of the variable
	pub storage_class: StorageClass,

	/// The location of the variable
	pub location: Option<u32>,
}

/// The wrapper for `VkShaderModule`
pub struct VulkanShader {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the shader
	shader: VkShaderModule,

	/// The parsed variables of the shader
	vars: Vec<ShaderVariable>,
}

/// The shader source
#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
	/// Vertex shader source code
	VertexShader(&'a str),

	/// Geometry shader source code
	GeometryShader(&'a str),

	/// Fragment shader source code
	FragmentShader(&'a str),

	/// Compute shader source code
	ComputeShader(&'a str),
}

/// The shader source
#[derive(Debug, Clone)]
pub enum ShaderSourceOwned {
	/// Vertex shader source code
	VertexShader(String),

	/// Geometry shader source code
	GeometryShader(String),

	/// Fragment shader source code
	FragmentShader(String),

	/// Compute shader source code
	ComputeShader(String),
}

/// The shader source path
#[derive(Debug, Clone, Copy)]
pub enum ShaderSourcePath<'a> {
	/// Vertex shader source code file path
	VertexShader(&'a Path),

	/// Geometry shader source code file path
	GeometryShader(&'a Path),

	/// Fragment shader source code file path
	FragmentShader(&'a Path),

	/// Compute shader source code file path
	ComputeShader(&'a Path),
}

impl ShaderSourcePath<'_> {
	pub fn load(&self) -> Result<ShaderSourceOwned, VulkanError> {
		Ok(match self {
			Self::VertexShader(path) => {let bytes = read(path)?; ShaderSourceOwned::VertexShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::GeometryShader(path) => {let bytes = read(path)?; ShaderSourceOwned::GeometryShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::FragmentShader(path) => {let bytes = read(path)?; ShaderSourceOwned::FragmentShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::ComputeShader(path) => {let bytes = read(path)?; ShaderSourceOwned::ComputeShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
		})
	}

	pub fn get_filename(&self) -> String {
		match self {
			Self::VertexShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
			Self::GeometryShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
			Self::FragmentShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
			Self::ComputeShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
		}
	}
}

impl ShaderSourceOwned {
	pub fn as_ref<'a>(&'a self) -> ShaderSource<'a> {
		match self {
			Self::VertexShader(string) => ShaderSource::VertexShader(string),
			Self::GeometryShader(string) => ShaderSource::GeometryShader(string),
			Self::FragmentShader(string) => ShaderSource::FragmentShader(string),
			Self::ComputeShader(string) => ShaderSource::ComputeShader(string),
		}
	}
}

/// Get the string of a target_id
fn get_name(module: &Module, target_id: Word) -> Option<String> {
	for inst in module.debug_names.iter() {
		if inst.class.opcode == Op::Name && inst.operands[0].unwrap_id_ref() == target_id {
			return Some(inst.operands[1].unwrap_literal_string().to_string());
		}
	}
	None
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

	/// Compile shader code to binary
	#[cfg(feature = "shaderc")]
	pub fn compile(code: ShaderSource, filename: &str, entry_point: &str, level: OptimizationLevel, debug_info: bool, warning_as_error: bool) -> Result<Vec<u32>, VulkanError> {
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
		Ok(artifact.as_binary().to_vec())
	}

	/// Create the `VulkanShader` from source code
	#[cfg(feature = "shaderc")]
	pub fn new_from_source(device: Arc<VulkanDevice>, code: ShaderSource, filename: &str, entry_point: &str, level: OptimizationLevel, debug_info: bool, warning_as_error: bool) -> Result<Self, VulkanError> {
		Self::new(device, &Self::compile(code, filename, entry_point, level, debug_info, warning_as_error))
	}

	/// Create the `VulkanShader` from source code
	#[cfg(feature = "shaderc")]
	pub fn new_from_source_file(device: Arc<VulkanDevice>, code_path: ShaderSourcePath, entry_point: &str, level: OptimizationLevel, debug_info: bool, warning_as_error: bool) -> Result<Self, VulkanError> {
		Self::new_from_source(device, code_path.load()?.as_ref(), &code_path.get_filename(), entry_point, level, debug_info, warning_as_error)
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
