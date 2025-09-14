
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	fs::read,
	mem::forget,
	path::Path,
	ptr::null,
	slice::from_raw_parts,
	sync::Arc,
};

/// The optimization level for shaderc
#[cfg(feature = "shaderc")]
pub use shaderc::OptimizationLevel;

use rspirv::{
	dr::Module,
	spirv::{Decoration, Op, StorageClass, Word},
};

/// The struct member type
#[derive(Debug, Clone)]
pub struct StructMember {
	member_type: VariableType,
	member_name: String,
}

/// The struct type
#[derive(Debug, Clone)]
pub struct StructType {
	name: String,
	members: Vec<StructMember>,
}

/// The variable type
#[derive(Debug, Clone)]
pub struct VariableArrayType {
	element_type: VariableType,
	element_count: usize,
}

/// The image type
#[derive(Debug, Clone)]
pub struct ImageType {
	result: VariableType,
	dim: String,
	depth: u32,
	arrayed: u32,
	multisample: u32,
	sampled: u32,
	format: String,
}

/// The variable type
#[derive(Debug, Clone)]
pub enum VariableType {
	/// Literal variable
	Literal(String),

	/// Struct
	Struct(StructType),

	/// Array
	Array(Box<VariableArrayType>),

	/// Image
	Image(Box<ImageType>),
}

impl VariableType {
	/// Unwrap for literal variable
	pub fn unwrap_literal(&self) -> &String {
		if let Self::Literal(ret) = self {
			ret
		} else {
			panic!("Expected `VariableType::Literal`, got {self:?}")
		}
	}

	/// Unwrap for struct
	pub fn unwrap_struct(&self) -> &StructType {
		if let Self::Struct(ret) = self {
			ret
		} else {
			panic!("Expected `VariableType::Struct`, got {self:?}")
		}
	}
	/// Unwrap for array
	pub fn unwrap_array(&self) -> &VariableArrayType {
		if let Self::Array(ret) = self {
			ret
		} else {
			panic!("Expected `VariableType::Array`, got {self:?}")
		}
	}
}

/// The input and output of the shader
#[derive(Debug, Clone)]
pub struct ShaderVariable {
	/// The type of the variable
	pub var_type: VariableType,

	/// The name of the variable
	pub var_name: Option<String>,

	/// The storage class of the variable
	pub storage_class: StorageClass,

	/// The location of the variable
	pub location: Option<u32>,

	/// The binding of the variable
	pub binding: Option<u32>,
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

#[derive(Debug, Clone)]
pub struct ShaderAnalyzer {
	module: Module,
}

impl ShaderAnalyzer {
	/// Create a new `ShaderAnalyzer`
	pub fn new(bytes: &[u8]) -> Result<Self, VulkanError> {
		use rspirv::{
			dr::Loader,
			binary::Parser,
		};
		let mut loader = Loader::new();
		Parser::new(bytes, &mut loader).parse()?;
		let module = loader.module();
		Ok(Self {
			module,
		})
	}

	/// Get the string of a target_id
	pub fn get_name(&self, target_id: Word) -> Option<String> {
		for inst in self.module.debug_names.iter() {
			if inst.class.opcode == Op::Name && inst.operands[0].unwrap_id_ref() == target_id {
				let ret = inst.operands[1].unwrap_literal_string().to_string();
				if ret.is_empty() {
					return None;
				} else {
					return Some(ret);
				}
			}
		}
		None
	}

	/// Get the string of a target_id
	pub fn get_member_name(&self, target_id: Word, member_id: u32) -> Option<String> {
		for inst in self.module.debug_names.iter() {
			if inst.class.opcode == Op::MemberName {
				if inst.operands[0].unwrap_id_ref() != target_id || inst.operands[1].unwrap_literal_bit32() != member_id {
					continue;
				}
				let ret = inst.operands[2].unwrap_literal_string().to_string();
				if ret.is_empty() {
					return None;
				} else {
					return Some(ret);
				}
			}
		}
		None
	}

	/// Get the location
	pub fn get_location(&self, target_id: Word) -> Option<u32> {
		for inst in self.module.annotations.iter() {
			if inst.class.opcode != Op::Decorate {
				continue;
			}

			let decorated_id = inst.operands[0].unwrap_id_ref();
			if decorated_id != target_id {
				continue;
			}

			let decoration = inst.operands[1].unwrap_decoration();
			if decoration == Decoration::Location {
				return Some(inst.operands[2].unwrap_literal_bit32());
			}
		}
		None
	}

	/// Get the binding
	pub fn get_binding(&self, target_id: Word) -> Option<u32> {
		for inst in self.module.annotations.iter() {
			if inst.class.opcode != Op::Decorate {
				continue;
			}

			let decorated_id = inst.operands[0].unwrap_id_ref();
			if decorated_id != target_id {
				continue;
			}

			let decoration = inst.operands[1].unwrap_decoration();
			if decoration == Decoration::Binding {
				return Some(inst.operands[2].unwrap_literal_bit32());
			}
		}
		None
	}

	/// Get the string type
	pub fn get_type(&self, type_id: u32) -> Result<VariableType, VulkanError> {
		for inst in self.module.types_global_values.iter() {
			if inst.result_id.unwrap() == type_id {
				return match inst.class.opcode {
					Op::TypePointer => {
						self.get_type(inst.operands[1].unwrap_id_ref())
					}
					Op::TypeBool => Ok(VariableType::Literal("bool".to_string())),
					Op::TypeInt => {
						let signed = inst.operands[1].unwrap_literal_bit32() != 0;
						let width = inst.operands[0].unwrap_literal_bit32();
						Ok(VariableType::Literal(format!("{}{width}", if signed {"i"} else {"u"})))
					}
					Op::TypeFloat => Ok(VariableType::Literal(format!("f{}", inst.operands[0].unwrap_literal_bit32()))),
					Op::TypeVector => {
						let component_type_id = inst.operands[0].unwrap_id_ref();
						let component_count = inst.operands[1].unwrap_literal_bit32();
						let component_type = self.get_type(component_type_id)?;
						match component_type.unwrap_literal().as_str() {
							"f32"  => Ok(VariableType::Literal(format!( "vec{component_count}"))),
							"f64"  => Ok(VariableType::Literal(format!("dvec{component_count}"))),
							"i32"  => Ok(VariableType::Literal(format!("ivec{component_count}"))),
							"u32"  => Ok(VariableType::Literal(format!("uvec{component_count}"))),
							"bool" => Ok(VariableType::Literal(format!("bvec{component_count}"))),
							_ => Err(VulkanError::ShaderParseIdUnknown),
						}
					}
					Op::TypeMatrix => {
						let column_type_id = inst.operands[0].unwrap_id_ref();
						let column_count = inst.operands[1].unwrap_literal_bit32();
						let column_type = self.get_type(column_type_id)?;
						let column_type_name = column_type.unwrap_literal();
						let column_dim = column_type_name.chars().last().unwrap().to_digit(10).unwrap();
						Ok(VariableType::Literal(match &column_type_name[..column_type_name.len() - 1] {
							"vec" => match (column_dim, column_count) {
								(1, 1) | (2, 2) | (3, 3) | (4, 4) => format!("mat{column_dim}"),
								_ => format!("mat{column_dim}{column_count}"),
							}
							"dvec" => match (column_dim, column_count) {
								(1, 1) | (2, 2) | (3, 3) | (4, 4) => format!("dmat{column_dim}"),
								_ => format!("dmat{column_dim}{column_count}"),
							}
							"ivec" => match (column_dim, column_count) {
								(1, 1) | (2, 2) | (3, 3) | (4, 4) => format!("imat{column_dim}"),
								_ => format!("imat{column_dim}{column_count}"),
							}
							"uvec" => match (column_dim, column_count) {
								(1, 1) | (2, 2) | (3, 3) | (4, 4) => format!("umat{column_dim}"),
								_ => format!("umat{column_dim}{column_count}"),
							}
							"bvec" => match (column_dim, column_count) {
								(1, 1) | (2, 2) | (3, 3) | (4, 4) => format!("bmat{column_dim}"),
								_ => format!("bmat{column_dim}{column_count}"),
							}
							_ => format!("{inst:?}"),
						}))
					}
					Op::TypeStruct => {
						let name = self.get_name(type_id).unwrap();
						let mut members: Vec<StructMember> = Vec::with_capacity(inst.operands.len());
						for (i, member) in inst.operands.iter().enumerate() {
							let id = member.unwrap_id_ref();
							let member_name = self.get_member_name(type_id, i as u32).unwrap_or(String::from("_"));
							let member_type = self.get_type(id).unwrap();
							members.push(StructMember {
								member_name,
								member_type,
							});
						}
						Ok(VariableType::Struct(StructType {
							name,
							members,
						}))
					}
					Op::TypeArray => {
						let element_type = self.get_type(inst.operands[0].unwrap_id_ref())?;
						let element_count = inst.operands[1].unwrap_id_ref() as usize;
						Ok(VariableType::Array(Box::new(VariableArrayType {
							element_type,
							element_count,
						})))
					}
					Op::TypeSampledImage => {
						self.get_type(inst.operands[0].unwrap_id_ref())
					}
					Op::TypeImage => {
						Ok(VariableType::Image(Box::new(ImageType {
							result: self.get_type(inst.operands[0].unwrap_id_ref())?,
							dim: format!("{:?}", inst.operands[1].unwrap_dim()),
							depth: inst.operands[2].unwrap_literal_bit32(),
							arrayed: inst.operands[3].unwrap_literal_bit32(),
							multisample: inst.operands[4].unwrap_literal_bit32(),
							sampled: inst.operands[5].unwrap_literal_bit32(),
							format: format!("{:?}", inst.operands[6].unwrap_image_format()),
						})))
					}
					_ => {
						println!("{:#?}", self.module);
						Err(VulkanError::ShaderParseIdUnknown)
					}
				}
			}
		}
		Err(VulkanError::ShaderParseIdUnknown)
	}

	pub fn get_global_vars(&self) -> Result<Vec<ShaderVariable>, VulkanError> {
		let mut vars: Vec<ShaderVariable> = Vec::with_capacity(self.module.types_global_values.len());
		for inst in self.module.global_inst_iter() {
			if inst.class.opcode != Op::Variable {
				continue;
			}

			let var_id = inst.result_id.unwrap();
			let var_type_id = inst.result_type.unwrap();
			let storage_class = inst.operands[0].unwrap_storage_class();

			let var_type = self.get_type(var_type_id).unwrap();
			let var_name = self.get_name(var_id);
			let location = self.get_location(var_id);
			let binding =  self.get_binding(var_id);

			vars.push(ShaderVariable {
				var_type,
				var_name,
				storage_class,
				location,
				binding,
			});
		}
		Ok(vars)
	}
}

impl VulkanShader {
	/// Create the `VulkanShader` from the shader code, it should be aligned to 32-bits
	pub fn new(device: Arc<VulkanDevice>, shader_code: &[u32]) -> Result<Self, VulkanError> {
		let bytes = unsafe {from_raw_parts(shader_code.as_ptr() as *const u8, shader_code.len() * 4)};
		let analyzer = ShaderAnalyzer::new(bytes)?;
		let vars = analyzer.get_global_vars()?;

		let vkdevice = device.get_vk_device();
		let shader_module_ci = VkShaderModuleCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			codeSize: shader_code.len() * 4,
			pCode: shader_code.as_ptr(),
		};
		let mut shader: VkShaderModule = null();
		device.vkcore.vkCreateShaderModule(vkdevice, &shader_module_ci, null(), &mut shader)?;
		Ok(Self {
			device,
			shader,
			vars,
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
	pub fn compile(device: Arc<VulkanDevice>, code: ShaderSource, filename: &str, entry_point: &str, level: OptimizationLevel, warning_as_error: bool) -> Result<Vec<u32>, VulkanError> {
		use shaderc::*;
		use ShaderSource::*;
		let compiler = Compiler::new()?;
		let mut options = CompileOptions::new()?;
		options.set_optimization_level(level);
		options.set_generate_debug_info();
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
	pub fn new_from_source(device: Arc<VulkanDevice>, code: ShaderSource, filename: &str, entry_point: &str, level: OptimizationLevel, warning_as_error: bool) -> Result<Self, VulkanError> {
		let artifact = Self::compile(device.clone(), code, filename, entry_point, level, warning_as_error)?;
		Self::new(device, &artifact)
	}

	/// Create the `VulkanShader` from source code from file
	#[cfg(feature = "shaderc")]
	pub fn new_from_source_file(device: Arc<VulkanDevice>, code_path: ShaderSourcePath, entry_point: &str, level: OptimizationLevel, warning_as_error: bool) -> Result<Self, VulkanError> {
		Self::new_from_source(device, code_path.load()?.as_ref(), &code_path.get_filename(), entry_point, level, warning_as_error)
	}

	/// Get the inner
	pub(crate) fn get_vk_shader(&self) -> VkShaderModule {
		self.shader
	}

	/// Get variables
	pub fn get_vars(&self) -> &[ShaderVariable] {
		&self.vars
	}
}

impl Debug for VulkanShader {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanShader")
		.field("shader", &self.shader)
		.field("vars", &self.vars)
		.finish()
	}
}

impl Drop for VulkanShader {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyShaderModule(self.device.get_vk_device(), self.shader, null()).unwrap();
	}
}
