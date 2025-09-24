
use crate::prelude::*;
use std::{
	any::TypeId,
	collections::HashMap,
	fmt::{self, Debug, Formatter},
	fs::read,
	mem::forget,
	path::Path,
	ptr::null,
	slice::from_raw_parts,
	sync::{Arc, RwLock, OnceLock},
};

/// The optimization level for shaderc
#[cfg(feature = "shaderc")]
pub use shaderc::OptimizationLevel;

pub mod shader_analyzer {
	use super::*;
	use rspirv::{
		dr::Module,
		spirv::*,
	};

	/// The input layout of the variable
	#[derive(Debug, Clone, Copy)]
	pub enum VariableLayout {
		None,
		Descriptor{set: u32, binding: u32, input_attachment_index: Option<u32>},
		Location(u32),
	}

	/// The struct member type
	#[derive(Debug, Clone)]
	pub struct StructMember {
		/// The name of the member
		pub member_name: String,

		/// The type of the member
		pub member_type: VariableType,
	}

	impl StructMember {
		pub fn size_of(&self) -> Result<usize, VulkanError> {
			self.member_type.size_of()
		}
	}

	/// The struct type
	#[derive(Debug, Clone)]
	pub struct StructType {
		/// The name of the struct type
		pub name: String,

		/// The members of the struct type
		pub members: Vec<StructMember>,
	}

	impl StructType {
		/// Get the size of this type
		pub fn size_of(&self) -> Result<usize, VulkanError> {
			let mut ret = 0;
			for m in self.members.iter() {
				ret += m.size_of()?;
			}
			Ok(ret)
		}
	}

	/// The variable type
	#[derive(Debug, Clone)]
	pub struct ArrayType {
		/// The type of the array element
		pub element_type: VariableType,

		/// The length of the array
		pub element_count: usize,
	}

	impl ArrayType {
		/// Get the size of this type
		pub fn size_of(&self) -> Result<usize, VulkanError> {
			Ok(self.element_type.size_of()? * self.element_count)
		}
	}

	/// The variable type
	#[derive(Debug, Clone)]
	pub struct RuntimeArrayType {
		/// The type of the array element
		pub element_type: VariableType,
	}

	impl RuntimeArrayType {
		/// Get the size of this type
		pub fn size_of(&self) -> Result<usize, VulkanError> {
			Ok(0)
		}
	}

	#[derive(Debug, Clone, Copy)]
	pub enum ImageDepth {
		NoDepth = 0,
		HasDepth = 1,
		NotIndicated = 2,
	}

	impl From<u32> for ImageDepth {
		fn from(val: u32) -> Self {
			match val {
				0 => Self::NoDepth,
				1 => Self::HasDepth,
				2 => Self::NotIndicated,
				_ => panic!("Invalid value for `ImageDepth`"),
			}
		}
	}

	#[derive(Debug, Clone, Copy)]
	pub enum ImageSampled {
		RunTimeOnly = 0,
		ReadOnly = 1,
		ReadWrite = 2,
	}

	impl From<u32> for ImageSampled {
		fn from(val: u32) -> Self {
			match val {
				0 => Self::RunTimeOnly,
				1 => Self::ReadOnly,
				2 => Self::ReadWrite,
				_ => panic!("Invalid value for `ImageSampled`"),
			}
		}
	}

	/// The image type
	#[derive(Debug, Clone)]
	pub struct ImageType {
		/// The sampled variable type
		pub result: VariableType,

		/// The dimension of the image
		pub dim: Dim,

		/// The depth of the image
		pub depth: ImageDepth,

		/// Is the image arrayed
		pub arrayed: bool,

		/// Is multisample enabled on this image
		pub multisample: bool,

		/// Is this image could be sampled or written
		pub sampled: ImageSampled,

		/// The format of the image
		pub format: ImageFormat,
	}

	/// The variable type
	#[derive(Debug, Clone)]
	pub enum VariableType {
		/// Literal variable
		Literal(String),

		/// Struct
		Struct(StructType),

		/// Array
		Array(Box<ArrayType>),

		/// RuntimeArray
		RuntimeArray(Box<RuntimeArrayType>),

		/// Image
		Image(Box<ImageType>),

		/// Opaque type
		Opaque(String),
	}

	impl VariableType {
		/// Unwrap for literal variable
		pub fn unwrap_literal(&self) -> &str {
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
		pub fn unwrap_array(&self) -> &ArrayType {
			if let Self::Array(ret) = self {
				ret
			} else {
				panic!("Expected `VariableType::Array`, got {self:?}")
			}
		}

		/// Unwrap for runtime array
		pub fn unwrap_runtime_array(&self) -> &RuntimeArrayType {
			if let Self::RuntimeArray(ret) = self {
				ret
			} else {
				panic!("Expected `VariableType::RuntimeArray`, got {self:?}")
			}
		}

		/// Unwrap for image
		pub fn unwrap_image(&self) -> &ImageType {
			if let Self::Image(ret) = self {
				ret
			} else {
				panic!("Expected `VariableType::Image`, got {self:?}")
			}
		}

		/// Unwrap for opaque variable
		pub fn unwrap_opaque(&self) -> &str {
			if let Self::Opaque(ret) = self {
				ret
			} else {
				panic!("Expected `VariableType::Opaque`, got {self:?}")
			}
		}

		/// Get the size of this type
		pub fn size_of(&self) -> Result<usize, VulkanError> {
			match self {
				VariableType::Literal(literal) => match literal.as_str() {
					"i8" | "u8" => Ok(1),
					"f16" | "i16" | "u16" => Ok(2),
					"f32" | "i32" | "u32" | "bool" => Ok(4),
					"f64" | "i64" | "u64" => Ok(8),
					_ => if let Some(index) = literal.find("vec") {
						let prefix = &literal[0..index];
						let suffix = &literal[index + 3..];
						let suffix_num = suffix.parse::<usize>().map_err(|_|VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))?;
						Ok(match prefix {
							"" | "i" | "u" | "b" => 4,
							"d" => 8,
							_ => Err(VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))?,
						} * suffix_num)
					} else if let Some(index) = literal.find("mat") {
						let prefix = &literal[0..index];
						let suffix = &literal[index + 3..];
						let (num_row, num_col);
						match suffix.len() {
							1 => {
								num_row = suffix.parse::<usize>().map_err(|_|VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))?;
								num_col = num_row;
							}
							3 => {
								num_row = (suffix[..1]).parse::<usize>().map_err(|_|VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))?;
								num_col = (suffix[2..]).parse::<usize>().map_err(|_|VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))?;
							}
							_ => return Err(VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}"))),
						}
						Ok(match prefix {
							"" | "i" | "u" | "b" => 4,
							"d" => 8,
							_ => Err(VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))?,
						} * num_row * num_col)
					} else {
						Err(VulkanError::ShaderParseTypeUnknown(format!("Unknown literal type {literal}")))
					}
				}
				VariableType::Struct(st) => {
					st.size_of()
				}
				VariableType::Array(arr) => {
					arr.size_of()
				}
				VariableType::RuntimeArray(rtarr) => {
					rtarr.size_of()
				}
				_ => Err(VulkanError::ShaderParseTypeUnknown(format!("Unable to evaluate the size of the type {self:?}")))
			}
		}

		/// Get the type name
		pub fn base_type_name(&self) -> String {
			match self {
				VariableType::Literal(literal) => literal.clone(),
				VariableType::Struct(st) => st.name.clone(),
				VariableType::Array(arr) => arr.element_type.base_type_name(),
				VariableType::RuntimeArray(arr) => arr.element_type.base_type_name(),
				VariableType::Image(img) => {
					let dim = match img.dim {
						Dim::Dim1D => "1D".to_string(),
						Dim::Dim2D => "2D".to_string(),
						Dim::Dim3D => "3D".to_string(),
						Dim::DimCube => "Cube".to_string(),
						Dim::DimRect => "Rect".to_string(),
						Dim::DimBuffer => "Buffer".to_string(),
						Dim::DimSubpassData => "subpassInput".to_string(),
						others => format!("{others:?}"),
					};
					let ms = if img.multisample {
						"MS"
					} else {
						""
					};
					let array = if img.arrayed {
						"Array"
					} else {
						""
					};
					format!("sampler{dim}{ms}{array}")
				}
				VariableType::Opaque(opaque) => opaque.clone(),
			}
		}
	}

	/// The input and output of the shader
	#[derive(Debug, Clone)]
	pub struct ShaderVariable {
		/// The type of the variable
		pub var_type: VariableType,

		/// The name of the variable
		pub var_name: String,

		/// The storage class of the variable
		pub storage_class: StorageClass,

		/// The layout of the variable
		pub layout: VariableLayout,
	}

	impl ShaderVariable {
		pub fn size_of(&self) -> Result<usize, VulkanError> {
			self.var_type.size_of()
		}
	}

	#[derive(Clone, Copy)]
	pub union ConstantValue {
		float: f32,
		double: f64,
		int: i32,
		uint: u32,
	}

	impl ConstantValue {
		/// Assign from an `u32`
		pub fn from_bit32(bit32: u32) -> Self {
			Self {
				uint: bit32,
			}
		}

		/// Assign from a `u64` bitvise, the fact is that the data is a `double` variable, but it was provided as a `u64` value.
		pub fn from_bit64(bit64: u64) -> Self {
			Self {
				double: f64::from_bits(bit64),
			}
		}
	}

	impl Debug for ConstantValue {
		fn fmt(&self, f: &mut Formatter) -> fmt::Result {
			unsafe {f.debug_struct("ConstantValue")
			.field("float", &self.float)
			.field("double", &self.double)
			.field("int", &self.int)
			.field("uint", &self.uint)
			.finish()}
		}
	}

	/// The input and output of the shader
	#[derive(Debug, Clone)]
	pub struct Constants {
		/// The type of the constant
		pub var_type: VariableType,

		/// The name of the constant
		pub var_name: String,

		/// The value
		pub value: ConstantValue,
	}

	#[derive(Debug, Clone)]
	pub struct ShaderAnalyzer {
		/// The analyzed shader module info and tokens and the IL instructions
		module: Module,

		/// The constant values
		constants: HashMap<Word, Constants>,
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
			let mut ret = Self {
				module,
				constants: HashMap::new(),
			};
			for inst in ret.module.global_inst_iter() {
				if inst.class.opcode == Op::Constant {
					let id = inst.result_id.unwrap();
					let var_type = ret.get_type(inst.result_type.unwrap())?;
					let var_name = ret.get_name(id);
					let value = match var_type.unwrap_literal() {
						"f32" | "i32" | "u32" => ConstantValue::from_bit32(inst.operands[0].unwrap_literal_bit32()),
						"f64" => ConstantValue::from_bit64(inst.operands[0].unwrap_literal_bit64()),
						others => return Err(VulkanError::ShaderParseTypeUnknown(format!("Unknown type of constant {var_name:?}: {others}"))),
					};
					ret.constants.insert(id, Constants {
						var_type,
						var_name,
						value,
					});
				}
			}
			Ok(ret)
		}

		/// Get the string of a target_id
		pub fn get_name(&self, target_id: Word) -> String {
			for inst in self.module.debug_names.iter() {
				if inst.class.opcode == Op::Name && inst.operands[0].unwrap_id_ref() == target_id {
					let ret = inst.operands[1].unwrap_literal_string().to_string();
					if ret.is_empty() {
						break;
					} else {
						return ret;
					}
				}
			}
			format!("id_{target_id}")
		}

		/// Get the constant value
		pub fn get_constant(&self, target_id: Word) -> Option<&Constants> {
			self.constants.get(&target_id)
		}

		/// Get the string of a target_id
		pub fn get_member_name(&self, target_id: Word, member_id: u32) -> String {
			for inst in self.module.debug_names.iter() {
				if inst.class.opcode == Op::MemberName {
					if inst.operands[0].unwrap_id_ref() != target_id || inst.operands[1].unwrap_literal_bit32() != member_id {
						continue;
					}
					let ret = inst.operands[2].unwrap_literal_string().to_string();
					if ret.is_empty() {
						break;
					} else {
						return ret;
					}
				}
			}
			format!("member_{member_id}")
		}

		/// Get the layout of the variable
		/// * `member_id`: To retrieve the layout of a struct member, this field should be `Some`
		/// * If you want to retrieve the layout of a variable rather than a struct member, this field should be `None`
		pub fn get_layout(&self, target_id: Word, member_id: Option<u32>) -> VariableLayout {
			let mut set = None;
			let mut binding = None;
			let mut input_attachment_index = None;
			for inst in self.module.annotations.iter() {
				if inst.class.opcode != Op::Decorate {
					continue;
				}
				if inst.operands[0].unwrap_id_ref() != target_id {
					continue;
				}

				let decoration;
				let value = if let Some(member_id) = member_id {
					if member_id != inst.operands[1].unwrap_literal_bit32() {
						continue;
					}
					decoration = inst.operands[2].unwrap_decoration();
					&inst.operands[3]
				} else {
					decoration = inst.operands[1].unwrap_decoration();
					&inst.operands[2]
				};
				match decoration {
					Decoration::Location => {
						return VariableLayout::Location(value.unwrap_literal_bit32());
					}
					Decoration::DescriptorSet => {
						set = Some(value.unwrap_literal_bit32());
					}
					Decoration::Binding => {
						binding = Some(value.unwrap_literal_bit32());
					}
					Decoration::InputAttachmentIndex => {
						input_attachment_index = Some(value.unwrap_literal_bit32());
					}
					_ => continue,
				}
				if let Some(set) = set && let Some(binding) = binding && input_attachment_index.is_some() {
					return VariableLayout::Descriptor{set, binding, input_attachment_index};
				}
			}
			if set.is_some() || binding.is_some() {
				VariableLayout::Descriptor{set: set.unwrap_or(0), binding: binding.unwrap_or(0), input_attachment_index}
			} else {
				VariableLayout::None
			}
		}

		/// Get the string type
		pub fn get_type(&self, type_id: u32) -> Result<VariableType, VulkanError> {
			let mut inst = None;
			for i in self.module.types_global_values.iter() {
				if i.result_id.unwrap() == type_id {
					inst = Some(i);
					break;
				}
			}
			if let Some(inst) = inst {
				match inst.class.opcode {
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
						match component_type.unwrap_literal() {
							"f32"  => Ok(VariableType::Literal(format!( "vec{component_count}"))),
							"f64"  => Ok(VariableType::Literal(format!("dvec{component_count}"))),
							"i32"  => Ok(VariableType::Literal(format!("ivec{component_count}"))),
							"u32"  => Ok(VariableType::Literal(format!("uvec{component_count}"))),
							"bool" => Ok(VariableType::Literal(format!("bvec{component_count}"))),
							others => Err(VulkanError::ShaderParseTypeUnknown(others.to_string())),
						}
					}
					Op::TypeMatrix => {
						let column_type_id = inst.operands[0].unwrap_id_ref();
						let column_count = inst.operands[1].unwrap_literal_bit32();
						let column_type = self.get_type(column_type_id)?;
						let column_type_name = column_type.unwrap_literal();
						let column_dim = column_type_name.chars().last().unwrap().to_digit(10).unwrap();
						let mut prefix = column_type_name.split("vec").next().unwrap_or("");
						if prefix == "v" {prefix = ""}
						match &column_type_name[prefix.len()..column_type_name.len() - 1] {
							"vec" => match (column_dim, column_count) {
								(1, 1) | (2, 2) | (3, 3) | (4, 4) => Ok(VariableType::Literal(format!("{prefix}mat{column_dim}"))),
								_ => Ok(VariableType::Literal(format!("{prefix}mat{column_count}x{column_dim}"))),
							}
							_ => Err(VulkanError::ShaderParseTypeUnknown(format!("{inst:?}"))),
						}
					}
					Op::TypeStruct => {
						let name = self.get_name(type_id);
						let mut members: Vec<StructMember> = Vec::with_capacity(inst.operands.len());
						for (i, member) in inst.operands.iter().enumerate() {
							let id = member.unwrap_id_ref();
							let member_name = self.get_member_name(type_id, i as u32);
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
						let element_count = unsafe {self.get_constant(inst.operands[1].unwrap_id_ref()).unwrap().value.uint} as usize;
						Ok(VariableType::Array(Box::new(ArrayType {
							element_type,
							element_count,
						})))
					}
					Op::TypeRuntimeArray => {
						let element_type = self.get_type(inst.operands[0].unwrap_id_ref())?;
						Ok(VariableType::RuntimeArray(Box::new(RuntimeArrayType {
							element_type,
						})))
					}
					Op::TypeSampledImage => {
						self.get_type(inst.operands[0].unwrap_id_ref())
					}
					Op::TypeImage => {
						Ok(VariableType::Image(Box::new(ImageType {
							result: self.get_type(inst.operands[0].unwrap_id_ref())?,
							dim: inst.operands[1].unwrap_dim(),
							depth: inst.operands[2].unwrap_literal_bit32().into(),
							arrayed: inst.operands[3].unwrap_literal_bit32() != 0,
							multisample: inst.operands[4].unwrap_literal_bit32() != 0,
							sampled: inst.operands[5].unwrap_literal_bit32().into(),
							format: inst.operands[6].unwrap_image_format(),
						})))
					}
					Op::TypeSampler => {
						Ok(VariableType::Literal(String::from("sampler")))
					}
					Op::TypeOpaque => {
						let name = if let Some(first_operand) = inst.operands.first() {
							first_operand.unwrap_literal_string().to_string()
						} else {
							format!("type_{type_id}")
						};
						Ok(VariableType::Opaque(name))
					}
					_ => {
						println!("{:#?}", self.module);
						Err(VulkanError::ShaderParseTypeUnknown(format!("{inst:?}")))
					}
				}
			} else {
				Err(VulkanError::ShaderParseIdUnknown(format!("Unknown type ID: {type_id}")))
			}
		}

		/// Get the global variables that may contain the uniform inputs, attribute inputs, and outputs of the shader.
		pub fn get_global_vars(&self) -> Result<Vec<Arc<ShaderVariable>>, VulkanError> {
			let mut vars: Vec<Arc<ShaderVariable>> = Vec::with_capacity(self.module.types_global_values.len());
			for inst in self.module.global_inst_iter() {
				if inst.class.opcode != Op::Variable {
					continue;
				}

				let var_id = inst.result_id.unwrap();
				let var_type_id = inst.result_type.unwrap();
				let storage_class = inst.operands[0].unwrap_storage_class();

				let var_type = self.get_type(var_type_id).unwrap();
				let var_name = self.get_name(var_id);
				let layout = self.get_layout(var_id, None);

				vars.push(Arc::new(ShaderVariable {
					var_type,
					var_name,
					storage_class,
					layout,
				}));
			}
			Ok(vars)
		}
	}
}

use shader_analyzer::*;

/// The texture image and sampler to put in a pipeline (used to input into a shader)
#[derive(Debug)]
pub struct TextureForSample {
	/// The texture
	pub texture: Arc<RwLock<VulkanTexture>>,

	/// The sampler
	pub sampler: Arc<VulkanSampler>,
}

/// The properties for the descriptor set
#[derive(Debug)]
pub enum DescriptorProp {
	/// The props for the samplers
	Samplers(Vec<Arc<VulkanSampler>>),

	/// The props for the image
	Images(Vec<TextureForSample>),

	/// The props for the uniform buffers
	UniformBuffers(Vec<RwLock<Box<dyn GenericUniformBuffer>>>),
}

impl DescriptorProp {
	/// Get samplers
	pub fn get_samplers(&self) -> Result<&[Arc<VulkanSampler>], VulkanError> {
		if let Self::Samplers(samplers) = self {
			Ok(samplers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::Samplers`, got {self:?}")))
		}
	}

	/// Get images
	pub fn get_images(&self) -> Result<&[TextureForSample], VulkanError> {
		if let Self::Images(images) = self {
			Ok(images)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::Images`, got {self:?}")))
		}
	}

	/// Get uniform buffers
	pub fn get_uniform_buffers(&self) -> Result<&[RwLock<Box<dyn GenericUniformBuffer>>], VulkanError> {
		if let Self::UniformBuffers(uniform_buffers) = self {
			Ok(uniform_buffers)
		} else {
			Err(VulkanError::ShaderInputTypeMismatch(format!("Expected `DescriptorProp::UniformBuffers`, got {self:?}")))
		}
	}

	/// Unwrap for samplers
	pub fn unwrap_samplers(&self) -> &[Arc<VulkanSampler>] {
		if let Self::Samplers(samplers) = self {
			samplers
		} else {
			panic!("Expected `DescriptorProp::Samplers`, got {self:?}")
		}
	}

	/// Unwrap for images
	pub fn unwrap_images(&self) -> &[TextureForSample] {
		if let Self::Images(images) = self {
			images
		} else {
			panic!("Expected `DescriptorProp::Images`, got {self:?}")
		}
	}

	/// Unwrap for uniform buffers
	pub fn unwrap_uniform_buffers(&self) -> &[RwLock<Box<dyn GenericUniformBuffer>>] {
		if let Self::UniformBuffers(uniform_buffers) = self {
			uniform_buffers
		} else {
			panic!("Expected `DescriptorProp::UniformBuffers`, got {self:?}")
		}
	}
}

/// The wrapper for `VkShaderModule`
pub struct VulkanShader {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the shader
	shader: VkShaderModule,

	/// The parsed variables of the shader
	vars: Vec<Arc<ShaderVariable>>,

	/// The properties of the shader descriptor sets
	desc_props: HashMap<String, DescriptorProp>
}

/// The shader source
#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
	/// Vertex shader source code
	VertexShader(&'a str),

	/// Tessellation control shader source code
	TessellationControlShader(&'a str),

	/// Tessellation evaluation shader source code
	TessellationEvaluationShader(&'a str),

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

	/// Tessellation control shader source code
	TessellationControlShader(String),

	/// Tessellation evaluation shader source code
	TessellationEvaluationShader(String),

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

	/// Tessellation control shader file path
	TessellationControlShader(&'a Path),

	/// Tessellation evaluation shader file path
	TessellationEvaluationShader(&'a Path),

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
			Self::TessellationControlShader(path) => {let bytes = read(path)?; ShaderSourceOwned::TessellationControlShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::TessellationEvaluationShader(path) => {let bytes = read(path)?; ShaderSourceOwned::TessellationEvaluationShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::GeometryShader(path) => {let bytes = read(path)?; ShaderSourceOwned::GeometryShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::FragmentShader(path) => {let bytes = read(path)?; ShaderSourceOwned::FragmentShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
			Self::ComputeShader(path) => {let bytes = read(path)?; ShaderSourceOwned::ComputeShader(unsafe {str::from_utf8_unchecked(&bytes).to_owned()})}
		})
	}

	pub fn get_filename(&self) -> String {
		match self {
			Self::VertexShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
			Self::TessellationControlShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
			Self::TessellationEvaluationShader(path) => path.file_name().unwrap().to_string_lossy().to_string(),
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
			Self::TessellationControlShader(string) => ShaderSource::TessellationControlShader(string),
			Self::TessellationEvaluationShader(string) => ShaderSource::TessellationEvaluationShader(string),
			Self::GeometryShader(string) => ShaderSource::GeometryShader(string),
			Self::FragmentShader(string) => ShaderSource::FragmentShader(string),
			Self::ComputeShader(string) => ShaderSource::ComputeShader(string),
		}
	}
}

impl VulkanShader {
	/// Create the `VulkanShader` from the shader code, it should be aligned to 32-bits
	pub fn new(device: Arc<VulkanDevice>, shader_code: &[u32], desc_props: HashMap<String, DescriptorProp>) -> Result<Self, VulkanError> {
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
			desc_props,
		})
	}

	/// Create the `VulkanShader` from file
	pub fn new_from_file(device: Arc<VulkanDevice>, shader_file: &Path, desc_props: HashMap<String, DescriptorProp>) -> Result<Self, VulkanError> {
		let mut shader_bytes = read(shader_file)?;
		shader_bytes.resize(((shader_bytes.len() - 1) / 4 + 1) * 4, 0);
		let shader_code = unsafe {
			let ptr = shader_bytes.as_mut_ptr() as *mut u32;
			let len = shader_bytes.len() >> 2;
			let cap = shader_bytes.capacity() >> 2;
			forget(shader_bytes);
			Vec::from_raw_parts(ptr, len, cap)
		};
		Self::new(device, &shader_code, desc_props)
	}

	/// Compile shader code to binary
	#[cfg(feature = "shaderc")]
	pub fn compile(device: Arc<VulkanDevice>, code: ShaderSource, is_hlsl: bool, filename: &str, entry_point: &str, level: OptimizationLevel, warning_as_error: bool) -> Result<Vec<u32>, VulkanError> {
		use shaderc::*;
		use ShaderSource::*;
		let compiler = Compiler::new()?;
		let mut options = CompileOptions::new()?;
		if is_hlsl {options.set_source_language(SourceLanguage::HLSL)}
		options.set_optimization_level(level);
		options.set_generate_debug_info();
		if warning_as_error {options.set_warnings_as_errors()}
		options.set_target_env(TargetEnv::Vulkan, device.vkcore.get_app_info().apiVersion);
		let source;
		let kind = match code {
			VertexShader(ref src) => {source = src; ShaderKind::Vertex}
			TessellationControlShader(ref src) => {source = src; ShaderKind::TessControl}
			TessellationEvaluationShader(ref src) => {source = src; ShaderKind::TessEvaluation}
			GeometryShader(ref src) => {source = src; ShaderKind::Geometry}
			FragmentShader(ref src) => {source = src; ShaderKind::Fragment}
			ComputeShader(ref src) => {source = src; ShaderKind::Compute}
		};
		let artifact = compiler.compile_into_spirv(source, kind, filename, entry_point, Some(&options))?;
		Ok(artifact.as_binary().to_vec())
	}

	/// Create the `VulkanShader` from source code
	/// * `level`: You could use one of these: `OptimizationLevel::Zero`, `OptimizationLevel::Size`, and `OptimizationLevel::Performance`
	#[cfg(feature = "shaderc")]
	pub fn new_from_source(device: Arc<VulkanDevice>, code: ShaderSource, is_hlsl: bool, filename: &str, entry_point: &str, level: OptimizationLevel, warning_as_error: bool, desc_props: HashMap<String, DescriptorProp>) -> Result<Self, VulkanError> {
		let artifact = Self::compile(device.clone(), code, is_hlsl, filename, entry_point, level, warning_as_error)?;
		Self::new(device, &artifact, desc_props)
	}

	/// Create the `VulkanShader` from source code from file
	/// * `level`: You could use one of these: `OptimizationLevel::Zero`, `OptimizationLevel::Size`, and `OptimizationLevel::Performance`
	#[cfg(feature = "shaderc")]
	pub fn new_from_source_file(device: Arc<VulkanDevice>, code_path: ShaderSourcePath, is_hlsl: bool, entry_point: &str, level: OptimizationLevel, warning_as_error: bool, desc_props: HashMap<String, DescriptorProp>) -> Result<Self, VulkanError> {
		Self::new_from_source(device, code_path.load()?.as_ref(), is_hlsl, &code_path.get_filename(), entry_point, level, warning_as_error, desc_props)
	}

	/// Get the inner
	pub(crate) fn get_vk_shader(&self) -> VkShaderModule {
		self.shader
	}

	/// Get variables
	pub fn get_vars(&self) -> &[Arc<ShaderVariable>] {
		&self.vars
	}

	/// Get the desc props
	pub fn get_desc_props(&self) -> &HashMap<String, DescriptorProp> {
		&self.desc_props
	}

	/// Get specific number of samplers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_samplers(&self, name: &str, desired_count: usize) -> Result<&[Arc<VulkanSampler>], VulkanError> {
		let name = name.to_string();
		let samplers = self.desc_props.get(&name).ok_or(VulkanError::MissingShaderInputs(name.clone()))?.get_samplers()?;
		if samplers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} sampler(s) is needed for `{}`, but {} sampler(s) were provided.", name, samplers.len())));
		}
		Ok(samplers)
	}

	/// Get specific number of textures from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_textures(&self, name: &str, desired_count: usize) -> Result<&[TextureForSample], VulkanError> {
		let name = name.to_string();
		let textures = self.desc_props.get(&name).ok_or(VulkanError::MissingShaderInputs(name.clone()))?.get_images()?;
		if textures.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} texture(s) is needed for `{}`, but {} texture(s) were provided.", name, textures.len())));
		}
		Ok(textures)
	}

	/// Get specific number of uniform buffers from a `HashMap<String, DescriptorProp>`
	pub fn get_desc_props_uniform_buffers(&self, name: &str, desired_count: usize) -> Result<&[RwLock<Box<dyn GenericUniformBuffer>>], VulkanError> {
		let name = name.to_string();
		let uniform_buffers = self.desc_props.get(&name).ok_or(VulkanError::MissingShaderInputs(name.clone()))?.get_uniform_buffers()?;
		if uniform_buffers.len() != desired_count {
			return Err(VulkanError::ShaderInputLengthMismatch(format!("{desired_count} uniform buffer(s) is needed for `{}`, but {} uniform buffer(s) were provided.", name, uniform_buffers.len())));
		}
		Ok(uniform_buffers)
	}
}

impl Debug for VulkanShader {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanShader")
		.field("shader", &self.shader)
		.field("vars", &self.vars)
		.field("desc_props", &self.desc_props)
		.finish()
	}
}

impl Drop for VulkanShader {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyShaderModule(self.device.get_vk_device(), self.shader, null()).unwrap();
	}
}

/// The type info of CPU bound variable for the shader inputs
#[derive(Debug, Clone)]
pub struct TypeInfo {
	/// Name of the type
	pub type_name: &'static str,

	/// Format for one row
	pub row_format: VkFormat,

	/// Number of rows
	pub num_rows: u32,

	/// Size of the type
	pub size: usize,
}

impl TypeInfo {
	/// Create the `TypeInfo`
	pub fn new(type_name: &'static str, row_format: VkFormat, num_rows: u32, size: usize) -> Self {
		Self {
			type_name,
			row_format,
			num_rows,
			size,
		}
	}

	/// Get a map use a `TypeId` as the key to retrieve the info of the type
	pub fn get_map_of_type_id_to_info() -> &'static HashMap<TypeId, Self> {
		static RET: OnceLock<HashMap<TypeId, TypeInfo>> = OnceLock::new();
		use VkFormat::*;
		RET.get_or_init(|| {
			[
				(TypeId::of::<i8>(), Self::new("i8", VK_FORMAT_R8_SINT, 1, size_of::<i8>())),
				(TypeId::of::<u8>(), Self::new("u8", VK_FORMAT_R8_UINT, 1, size_of::<u8>())),
				(TypeId::of::<i16>(), Self::new("i16", VK_FORMAT_R16_SINT, 1, size_of::<i16>())),
				(TypeId::of::<u16>(), Self::new("u16", VK_FORMAT_R16_UINT, 1, size_of::<u16>())),
				(TypeId::of::<i32>(), Self::new("i32", VK_FORMAT_R32_SINT, 1, size_of::<i32>())),
				(TypeId::of::<u32>(), Self::new("u32", VK_FORMAT_R32_UINT, 1, size_of::<u32>())),
				(TypeId::of::<i64>(), Self::new("i64", VK_FORMAT_R64_SINT, 1, size_of::<i64>())),
				(TypeId::of::<u64>(), Self::new("u64", VK_FORMAT_R64_UINT, 1, size_of::<u64>())),
				(TypeId::of::<f16>(), Self::new("f16", VK_FORMAT_R16_SFLOAT, 1, size_of::<f16>())),
				(TypeId::of::<f32>(), Self::new("f32", VK_FORMAT_R32_SFLOAT, 1, size_of::<f32>())),
				(TypeId::of::<f64>(), Self::new("f64", VK_FORMAT_R64_SFLOAT, 1, size_of::<f64>())),
				(TypeId::of::<Vec1>(), Self::new("vec1", VK_FORMAT_R32_SFLOAT, 1, size_of::<Vec1>())),
				(TypeId::of::<Vec2>(), Self::new("vec2", VK_FORMAT_R32G32_SFLOAT, 1, size_of::<Vec2>())),
				(TypeId::of::<Vec3>(), Self::new("vec3", VK_FORMAT_R32G32B32_SFLOAT, 1, size_of::<Vec3>())),
				(TypeId::of::<Vec4>(), Self::new("vec4", VK_FORMAT_R32G32B32A32_SFLOAT, 1, size_of::<Vec4>())),
				(TypeId::of::<DVec1>(), Self::new("dvec1", VK_FORMAT_R64_SFLOAT, 1, size_of::<DVec1>())),
				(TypeId::of::<DVec2>(), Self::new("dvec2", VK_FORMAT_R64G64_SFLOAT, 1, size_of::<DVec2>())),
				(TypeId::of::<DVec3>(), Self::new("dvec3", VK_FORMAT_R64G64B64_SFLOAT, 1, size_of::<DVec3>())),
				(TypeId::of::<DVec4>(), Self::new("dvec4", VK_FORMAT_R64G64B64A64_SFLOAT, 1, size_of::<DVec4>())),
				(TypeId::of::<IVec1>(), Self::new("ivec1", VK_FORMAT_R32_SINT, 1, size_of::<IVec1>())),
				(TypeId::of::<IVec2>(), Self::new("ivec2", VK_FORMAT_R32G32_SINT, 1, size_of::<IVec2>())),
				(TypeId::of::<IVec3>(), Self::new("ivec3", VK_FORMAT_R32G32B32_SINT, 1, size_of::<IVec3>())),
				(TypeId::of::<IVec4>(), Self::new("ivec4", VK_FORMAT_R32G32B32A32_SINT, 1, size_of::<IVec4>())),
				(TypeId::of::<UVec1>(), Self::new("uvec1", VK_FORMAT_R32_UINT, 1, size_of::<UVec1>())),
				(TypeId::of::<UVec2>(), Self::new("uvec2", VK_FORMAT_R32G32_UINT, 1, size_of::<UVec2>())),
				(TypeId::of::<UVec3>(), Self::new("uvec3", VK_FORMAT_R32G32B32_UINT, 1, size_of::<UVec3>())),
				(TypeId::of::<UVec4>(), Self::new("uvec4", VK_FORMAT_R32G32B32A32_UINT, 1, size_of::<UVec4>())),
				(TypeId::of::<Mat2>(), Self::new("mat2", VK_FORMAT_R32G32_SFLOAT, 2, size_of::<Mat2>())),
				(TypeId::of::<Mat3>(), Self::new("mat3", VK_FORMAT_R32G32B32_SFLOAT, 3, size_of::<Mat3>())),
				(TypeId::of::<Mat4>(), Self::new("mat4", VK_FORMAT_R32G32B32A32_SFLOAT, 4, size_of::<Mat4>())),
				(TypeId::of::<Mat2x3>(), Self::new("mat2x3", VK_FORMAT_R32G32B32_SFLOAT, 2, size_of::<Mat2x3>())),
				(TypeId::of::<Mat2x4>(), Self::new("mat2x4", VK_FORMAT_R32G32B32A32_SFLOAT, 2, size_of::<Mat2x4>())),
				(TypeId::of::<Mat3x2>(), Self::new("mat3x2", VK_FORMAT_R32G32_SFLOAT, 3, size_of::<Mat3x2>())),
				(TypeId::of::<Mat3x4>(), Self::new("mat3x4", VK_FORMAT_R32G32B32A32_SFLOAT, 3, size_of::<Mat3x4>())),
				(TypeId::of::<Mat4x2>(), Self::new("mat4x2", VK_FORMAT_R32G32_SFLOAT, 4, size_of::<Mat4x2>())),
				(TypeId::of::<Mat4x3>(), Self::new("mat4x3", VK_FORMAT_R32G32B32_SFLOAT, 4, size_of::<Mat4x3>())),
				(TypeId::of::<DMat2>(), Self::new("dmat2", VK_FORMAT_R64G64_SFLOAT, 2, size_of::<DMat2>())),
				(TypeId::of::<DMat3>(), Self::new("dmat3", VK_FORMAT_R64G64B64_SFLOAT, 3, size_of::<DMat3>())),
				(TypeId::of::<DMat4>(), Self::new("dmat4", VK_FORMAT_R64G64B64A64_SFLOAT, 4, size_of::<DMat4>())),
				(TypeId::of::<DMat2x3>(), Self::new("dmat2x3", VK_FORMAT_R64G64B64_SFLOAT, 2, size_of::<DMat2x3>())),
				(TypeId::of::<DMat2x4>(), Self::new("dmat2x4", VK_FORMAT_R64G64B64A64_SFLOAT, 2, size_of::<DMat2x4>())),
				(TypeId::of::<DMat3x2>(), Self::new("dmat3x2", VK_FORMAT_R64G64_SFLOAT, 3, size_of::<DMat3x2>())),
				(TypeId::of::<DMat3x4>(), Self::new("dmat3x4", VK_FORMAT_R64G64B64A64_SFLOAT, 3, size_of::<DMat3x4>())),
				(TypeId::of::<DMat4x2>(), Self::new("dmat4x2", VK_FORMAT_R64G64_SFLOAT, 4, size_of::<DMat4x2>())),
				(TypeId::of::<DMat4x3>(), Self::new("dmat4x3", VK_FORMAT_R64G64B64_SFLOAT, 4, size_of::<DMat4x3>())),
			].into_iter().collect()
		})
	}
}
