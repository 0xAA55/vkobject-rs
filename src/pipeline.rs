
use crate::prelude::*;
use std::{
	collections::{BTreeMap, HashMap},
	fmt::Debug,
	ptr::null,
	sync::{Arc, Mutex},
};
use struct_iterable::Iterable;
use shader_analyzer::*;
use rspirv::spirv::*;

/// The trait that the struct of vertices or instances must implement
pub trait VertexType: Copy + Clone + Sized + Default + Debug + Iterable {}
impl<T> VertexType for T where T: Copy + Clone + Sized + Default + Debug + Iterable {}

#[macro_export]
macro_rules! derive_vertex_type {
	($item: item) => {
		#[derive(Iterable, Default, Debug, Clone, Copy)]
		$item
	};
}

/// The shaders to use
#[derive(Debug, Clone)]
pub struct DrawShaders {
	/// The vertex shader cannot be absent
	vertex_shader: Arc<VulkanShader>,

	/// The optional tessellation control shader
	tessellation_control_shader: Option<Arc<VulkanShader>>,

	/// The optional tessellation evaluation shader
	tessellation_evaluation_shader: Option<Arc<VulkanShader>>,

	/// The optional geometry shader
	geometry_shader: Option<Arc<VulkanShader>>,

	/// The fragment shader cannot be absent
	fragment_shader: Arc<VulkanShader>,
}

impl DrawShaders {
	/// Create the `DrawShaders`
	pub fn new(
		vertex_shader: Arc<VulkanShader>,
		tessellation_control_shader: Option<Arc<VulkanShader>>,
		tessellation_evaluation_shader: Option<Arc<VulkanShader>>,
		geometry_shader: Option<Arc<VulkanShader>>,
		fragment_shader: Arc<VulkanShader>) -> Self {
		Self {
			vertex_shader,
			tessellation_control_shader,
			tessellation_evaluation_shader,
			geometry_shader,
			fragment_shader,
		}
	}

	/// Create an iterator that iterates through all of the shaders variables
	pub fn iter_vars(&self) -> impl Iterator<Item = &Arc<ShaderVariable>> {
		self.vertex_shader.get_vars().iter().chain(
		if let Some(geometry_shader) = &self.geometry_shader {
			geometry_shader.get_vars().iter()
		} else {
			[].iter()
		}).chain(
			self.fragment_shader.get_vars().iter()
		)
	}
}

unsafe impl Send for DrawShaders {}
unsafe impl Sync for DrawShaders {}

/// The descriptor set layout object
#[derive(Debug)]
pub struct DescriptorSetLayout {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The Vulkan handle of the descriptor set layout
	pub(crate) descriptor_set_layout: VkDescriptorSetLayout,
}

impl DescriptorSetLayout {
	/// Create the `DescriptorSetLayout` by a `VkDescriptorSetLayoutCreateInfo`
	pub fn new(device: Arc<VulkanDevice>, layout_ci: &VkDescriptorSetLayoutCreateInfo) -> Result<Self, VulkanError> {
		let mut descriptor_set_layout = null();
		device.vkcore.vkCreateDescriptorSetLayout(device.get_vk_device(), layout_ci, null(), &mut descriptor_set_layout)?;
		Ok(Self {
			device,
			descriptor_set_layout,
		})
	}
}

impl Drop for DescriptorSetLayout {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyDescriptorSetLayout(self.device.get_vk_device(), self.descriptor_set_layout, null()).unwrap();
	}
}

/// The descriptor set object
#[derive(Debug)]
pub struct DescriptorSets {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The shader stage
	shader_stage: VkShaderStageFlags,

	/// The pool
	pub desc_pool: Arc<DescriptorPool>,

	/// The descriptor set layouts
	descriptor_set_layouts: BTreeMap<u32, DescriptorSetLayout>,

	/// The descriptor sets
	descriptor_sets: VkDescriptorSet,

	/// The associated shader
	pub shader: Arc<VulkanShader>,
}

impl DescriptorSets {
	/// Create the `DescriptorSetLayout` by parsing the shader variable
	pub fn new(device: Arc<VulkanDevice>, shader: Arc<VulkanShader>, shader_stage: VkShaderStageFlags, desc_pool: Arc<DescriptorPool>) -> Result<Self, VulkanError> {
		let mut layout_bindings: HashMap<u32, HashMap<u32, VkDescriptorSetLayoutBinding>> = HashMap::new();
		let desc_props = shader.get_desc_props();
		for var in shader.get_vars() {
			match var.layout {
				VariableLayout::Descriptor{set, binding} => {
					let set_binding = if let Some(set_binding) = layout_bindings.get_mut(&set) {
						set_binding
					} else {
						layout_bindings.insert(set, HashMap::new());
						layout_bindings.get_mut(&set).unwrap()
					};
					match var.storage_class {
						StorageClass::UniformConstant => {
							let samplers: Vec<VkSampler> = if let Some(props) = desc_props.get(&var.var_name) {
								if let DescriptorProp::Samplers(samplers) = props {
									samplers.iter().map(|s|s.get_vk_sampler()).collect()
								} else {
									Vec::new()
								}
							} else {
								Vec::new()
							};
							match &var.var_type {
								VariableType::Literal(literal_type) => {
									assert_eq!(1, samplers.len());
									if literal_type == "sampler" {
										set_binding.insert(binding, VkDescriptorSetLayoutBinding {
											binding,
											descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLER,
											descriptorCount: 1,
											stageFlags: shader_stage,
											pImmutableSamplers: samplers.as_ptr(),
										});
									}
								}
								VariableType::Image(_) => {
									assert_eq!(1, samplers.len());
									set_binding.insert(binding, VkDescriptorSetLayoutBinding {
										binding,
										descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
										descriptorCount: 1,
										stageFlags: shader_stage,
										pImmutableSamplers: samplers.as_ptr(),
									});
								}
								VariableType::Array(array_info) => {
									assert_eq!(array_info.element_count, samplers.len());
									match &array_info.element_type {
										VariableType::Literal(literal_type) => {
											if literal_type == "sampler" {
												set_binding.insert(binding, VkDescriptorSetLayoutBinding {
													binding,
													descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLER,
													descriptorCount: array_info.element_count as u32,
													stageFlags: shader_stage,
													pImmutableSamplers: samplers.as_ptr(),
												});
											}
										}
										VariableType::Image(_) => {
											set_binding.insert(binding, VkDescriptorSetLayoutBinding {
												binding,
												descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
												descriptorCount: array_info.element_count as u32,
												stageFlags: shader_stage,
												pImmutableSamplers: samplers.as_ptr(),
											});
										}
										_ => eprintln!("[WARN] Unknown array type of uniform constant {}: {:?}", var.var_name, var.var_type),
									}
								}
								others => eprintln!("[WARN] Unknown type of uniform constant {}: {others:?}", var.var_name),
							}
						}
						StorageClass::Uniform => {
							match &var.var_type {
								VariableType::Array(array_info) => {
									set_binding.insert(binding, VkDescriptorSetLayoutBinding {
										binding,
										descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
										descriptorCount: array_info.element_count as u32,
										stageFlags: shader_stage,
										pImmutableSamplers: null(),
									});
								}
								_ => {
									set_binding.insert(binding, VkDescriptorSetLayoutBinding {
										binding,
										descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
										descriptorCount: 1,
										stageFlags: shader_stage,
										pImmutableSamplers: null(),
									});
								}
							}
						}
						_ => {}
					}
				}
				_ => {}
			}
		}
		let mut bindings_of_set: BTreeMap<u32, Vec<VkDescriptorSetLayoutBinding>> = BTreeMap::new();
		for (set_key, set_val) in layout_bindings.iter() {
			for (_, binding_val) in set_val.iter() {
				let array = if let Some(array) = bindings_of_set.get_mut(set_key) {
					array
				} else {
					bindings_of_set.insert(*set_key, Vec::new());
					bindings_of_set.get_mut(set_key).unwrap()
				};
				array.push(*binding_val);
			}
		}
		let mut descriptor_set_layouts: BTreeMap<u32, DescriptorSetLayout> = BTreeMap::new();
		for (key, val) in bindings_of_set.iter() {
			let layout_ci = VkDescriptorSetLayoutCreateInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				pNext: null(),
				flags: 0,
				bindingCount: val.len() as u32,
				pBindings: val.as_ptr(),
			};
			descriptor_set_layouts.insert(*key, DescriptorSetLayout::new(device.clone(), &layout_ci)?);
		}
		let layout_array: Vec<VkDescriptorSetLayout> = descriptor_set_layouts.iter().map(|(_, v)|v.descriptor_set_layout).collect();
		let desc_sets_ai = VkDescriptorSetAllocateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			pNext: null(),
			descriptorPool: desc_pool.get_vk_pool(),
			descriptorSetCount: layout_array.len() as u32,
			pSetLayouts: layout_array.as_ptr(),
		};
		let mut descriptor_sets: VkDescriptorSet = null();
		device.vkcore.vkAllocateDescriptorSets(device.get_vk_device(), &desc_sets_ai, &mut descriptor_sets)?;
		Ok(Self {
			device,
			shader_stage,
			desc_pool,
			descriptor_set_layouts,
			descriptor_sets,
			shader,
		})
	}

	/// Get the `VkDescriptorSetLayout`
	pub(crate) fn get_vk_descriptor_sets(&self) -> VkDescriptorSet {
		self.descriptor_sets
	}

	/// Get the descriptor set layouts
	pub fn get_descriptor_set_layouts(&self) -> &BTreeMap<u32, DescriptorSetLayout> {
		&self.descriptor_set_layouts
	}
}

impl Clone for DescriptorSets {
	fn clone(&self) -> Self {
		Self::new(self.device.clone(), self.shader.clone(), self.shader_stage, self.desc_pool.clone()).unwrap()
	}
}

unsafe impl Send for DescriptorSets {}
unsafe impl Sync for DescriptorSets {}

#[derive(Debug)]
pub struct Pipeline {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The meshes to draw
	pub meshes: Arc<HashMap<String, Arc<Mutex<GenericMeshWithMaterial>>>>,

	/// The shaders to use
	pub shaders: Arc<DrawShaders>,

	/// The pool
	pub desc_pool: Arc<DescriptorPool>,

	/// The pipeline
	pipeline: VkPipeline,
}

impl Pipeline {
	/// Create the `Pipeline`
	pub fn new(device: Arc<VulkanDevice>, meshes: Arc<HashMap<String, Arc<Mutex<GenericMeshWithMaterial>>>>, shaders: Arc<DrawShaders>, desc_pool: Arc<DescriptorPool>) -> Result<Self, VulkanError> {
		Ok(Self {
			device,
			meshes,
			shaders,
			desc_pool,
		})
	}

	/// Get the descriptor set layouts
	pub(crate) fn get_vk_pipeline(&self) -> VkPipeline {
		self.pipeline
	}
}

impl Drop for Pipeline {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyPipeline(self.device.get_vk_device(), self.pipeline, null()).unwrap();
	}
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}
