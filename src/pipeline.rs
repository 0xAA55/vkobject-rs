
use crate::prelude::*;
use std::{
	collections::{BTreeMap, HashMap, HashSet},
	fmt::{self, Debug, Formatter},
	iter,
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
			if let VariableLayout::Descriptor{set, binding, input_attachment_index: _} = var.layout {
				let set_binding = if let Some(set_binding) = layout_bindings.get_mut(&set) {
					set_binding
				} else {
					layout_bindings.insert(set, HashMap::new());
					layout_bindings.get_mut(&set).unwrap()
				};
				match var.storage_class {
					StorageClass::UniformConstant => {
						let samplers: Vec<VkSampler> = if let Some(props) = desc_props.get(&var.var_name) && let DescriptorProp::Samplers(samplers) = props {
							samplers.iter().map(|s|s.get_vk_sampler()).collect()
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
					// Ignore other storage classes
					_ => {}
				}
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
		let layout_array: Vec<VkDescriptorSetLayout> = descriptor_set_layouts.values().map(|v|v.descriptor_set_layout).collect();
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
	pub fn iter_vars(&self) -> impl Iterator<Item = (VkShaderStageFlagBits, &Arc<ShaderVariable>)> {
		self.vertex_shader.get_vars().iter().map(|v| (VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT, v)).chain(
			if let Some(tessellation_control_shader) = &self.tessellation_control_shader {
				tessellation_control_shader.get_vars().iter()
			} else {
				[].iter()
			}.map(|v| (VkShaderStageFlagBits::VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, v))
		).chain(
			if let Some(tessellation_evaluation_shader) = &self.tessellation_evaluation_shader {
				tessellation_evaluation_shader.get_vars().iter()
			} else {
				[].iter()
			}.map(|v| (VkShaderStageFlagBits::VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, v))
		).chain(
			if let Some(geometry_shader) = &self.geometry_shader {
				geometry_shader.get_vars().iter()
			} else {
				[].iter()
			}.map(|v| (VkShaderStageFlagBits::VK_SHADER_STAGE_GEOMETRY_BIT, v))
		).chain(self.fragment_shader.get_vars().iter().map(|v| (VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, v)))
	}

	/// Create an iterator that iterates through all of the shaders
	pub fn iter_shaders(&self) -> impl Iterator<Item = (VkShaderStageFlagBits, &Arc<VulkanShader>)> {
        iter::once((VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT, &self.vertex_shader))
		.chain(self.tessellation_control_shader.as_ref().map(|shader| (VkShaderStageFlagBits::VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, shader)))
		.chain(self.tessellation_evaluation_shader.as_ref().map(|shader| (VkShaderStageFlagBits::VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, shader)))
		.chain(self.geometry_shader.as_ref().map(|shader| (VkShaderStageFlagBits::VK_SHADER_STAGE_GEOMETRY_BIT, shader)))
		.chain(iter::once((VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT, &self.fragment_shader)))
	}
}

unsafe impl Send for DrawShaders {}
unsafe impl Sync for DrawShaders {}

pub struct PipelineBuilder {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The meshes to draw
	pub mesh: Arc<Mutex<GenericMeshWithMaterial>>,

	/// The shaders to use
	pub shaders: Arc<DrawShaders>,

	/// The pool
	pub desc_pool: Arc<DescriptorPool>,

	/// The descriptor sets
	pub descriptor_sets: HashMap<VkShaderStageFlagBits, DescriptorSets>,

	/// The render target props
	pub rt_props: Arc<RenderTargetProps>,

	/// The pipeline cache
	pub pipeline_cache: Arc<VulkanPipelineCache>,

	/// The rasterization state create info
	pub rasterization_state_ci: VkPipelineRasterizationStateCreateInfo,

	/// The MSAA state create info
	pub msaa_state_ci: VkPipelineMultisampleStateCreateInfo,

	/// The depth stencil state create info
	pub depth_stenctil_ci: VkPipelineDepthStencilStateCreateInfo,

	/// The color blend state create info
	pub color_blend_state_ci: VkPipelineColorBlendStateCreateInfo,

	/// The color blend attachment states
	pub color_blend_attachment_states: Vec<VkPipelineColorBlendAttachmentState>,

	/// The dynamic states
	pub dynamic_states: HashSet<VkDynamicState>,

	/// The pipeline layout was created by providing descriptor layout there.
	pipeline_layout: VkPipelineLayout,
}

impl PipelineBuilder {
	/// Create the `PipelineBuilder`
	pub fn new(device: Arc<VulkanDevice>, mesh: Arc<Mutex<GenericMeshWithMaterial>>, shaders: Arc<DrawShaders>, desc_pool: Arc<DescriptorPool>, rt_props: Arc<RenderTargetProps>, pipeline_cache: Arc<VulkanPipelineCache>) -> Result<Self, VulkanError> {
		let mut descriptor_sets: HashMap<VkShaderStageFlagBits, DescriptorSets> = HashMap::new();
		let mut desc_set_layouts: Vec<VkDescriptorSetLayout> = Vec::with_capacity(5);
		let mut push_constant_ranges: Vec<VkPushConstantRange> = Vec::with_capacity(5);
		let mut cur_offset: u32 = 0;
		for (stage, shader) in shaders.iter_shaders() {
			let ds = DescriptorSets::new(device.clone(), shader.clone(), stage as VkShaderStageFlags, desc_pool.clone())?;
			for (_, dsl) in ds.get_descriptor_set_layouts().iter() {
				desc_set_layouts.push(dsl.descriptor_set_layout);
			}
			descriptor_sets.insert(stage, ds);
			for var in shader.get_vars() {
				if let StorageClass::PushConstant = var.storage_class {
					let cur_size = (((var.size_of()? - 1) / 4 + 1) * 4) as u32;
					push_constant_ranges.push(VkPushConstantRange {
						stageFlags: stage as VkShaderStageFlags,
						offset: cur_offset,
						size: cur_size,
					});
					cur_offset += cur_size;
				}
			}
		}
		let rasterization_state_ci = VkPipelineRasterizationStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			depthClampEnable: 0,
			rasterizerDiscardEnable: 0,
			polygonMode: VkPolygonMode::VK_POLYGON_MODE_FILL,
			cullMode: VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT as VkCullModeFlags,
			frontFace: VkFrontFace::VK_FRONT_FACE_COUNTER_CLOCKWISE,
			depthBiasEnable: 0,
			depthBiasConstantFactor: 0.0,
			depthBiasClamp: 0.0,
			depthBiasSlopeFactor: 1.0,
			lineWidth: 1.0,
		};
		let msaa_state_ci = VkPipelineMultisampleStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			rasterizationSamples: VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT,
			sampleShadingEnable: 0,
			minSampleShading: 0.0,
			pSampleMask: null(),
			alphaToCoverageEnable: 0,
			alphaToOneEnable: 0,
		};
		let depth_stenctil_ci = VkPipelineDepthStencilStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			depthTestEnable: 1,
			depthWriteEnable: 1,
			depthCompareOp: VkCompareOp::VK_COMPARE_OP_LESS_OR_EQUAL,
			depthBoundsTestEnable: 0,
			stencilTestEnable: 0,
			front: VkStencilOpState {
				failOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				passOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				depthFailOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
				compareMask: 0xFFFFFFFF,
				writeMask: 0xFFFFFFFF,
				reference: 0,
			},
			back: VkStencilOpState {
				failOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				passOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				depthFailOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
				compareMask: 0xFFFFFFFF,
				writeMask: 0xFFFFFFFF,
				reference: 0,
			},
			minDepthBounds: 0.0,
			maxDepthBounds: 0.0,
		};
		let mut color_blend_attachment_states: Vec<VkPipelineColorBlendAttachmentState> = Vec::with_capacity(rt_props.renderpass.attachments.len());
		for attachment in rt_props.renderpass.attachments.iter() {
			if attachment.is_depth_stencil {
				color_blend_attachment_states.push(VkPipelineColorBlendAttachmentState {
					blendEnable: 0,
					srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
					dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
					colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
					srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
					dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
					alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
					colorWriteMask: VkColorComponentFlagBits::combine(&[
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
					]),
				});
			} else {
				color_blend_attachment_states.push(VkPipelineColorBlendAttachmentState {
					blendEnable: 0,
					srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA,
					dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
					colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
					srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA,
					dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
					alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
					colorWriteMask: VkColorComponentFlagBits::combine(&[
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
					]),
				});
			}
		}
		let color_blend_state_ci = VkPipelineColorBlendStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			logicOpEnable: 0,
			logicOp: VkLogicOp::VK_LOGIC_OP_COPY,
			attachmentCount: 0,
			pAttachments: null(),
			blendConstants: [0_f32; 4_usize],
		};
		let pipeline_layout_ci = VkPipelineLayoutCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			pNext: null(),
			flags: 0,
			setLayoutCount: desc_set_layouts.len() as u32,
			pSetLayouts: desc_set_layouts.as_ptr(),
			pushConstantRangeCount: push_constant_ranges.len() as u32,
			pPushConstantRanges: push_constant_ranges.as_ptr(),
		};
		let dynamic_states = [
			VkDynamicState::VK_DYNAMIC_STATE_VIEWPORT,
			VkDynamicState::VK_DYNAMIC_STATE_SCISSOR,
		].into_iter().collect();
		let mut pipeline_layout = null();
		device.vkcore.vkCreatePipelineLayout(device.get_vk_device(), &pipeline_layout_ci, null(), &mut pipeline_layout)?;
		Ok(Self {
			device,
			mesh,
			shaders,
			desc_pool,
			descriptor_sets,
			rt_props,
			pipeline_cache,
			rasterization_state_ci,
			msaa_state_ci,
			depth_stenctil_ci,
			color_blend_state_ci,
			color_blend_attachment_states,
			dynamic_states,
			pipeline_layout,
		})
	}

	/// Set depth clamp enabled/disabled
	pub fn set_depth_clamp_mode(mut self, enabled: bool) -> Self {
		self.rasterization_state_ci.depthClampEnable = if enabled {1} else {0};
		self
	}

	/// Set depth clamp enabled/disabled
	pub fn set_disable_fragment_stage(mut self, disabled: bool) -> Self {
		self.rasterization_state_ci.rasterizerDiscardEnable = if disabled {1} else {0};
		self
	}

	/// Set polygon mode
	pub fn set_polygon_mode(mut self, mode: VkPolygonMode) -> Self {
		self.rasterization_state_ci.polygonMode = mode;
		self
	}

	/// Set cull mode
	pub fn set_cull_mode(mut self, mode: VkCullModeFlags) -> Self {
		self.rasterization_state_ci.cullMode = mode;
		self
	}

	/// Set front face
	pub fn set_front_face(mut self, front_face: VkFrontFace) -> Self {
		self.rasterization_state_ci.frontFace = front_face;
		self
	}

	/// Set enable depth bias mode
	pub fn enable_depth_bias(mut self, constant: f32, slope_scale: f32, clamp: f32) -> Self {
		self.rasterization_state_ci.depthBiasEnable = 1;
		self.rasterization_state_ci.depthBiasConstantFactor = constant;
		self.rasterization_state_ci.depthBiasClamp = clamp;
		self.rasterization_state_ci.depthBiasSlopeFactor= slope_scale;
		self
	}

	/// Set disable depth bias mode
	pub fn disable_depth_bias(mut self) -> Self {
		self.rasterization_state_ci.depthBiasEnable = 0;
		self
	}

	/// Set line width
	pub fn set_line_width(mut self, line_width: f32) -> Self {
		self.rasterization_state_ci.lineWidth = line_width;
		self
	}

	/// Set MSAA sample cound
	pub fn set_msaa_samples(mut self, msaa_samples: VkSampleCountFlagBits) -> Self {
		self.msaa_state_ci.rasterizationSamples = msaa_samples;
		self
	}

	/// Set use dithering to handle MSAA alpha
	pub fn set_msaa_alpha_to_coverage(mut self, enabled: bool) -> Self {
		self.msaa_state_ci.alphaToCoverageEnable = if enabled {1} else {0};
		self
	}

	/// Set MSAA supersampling state
	pub fn set_msaa_super_sampling(mut self, quality: Option<f32>) -> Self {
		if let Some(quality) = quality {
			self.msaa_state_ci.sampleShadingEnable = 1;
			self.msaa_state_ci.minSampleShading = quality;
		} else {
			self.msaa_state_ci.sampleShadingEnable = 0;
		}
		self
	}

	/// Set depth test state
	pub fn set_depth_test(mut self, enabled: bool) -> Self {
		self.depth_stenctil_ci.depthTestEnable = if enabled {1} else {0};
		self
	}

	/// Set depth write state
	pub fn set_depth_write(mut self, enabled: bool) -> Self {
		self.depth_stenctil_ci.depthWriteEnable = if enabled {1} else {0};
		self
	}

	/// Set depth compare mode
	pub fn set_depth_compare_mode(mut self, mode: VkCompareOp) -> Self {
		self.depth_stenctil_ci.depthCompareOp = mode;
		self
	}

	/// Set depth bound test mode
	pub fn set_depth_bound_test_mode(mut self, bounds: Option<(f32, f32)>) -> Self {
		if let Some((min_bound, max_bound)) = bounds {
			self.depth_stenctil_ci.depthBoundsTestEnable = 1;
			self.depth_stenctil_ci.minDepthBounds = min_bound;
			self.depth_stenctil_ci.maxDepthBounds = max_bound;
		} else {
			self.depth_stenctil_ci.depthBoundsTestEnable = 0;
		}
		self
	}

	/// Set stencil test mode
	pub fn set_stencil_test(mut self, enabled: bool) -> Self {
		self.depth_stenctil_ci.stencilTestEnable = if enabled {1} else {0};
		self
	}

	/// Set stencil mode
	pub fn set_stencil_mode(mut self, front_face: VkStencilOpState, back_face: VkStencilOpState) -> Self {
		self.depth_stenctil_ci.front = front_face;
		self.depth_stenctil_ci.back = back_face;
		self
	}

	/// Return a color blend mode for normal alpha blend
	pub fn normal_blend_mode() -> VkPipelineColorBlendAttachmentState {
		VkPipelineColorBlendAttachmentState {
			blendEnable: 1,
			srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			colorWriteMask: VkColorComponentFlagBits::combine(&[
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
			]),
		}
	}

	/// Return a color blend mode for additive blend
	pub fn additive_blend_mode() -> VkPipelineColorBlendAttachmentState {
		VkPipelineColorBlendAttachmentState {
			blendEnable: 1,
			srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			alphaBlendOp: VkBlendOp::VK_BLEND_OP_MAX,
			colorWriteMask: VkColorComponentFlagBits::combine(&[
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
			]),
		}
	}

	/// Return a color blend mode for no blending
	pub fn disabled_blend_mode() -> VkPipelineColorBlendAttachmentState {
		VkPipelineColorBlendAttachmentState {
			blendEnable: 0,
			srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			colorWriteMask: VkColorComponentFlagBits::combine(&[
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
			]),
		}
	}

	/// Set alpha blend mode
	pub fn set_color_blend_mode(mut self, attachment_index: usize, color_blend_mode: VkPipelineColorBlendAttachmentState) -> Self {
		self.color_blend_attachment_states[attachment_index] = color_blend_mode;
		self
	}

	/// Add a dynamic state
	pub fn add_dynamic_state(mut self, dynamic_state: VkDynamicState) -> Self {
		self.dynamic_states.insert(dynamic_state);
		self
	}

	/// Remove a dynamic state
	pub fn remove_dynamic_state(mut self, dynamic_state: VkDynamicState) -> Self {
		self.dynamic_states.remove(&dynamic_state);
		self
	}

	/// Generate the pipeline
	pub fn build(self) -> Result<Pipeline, VulkanError> {
		Pipeline::new(self)
	}
}

impl Debug for PipelineBuilder {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("PipelineBuilder")
		.field("mesh", &self.mesh)
		.field("shaders", &self.shaders)
		.field("desc_pool", &self.desc_pool)
		.field("descriptor_sets", &self.descriptor_sets)
		.field("rt_props", &self.rt_props)
		.field("pipeline_cache", &self.pipeline_cache)
		.field("pipeline_layout", &self.pipeline_layout)
		.finish()
	}
}

impl Drop for PipelineBuilder {
	fn drop(&mut self) {
		if !self.pipeline_layout.is_null() {
			self.device.vkcore.vkDestroyPipelineLayout(self.device.get_vk_device(), self.pipeline_layout, null()).unwrap();
		}
	}
}

pub struct Pipeline {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The meshes to draw
	pub mesh: Arc<Mutex<GenericMeshWithMaterial>>,

	/// The shaders to use
	pub shaders: Arc<DrawShaders>,

	/// The pool
	pub desc_pool: Arc<DescriptorPool>,

	/// The render target props
	pub rt_props: Arc<RenderTargetProps>,

	/// The pipeline cache
	pub pipeline_cache: Arc<VulkanPipelineCache>,

	/// The pipeline layout was created by providing descriptor layout there.
	pipeline_layout: VkPipelineLayout,

	/// The pipeline
	pipeline: VkPipeline,
}

struct MemberInfo<'a> {
	name: &'a str,
	type_name: &'static str,
	row_format: VkFormat,
	num_rows: u32,
	offset: u32,
	size: usize,
}

impl Pipeline {
	/// Create the `Pipeline`
	pub fn new(mut builder: PipelineBuilder) -> Result<Self, VulkanError> {
		let device = builder.device.clone();
		let mesh = builder.mesh.clone();
		let shaders = builder.shaders.clone();
		let desc_pool = builder.desc_pool.clone();
		let rt_props = builder.rt_props.clone();
		let pipeline_cache = builder.pipeline_cache.clone();
		let pipeline_layout = builder.pipeline_layout;
		builder.pipeline_layout = null();
		let shader_stages: Vec<VkPipelineShaderStageCreateInfo> = shaders.iter_shaders().map(|(stage, shader)| VkPipelineShaderStageCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			stage,
			module: shader.get_vk_shader(),
			pName: "main\0".as_ptr() as *const i8,
			pSpecializationInfo: null(),
		}).collect();
		let type_id_to_info = TypeInfo::get_map_of_type_id_to_info();
		let mut mesh_vertex_inputs: HashMap<String, MemberInfo> = HashMap::new();
		let mut mesh_instance_inputs: HashMap<String, MemberInfo> = HashMap::new();
		let mesh_lock = mesh.lock().unwrap();
		let vertex_stride = mesh_lock.mesh.get_vertex_stride();
		let instance_stride = mesh_lock.mesh.get_instance_stride();
		let topology = mesh_lock.mesh.get_primitive_type();
		let mut cur_vertex_offset = 0;
		for (name, var) in mesh_lock.mesh.iter_vertex_buffer_struct_members() {
			if let Some(info) = type_id_to_info.get(&var.type_id()) {
				mesh_vertex_inputs.insert(name.to_string(), MemberInfo {
					name: &name,
					type_name: info.type_name,
					row_format: info.row_format,
					num_rows: info.num_rows,
					offset: cur_vertex_offset,
					size: info.size,
				});
				cur_vertex_offset += info.size as u32;
			} else {
				panic!("Unknown member {:?} of the vertex struct: `{:?}`", var, var.type_id());
			}
		}
		if let Some(instance_member_iter) = mesh_lock.mesh.iter_instance_buffer_struct_members() {
			let mut cur_instance_offset = 0;
			for (name, var) in instance_member_iter {
				if let Some(info) = type_id_to_info.get(&var.type_id()) {
					mesh_instance_inputs.insert(name.to_string(), MemberInfo {
						name: &name,
						type_name: info.type_name,
						row_format: info.row_format,
						num_rows: info.num_rows,
						offset: cur_instance_offset,
						size: info.size,
					});
					cur_instance_offset += info.size as u32;
				} else {
					panic!("Unknown member {:?} of the instance struct: `{:?}`", var, var.type_id());
				}
			}
		}
		drop(mesh_lock);
		let mut vertex_input_bindings: Vec<VkVertexInputBindingDescription> = Vec::with_capacity(2);
		vertex_input_bindings.push(VkVertexInputBindingDescription {
			binding: 0,
			stride: vertex_stride as u32,
			inputRate: VkVertexInputRate::VK_VERTEX_INPUT_RATE_VERTEX,
		});
		if !mesh_instance_inputs.is_empty() {
			vertex_input_bindings.push(VkVertexInputBindingDescription {
				binding: 1,
				stride: instance_stride as u32,
				inputRate: VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE,
			});
		}
		let mut vertex_attrib_bindings: Vec<VkVertexInputAttributeDescription> = Vec::with_capacity(mesh_vertex_inputs.len() + mesh_instance_inputs.len());
		for var in shaders.vertex_shader.get_vars() {
			if let VariableLayout::Location(location) = var.layout {
				if let Some(member_info) = mesh_vertex_inputs.get(&var.var_name) {
					let row_stride = member_info.size as u32 / member_info.num_rows;
					for row in 0..member_info.num_rows {
						vertex_attrib_bindings.push(VkVertexInputAttributeDescription {
							location: location + row,
							binding: 0,
							format: member_info.row_format,
							offset: member_info.offset + row * row_stride,
						});
					}
				} else if let Some(member_info) = mesh_instance_inputs.get(&var.var_name) {
					let row_stride = member_info.size as u32 / member_info.num_rows;
					for row in 0..member_info.num_rows {
						vertex_attrib_bindings.push(VkVertexInputAttributeDescription {
							location: location + row,
							binding: 1,
							format: member_info.row_format,
							offset: member_info.offset + row * row_stride,
						});
					}
				}
			}
		}
		let vertex_input_state_ci = VkPipelineVertexInputStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			vertexBindingDescriptionCount: vertex_input_bindings.len() as u32,
			pVertexBindingDescriptions: vertex_input_bindings.as_ptr(),
			vertexAttributeDescriptionCount: vertex_attrib_bindings.len() as u32,
			pVertexAttributeDescriptions: vertex_attrib_bindings.as_ptr(),
		};
		let input_assembly_state_ci = VkPipelineInputAssemblyStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			topology,
			primitiveRestartEnable: 0,
		};
		let viewport_state_ci = VkPipelineViewportStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			viewportCount: 1,
			pViewports: null(),
			scissorCount: 1,
			pScissors: null(),
		};
		builder.color_blend_state_ci.attachmentCount = builder.color_blend_attachment_states.len() as u32;
		builder.color_blend_state_ci.pAttachments = builder.color_blend_attachment_states.as_ptr();
		let dynamic_states: Vec<VkDynamicState> = builder.dynamic_states.clone().into_iter().collect();
		let dynamic_state_ci = VkPipelineDynamicStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			dynamicStateCount: dynamic_states.len() as u32,
			pDynamicStates: dynamic_states.as_ptr(),
		};
		let pipeline_ci = VkGraphicsPipelineCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			pNext: null(),
			flags: 0,
			stageCount: shader_stages.len() as u32,
			pStages: shader_stages.as_ptr(),
			pVertexInputState: &vertex_input_state_ci,
			pInputAssemblyState: &input_assembly_state_ci,
			pTessellationState: null(), // Currently not supported
			pViewportState: &viewport_state_ci,
			pRasterizationState: &builder.rasterization_state_ci,
			pMultisampleState: &builder.msaa_state_ci,
			pDepthStencilState: &builder.depth_stenctil_ci,
			pColorBlendState: &builder.color_blend_state_ci,
			pDynamicState: &dynamic_state_ci,
			layout: pipeline_layout,
			renderPass: rt_props.renderpass.get_vk_renderpass(),
			subpass: 0,
			basePipelineHandle: null(),
			basePipelineIndex: 0,
		};
		let mut pipeline = null();
		device.vkcore.vkCreateGraphicsPipelines(device.get_vk_device(), pipeline_cache.get_vk_pipeline_cache(), 1, &pipeline_ci, null(), &mut pipeline)?;
		Ok(Self {
			device,
			mesh,
			shaders,
			desc_pool,
			rt_props,
			pipeline_cache,
			pipeline_layout,
			pipeline,
		})
	}

	/// Get the descriptor set layouts
	pub(crate) fn get_vk_pipeline(&self) -> VkPipeline {
		self.pipeline
	}
}

impl Debug for Pipeline {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("Pipeline")
		.field("mesh", &self.mesh)
		.field("shaders", &self.shaders)
		.field("desc_pool", &self.desc_pool)
		.field("rt_props", &self.rt_props)
		.field("pipeline_cache", &self.pipeline_cache)
		.field("pipeline_layout", &self.pipeline_layout)
		.field("pipeline", &self.pipeline)
		.finish()
	}
}

impl Drop for Pipeline {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyPipelineLayout(self.device.get_vk_device(), self.pipeline_layout, null()).unwrap();
		self.device.vkcore.vkDestroyPipeline(self.device.get_vk_device(), self.pipeline, null()).unwrap();
	}
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}
