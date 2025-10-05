
use std::{
	collections::{BTreeMap, BTreeSet, HashMap},
	fmt::Debug,
	fs::File,
	io::{self, BufRead, BufReader, ErrorKind},
	path::Path,
	str::FromStr,
	sync::{Arc, OnceLock, RwLock},
};
extern crate nalgebra_glm as glm;
use glm::*;

/// The OBJ error
#[derive(Debug, Clone)]
pub enum ObjError {
	ParseError{line: usize, what: String},
	IOError{kind: ErrorKind, what: String},
}

impl From<io::Error> for ObjError {
	fn from(e: io::Error) -> Self {
		Self::IOError {
			kind: e.kind(),
			what: format!("{e:?}"),
		}
	}
}

/// The index of the OBJ file faces
#[derive(Default, Debug, Clone, Copy)]
pub struct ObjVTNIndex {
	/// Vertex, 0 is illegal
	pub v: u32,

	/// Texcoord, 0 means no texcoord
	pub vt: u32,

	/// Normal, 0 means no normal
	pub vn: u32,
}

impl ObjVTNIndex {
	/// Parse the string into a VTN index, the string should be like `1/2/3`
	pub fn parse(line_number: usize, s: &str) -> Result<Self, ObjError> {
		let mut parts: Vec<&str> = s.split('/').collect();
		while parts.len() < 3 {
			parts.push("");
		}
		if parts.len() > 3 {
			return Err(ObjError::ParseError {
				line: line_number,
				what: format!("Unknown line which could be splitted into {} parts: `{s}`", parts.len()),
			});
		}
		let v = parts[0].parse::<u32>().ok().ok_or(ObjError::ParseError {
			line: line_number,
			what: format!("Parse vertex index failed from the data `{s}`"),
		})?;
		let vt = if parts[1].is_empty() {
			0
		} else {
			parts[1].parse::<u32>().ok().ok_or(ObjError::ParseError {
				line: line_number,
				what: format!("Parse texcoord index failed from the data `{s}`"),
			})?
		};
		let vn = if parts[2].is_empty() {
			0
		} else {
			parts[2].parse::<u32>().ok().ok_or(ObjError::ParseError {
				line: line_number,
				what: format!("Parse normal index failed from the data `{s}`"),
			})?
		};
		Ok(Self {
			v,
			vt,
			vn,
		})
	}
}

/// The smooth subset of the OBJ file
#[derive(Default, Debug, Clone)]
pub struct ObjSmoothGroup {
	/// The lines of the group
	pub lines: Vec<Vec<u32>>,

	/// The triangle faces of the group. If the original face is a quad, the quad is broken into 2 triangles
	pub triangles: Vec<(ObjVTNIndex, ObjVTNIndex, ObjVTNIndex)>,
}

/// The material subset of the OBJ file
#[derive(Default, Debug, Clone)]
pub struct ObjMaterialGroup {
	/// The optional smooth group of the geometry in bitfields. If the group doesn't provide, its value is 0 means no smoothing.
	pub smooth_group: BTreeMap<i32, ObjSmoothGroup>,
}

/// The group subset of the OBJ file
#[derive(Default, Debug, Clone)]
pub struct ObjGroups {
	/// The material groups in the object
	pub material_groups: BTreeMap<String, ObjMaterialGroup>,
}

/// The object subset of the OBJ file
#[derive(Default, Debug, Clone)]
pub struct ObjObjects {
	/// The groups in the object
	pub groups: BTreeMap<String, ObjGroups>,
}

/// The material component
#[derive(Debug, Clone)]
pub enum ObjMaterialComponent {
	Texture(PathBuf),
	Color(Vec4),
	Luminance(f32),
}

/// The legacy illumination model material
#[derive(Debug, Clone)]
pub struct ObjMaterialLegacy {
	/// Base brightness
	pub ambient: ObjMaterialComponent,

	/// Base color
	pub diffuse: ObjMaterialComponent,

	/// Specular color
	pub specular: ObjMaterialComponent,

	/// Specular power
	pub specular_power: ObjMaterialComponent,

	/// Normal map
	pub normal: ObjMaterialComponent,

	/// Emissive, self-lighting
	pub emissive: ObjMaterialComponent,

	/// The other type of components
	pub others: HashMap<String, ObjMaterialComponent>,
}

impl Default for ObjMaterialLegacy {
	fn default() -> Self {
		Self {
			ambient: ObjMaterialComponent::default(),
			diffuse: ObjMaterialComponent::default(),
			specular: ObjMaterialComponent::default(),
			specular_power: ObjMaterialComponent::Luminance(1.0),
			normal: ObjMaterialComponent::default(),
			emissive: ObjMaterialComponent::default(),
			others: HashMap::new(),
		}
	}
}

/// The physically based rendering illumination model material
#[derive(Default, Debug, Clone)]
pub struct ObjMaterialPbr {
	/// Base color
	pub albedo: ObjMaterialComponent,

	/// Normal map
	pub normal: ObjMaterialComponent,

	/// Ambient occlusion
	pub ao: ObjMaterialComponent,

	/// A.k.a. Height map. The renderer must render this map by extruding the mesh, or use ray-marching to cast the protrusions of the map
	pub displacement: ObjMaterialComponent,

	/// Roughness, something sort of the legacy specular-key map
	pub roughness: ObjMaterialComponent,

	/// Metalness, something sort of the legacy specular map
	pub metalness: ObjMaterialComponent,

	/// Emissive, self-lighting
	pub emissive: ObjMaterialComponent,

	/// The other type of components
	pub others: HashMap<String, ObjMaterialComponent>,
}

impl Default for ObjMaterialComponent {
	fn default() -> Self {
		Self::Color(Vec4::new(0.5, 0.5, 0.5, 1.0))
	}
}

/// The `Material` trait helps the `MaterialLegacy` struct or the `MaterialPbr` struct to be able to turn into an object
pub trait ObjMaterial: Debug {
	/// Get the ambient color
	fn get_ambient(&self) -> Option<&ObjMaterialComponent>;

	/// Get the diffuse color
	fn get_diffuse(&self) -> Option<&ObjMaterialComponent>;

	/// Get the specular color
	fn get_specular(&self) -> Option<&ObjMaterialComponent>;

	/// Get the specular power
	fn get_specular_power(&self) -> Option<&ObjMaterialComponent>;

	/// Get the base color (PBR)
	fn get_albedo(&self) -> Option<&ObjMaterialComponent>;

	/// Get the ambient occlusion (PBR)
	fn get_ao(&self) -> Option<&ObjMaterialComponent>;

	/// Get the displacement map (A.k.a. height map) (PBR)
	fn get_displacement(&self) -> Option<&ObjMaterialComponent>;

	/// Get the roughness map (PBR)
	fn get_roughness(&self) -> Option<&ObjMaterialComponent>;

	/// Get the metalness map (PBR)
	fn get_metalness(&self) -> Option<&ObjMaterialComponent>;

	/// Get the normal map
	fn get_normal(&self) -> Option<&ObjMaterialComponent>;

	/// Get the emissive color
	fn get_emissive(&self) -> Option<&ObjMaterialComponent>;

	/// Get all of the component names exists in this material
	fn get_names(&self) -> BTreeSet<String>;

	/// Get a component by the name of the component
	fn get_by_name(&self, name: &str) -> Option<&ObjMaterialComponent>;

	/// Set a componnet by the name of the component
	fn set_by_name(&mut self, name: &str, texture: ObjMaterialComponent);
}

impl ObjMaterial for ObjMaterialLegacy {
	fn get_ambient(&self) ->		Option<&ObjMaterialComponent> {Some(&self.ambient)}
	fn get_diffuse(&self) ->		Option<&ObjMaterialComponent> {Some(&self.diffuse)}
	fn get_specular(&self) ->		Option<&ObjMaterialComponent> {Some(&self.specular)}
	fn get_specular_power(&self) ->	Option<&ObjMaterialComponent> {Some(&self.specular_power)}
	fn get_normal(&self) ->			Option<&ObjMaterialComponent> {Some(&self.normal)}
	fn get_emissive(&self) ->		Option<&ObjMaterialComponent> {Some(&self.emissive)}

	fn get_albedo(&self) ->			Option<&ObjMaterialComponent> {None}
	fn get_ao(&self) ->				Option<&ObjMaterialComponent> {None}
	fn get_displacement(&self) ->	Option<&ObjMaterialComponent> {None}
	fn get_roughness(&self) ->		Option<&ObjMaterialComponent> {None}
	fn get_metalness(&self) ->		Option<&ObjMaterialComponent> {None}

	fn get_names(&self) -> BTreeSet<String> {
		let mut ret = BTreeSet::new();
		ret.insert("ambient".to_owned());
		ret.insert("diffuse".to_owned());
		ret.insert("specular".to_owned());
		ret.insert("specular_power".to_owned());
		ret.insert("normal".to_owned());
		ret.insert("emissive".to_owned());
		for (name, _) in self.others.iter() {
			ret.insert(name.clone());
		}
		ret
	}

	fn get_by_name(&self, name: &str) -> Option<&ObjMaterialComponent> {
		match self.others.get(name) {
			Some(data) => Some(data),
			None => {
				match name {
					"ambient" =>		self.get_ambient(),
					"diffuse" =>		self.get_diffuse(),
					"specular" =>		self.get_specular(),
					"specular_power" =>	self.get_specular_power(),
					"normal" =>			self.get_normal(),
					"emissive" =>		self.get_emissive(),
					_ => None,
				}
			}
		}
	}

	fn set_by_name(&mut self, name: &str, texture: ObjMaterialComponent) {
		match name {
			"ambient" =>		self.ambient = texture,
			"diffuse" =>		self.diffuse = texture,
			"specular" =>		self.specular = texture,
			"specular_power" =>	self.specular_power = texture,
			"normal" =>			self.normal = texture,
			"emissive" =>		self.emissive = texture,
			others =>{
				self.others.insert(others.to_owned(), texture);
			}
		}
	}
}

impl ObjMaterial for ObjMaterialPbr {
	fn get_albedo(&self) ->			Option<&ObjMaterialComponent> {Some(&self.albedo)}
	fn get_ao(&self) ->				Option<&ObjMaterialComponent> {Some(&self.ao)}
	fn get_displacement(&self) ->	Option<&ObjMaterialComponent> {Some(&self.displacement)}
	fn get_roughness(&self) ->		Option<&ObjMaterialComponent> {Some(&self.roughness)}
	fn get_metalness(&self) ->		Option<&ObjMaterialComponent> {Some(&self.metalness)}
	fn get_normal(&self) ->			Option<&ObjMaterialComponent> {Some(&self.normal)}
	fn get_emissive(&self) ->		Option<&ObjMaterialComponent> {Some(&self.emissive)}

	fn get_ambient(&self) ->		Option<&ObjMaterialComponent> {None}
	fn get_diffuse(&self) ->		Option<&ObjMaterialComponent> {None}
	fn get_specular(&self) ->		Option<&ObjMaterialComponent> {None}
	fn get_specular_power(&self) ->	Option<&ObjMaterialComponent> {None}

	fn get_names(&self) -> BTreeSet<String> {
		let mut ret = BTreeSet::new();
		ret.insert("albedo".to_owned());
		ret.insert("ao".to_owned());
		ret.insert("displacement".to_owned());
		ret.insert("roughness".to_owned());
		ret.insert("metalness".to_owned());
		ret.insert("normal".to_owned());
		ret.insert("emissive".to_owned());
		for (name, _) in self.others.iter() {
			ret.insert(name.clone());
		}
		ret
	}

	fn get_by_name(&self, name: &str) -> Option<&ObjMaterialComponent> {
		match self.others.get(name) {
			Some(data) => Some(data),
			None => {
				match name {
					"albedo" =>			self.get_albedo(),
					"ao" =>				self.get_ao(),
					"displacement" =>	self.get_displacement(),
					"roughness" =>		self.get_roughness(),
					"metalness" =>		self.get_metalness(),
					"normal" =>			self.get_normal(),
					"emissive" =>		self.get_emissive(),
					_ => None,
				}
			}
		}
	}

	fn set_by_name(&mut self, name: &str, texture: ObjMaterialComponent) {
		match name {
			"albedo" =>			self.albedo = texture,
			"ao" =>				self.ao = texture,
			"displacement" =>	self.displacement = texture,
			"roughness" =>		self.roughness = texture,
			"metalness" =>		self.metalness = texture,
			"normal" =>			self.normal = texture,
			"emissive" =>		self.emissive = texture,
			others =>{
				self.others.insert(others.to_owned(), texture);
			}
		}
	}
}

/// The raw objmesh that's just the intepreted result of a obj file.
#[derive(Default, Debug, Clone)]
pub struct ObjMesh<F>
where
	F: Clone + Copy + Sized {
	/// All of the vertices, the `v x y z` lines.
	pub vertices: Vec<TVec3<F>>,

	/// All of the normals, the `vn x y z` lines.
	pub normals: Vec<TVec3<F>>,

	/// All of the texture coords, the `vt x y [z]` lines, with the optional `z` component (default to 0.0)
	pub texcoords: Vec<TVec3<F>>,

	/// The objects of the OBJ file
	pub objects: BTreeMap<String, ObjObjects>,

	/// The materials of the OBJ file
	pub materials: BTreeMap<String, Arc<RwLock<dyn ObjMaterial>>>,
}

/// Trim line and remove comments
fn concentrate_line(line: &str) -> &str {
	let line = line.trim_start();
	let line = if let Some(pos) = line.find('#') {
		&line[0..pos]
	} else {
		line
	};
	line.trim_end()
}

impl<F> ObjMesh<F>
where
	F: Clone + Copy + Sized + FromStr {
	/// Parse an OBJ file.
	pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ObjError> {
		let reader = BufReader::new(File::open(path)?);
		let mut vertices: Vec<TVec3<F>> = Vec::new();
		let mut normals: Vec<TVec3<F>> = Vec::new();
		let mut texcoords: Vec<TVec3<F>> = Vec::new();
		let mut objects: BTreeMap<String, ObjObjects> = BTreeMap::new();
		let mut materials: BTreeMap<String, Arc<RwLock<dyn ObjMaterial>>> = BTreeMap::new();
		let mut mat_loader = None;
		let mut object_name = String::from("");
		let mut group_name = String::from("");
		let mut material_name = String::from("");
		let mut smoothing_group = 0;
		for (line_number, line) in reader.lines().enumerate() {
			let line = line?;
			let line = concentrate_line(&line);
			if line.is_empty() {
				continue;
			}
			if line.starts_with("mtllib ") {
				for (mat_name, mat_data) in ObjMaterialLoader::from_file(line["mtllib ".len()..].trim())?.iter() {
					materials.insert(mat_name.to_string(), mat_data.clone());
				}
				continue;
			} else if line.starts_with("newmtl ") {
				if mat_loader.is_none() {
					mat_loader = Some(ObjMaterialLoader::default());
				}
				mat_loader.as_mut().unwrap().process_line(line_number, line)?;
				continue;
			} else if let Some(ref mut loader) = mat_loader {
				match loader.process_line(line_number, line) {
					Ok(_) => continue,
					Err(_) => {
						loader.finish_material();
						for (mat_name, mat_data) in loader.materials.iter() {
							materials.insert(mat_name.to_string(), mat_data.clone());
						}
						mat_loader = None;
					}
				}
			}
			if let Some(data) = line.strip_prefix("v ") {
				let value = data.trim();
				let mut parts: Vec<&str> = value.split_whitespace().collect();
				while parts.len() < 3 {
					parts.push(" 0.0");
				}
				let x = parts[0].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[0])})?;
				let y = parts[1].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[1])})?;
				let z = parts[2].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[2])})?;
				vertices.push(TVec3::new(x, y, z));
			} else if let Some(data) = line.strip_prefix("vt ") {
				let value = data.trim();
				let mut parts: Vec<&str> = value.split_whitespace().collect();
				while parts.len() < 3 {
					parts.push(" 0.0");
				}
				let x = parts[0].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[0])})?;
				let y = parts[1].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[1])})?;
				let z = parts[2].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[2])})?;
				normals.push(TVec3::new(x, y, z));
			} else if let Some(data) = line.strip_prefix("vn ") {
				let value = data.trim();
				let mut parts: Vec<&str> = value.split_whitespace().collect();
				while parts.len() < 3 {
					parts.push(" 0.0");
				}
				let x = parts[0].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[0])})?;
				let y = parts[1].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[1])})?;
				let z = parts[2].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[2])})?;
				texcoords.push(TVec3::new(x, y, z));
			} else if line.starts_with("f ") {
				let value = line["f ".len()..].trim();
				let parts: Vec<&str> = value.split_whitespace().collect();
				if parts.len() < 3 {
					return Err(ObjError::ParseError {line: line_number, what: format!("Insufficient index count for a face.")});
				}
				let object = if let Some(object) = objects.get_mut(&object_name) {
					object
				} else {
					objects.insert(object_name.clone(), ObjObjects::default());
					objects.get_mut(&object_name).unwrap()
				};
				let group = if let Some(group) = object.get_mut(&group_name) {
					group
				} else {
					object.insert(group_name.clone(), ObjGroups::default());
					object.get_mut(&group_name).unwrap()
				};
				let matgroup = if let Some(matgroup) = group.get_mut(&material_name) {
					matgroup
				} else {
					group.insert(material_name.clone(), ObjMaterialGroup::default());
					group.get_mut(&material_name).unwrap()
				};
				let smthgroup = if let Some(smthgroup) = matgroup.get_mut(&smoothing_group) {
					smthgroup
				} else {
					matgroup.insert(smoothing_group.clone(), ObjMaterialGroup::default());
					matgroup.get_mut(&smoothing_group).unwrap()
				};

			}
		}
		Ok(Self {
			vertices,
			normals,
			texcoords,
			objects,
			materials,
		})
	}
}

/// The material loader for the OBJ mesh
#[derive(Debug, Clone)]
pub struct ObjMaterialLoader {
	/// The path of the MTL file
	pub path: PathBuf,

	/// The materials of the OBJ file
	pub materials: BTreeMap<String, Arc<RwLock<dyn ObjMaterial>>>,

	/// The current loading material name
	pub cur_material_name: String,

	/// The current material fields
	pub cur_material_fields: BTreeMap<String, (usize, String)>,
}

impl ObjMaterialLoader {
	/// Parse a MTL file
	pub fn from_file<P: AsRef<Path>>(path: P) -> Result<BTreeMap<String, Arc<RwLock<dyn ObjMaterial>>>, ObjError> {
		let reader = BufReader::new(File::open(&path)?);
		let mut ret = Self {
			path: PathBuf::from(path.as_ref()),
			materials: BTreeMap::new(),
			cur_material_name: String::default(),
			cur_material_fields: BTreeMap::new(),
		};
		for (line_number, line) in reader.lines().enumerate() {
			let line = line?;
			let line = concentrate_line(&line);
			if line.is_empty() {
				continue;
			}
			ret.process_line(line_number, line)?;
		}
		ret.finish_material();
		Ok(ret.materials)
	}

	/// Process a line of the material
	pub fn process_line(&mut self, line_number: usize, line: &str) -> Result<(), ObjError> {
		if let Some(material_name) = line.strip_prefix("newmtl ") {
			self.finish_material();
			self.cur_material_name = material_name.trim().to_string();
			Ok(())
		} else if let Some((key, value)) = line.split_once(' ') {
			self.cur_material_fields.insert(key.to_string(), (line_number, value.to_string()));
			Ok(())
		} else {
			Err(ObjError::ParseError {
				line: line_number,
				what: format!("Unknown line: `{line}`"),
			})
		}
	}

	/// Check if the current material is a PBR material
	fn cur_is_pbr(&self) -> bool {
		let pbr_slots = [
			"Pr",
			"Pm",
			"Ps",
			"aniso",
			"disp",
		];
		for slot in pbr_slots {
			let pure_color = slot;
			let mapped_color = format!("map_{slot}");
			if self.cur_material_fields.contains_key(pure_color) {
				return true;
			}
			if self.cur_material_fields.contains_key(&mapped_color) {
				return true;
			}
		}
		false
	}

	/// Finish the current material
	pub fn finish_material(&mut self) {
		if self.cur_material_name.is_empty() || self.cur_material_fields.is_empty() {
			return;
		}
		let new_mat: Arc<RwLock<dyn ObjMaterial>> = if self.cur_is_pbr() {
			Arc::new(RwLock::new(ObjMaterialPbr::default()))
		} else {
			Arc::new(RwLock::new(ObjMaterialLegacy::default()))
		};
		static LEGACY_SLOT_MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
		static PBR_SLOT_MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
		let legacy_slot_map = LEGACY_SLOT_MAP.get_or_init(|| {
			[
				("Ka", "ambient"),
				("Kd", "diffuse"),
				("Ks", "specular"),
				("Ns", "specular_power"),
				("norm", "normal"),
				("Ke", "emissive"),
			].into_iter().collect()
		});
		let pbr_slot_map = PBR_SLOT_MAP.get_or_init(|| {
			let mut ret = legacy_slot_map.clone();
			ret.insert("Kd", "albedo");
			ret.insert("Pr", "roughness");
			ret.insert("Pm", "metalness");
			ret.insert("Ps", "sheen");
			ret.insert("aniso", "anisotropy");
			ret.insert("disp", "displacement");
			ret
		});
		let slot_map = if self.cur_is_pbr() {
			&pbr_slot_map
		} else {
			&legacy_slot_map
		};
		let mut mat_lock = new_mat.write().unwrap();
		for (key, value) in self.cur_material_fields.iter() {
			let (line_number, value) = value;
			let mut is_map = false;
			let slot_name = if let Some(suffix) = key.strip_prefix("map_") {
				is_map = true;
				suffix.to_string()
			} else {
				key.to_string()
			};
			if slot_name == "norm" || slot_name == "disp" {
				is_map = true;
			}
			if is_map {
				let slot_name: &str = &slot_name;
				let slot_name_mapped = slot_map.get(&slot_name).unwrap_or(&slot_name).to_string();
				let mut texture_file_path = self.path.clone();
				texture_file_path.set_file_name(value);
				mat_lock.set_by_name(&slot_name_mapped, ObjMaterialComponent::Texture(texture_file_path));
			} else {
				let slot_name: &str = &slot_name;
				let slot_name_mapped = slot_map.get(&slot_name).unwrap_or(&slot_name).to_string();
				let parts: Vec<&str> = value.split_whitespace().collect();
				if parts.len() == 3 {
					let r = if let Ok(number) = parts[0].parse::<f32>() {number} else {eprintln!("Ignored material line in {line_number}: {key} {value}"); continue;};
					let g = if let Ok(number) = parts[1].parse::<f32>() {number} else {eprintln!("Ignored material line in {line_number}: {key} {value}"); continue;};
					let b = if let Ok(number) = parts[2].parse::<f32>() {number} else {eprintln!("Ignored material line in {line_number}: {key} {value}"); continue;};
					mat_lock.set_by_name(&slot_name_mapped, ObjMaterialComponent::Color(Vec4::new(r, g, b, 1.0)));
				} else if let Ok(number) = value.parse::<f32>() {
					mat_lock.set_by_name(&slot_name_mapped, ObjMaterialComponent::Luminance(number));
				} else {
					eprintln!("Ignored material line in {line_number}: {key} {value}");
				}
			}
		}
		drop(mat_lock);
		self.materials.insert(self.cur_material_name.clone(), new_mat);
		self.cur_material_fields = BTreeMap::new();
	}
}
