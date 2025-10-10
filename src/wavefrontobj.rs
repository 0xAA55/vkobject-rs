
use std::{
	any::Any,
	cmp::max,
	collections::{BTreeMap, BTreeSet, HashMap},
	fmt::{self, Debug, Formatter},
	fs::File,
	hash::{Hash, Hasher},
	io::{self, BufRead, BufReader, ErrorKind},
	mem::size_of,
	ops::{Add, Sub, Mul, Div, Rem, Neg, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign},
	path::{Path, PathBuf},
	slice,
	str::FromStr,
	sync::{Arc, OnceLock, RwLock},
};
extern crate nalgebra_glm as glm;
use glm::*;
use simba::simd::SimdComplexField;

/// The OBJ error
#[derive(Debug, Clone)]
pub enum ObjError {
	ParseError{line: usize, what: String},
	IOError{kind: ErrorKind, what: String},
	MeshIndicesUnderflow,
	MeshIndicesOverflow,
	NeedTexCoordAndNormal,
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
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
	pub smooth_groups: BTreeMap<i32, Arc<RwLock<ObjSmoothGroup>>>,
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

/// A trait that tells how many operators could be used on a floating number
pub trait FloatOps: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self> + Neg<Output = Self> + PartialEq + PartialOrd + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sized + num_traits::identities::Zero + num_traits::Float + SimdComplexField{}

/// The trait for `TVecN<>` component type
pub trait ObjMeshVecCompType: Default + Clone + Copy + Sized + PartialEq + Eq + Debug + FromStr + Any + FloatOps + Hash + 'static {}
impl<T> ObjMeshVecCompType for T where T: Default + Clone + Copy + Sized + PartialEq + Eq + Debug + FromStr + Any + FloatOps + Hash + 'static {}

/// The trait for indices type
pub trait ObjMeshIndexType: Default + Clone + Copy + Sized + PartialEq + Eq + TryFrom<usize> + TryInto<usize> + Any + Debug + 'static {}
impl<T> ObjMeshIndexType for T where T: Default + Clone + Copy + Sized + PartialEq + Eq + TryFrom<usize> + TryInto<usize> + Any + Debug + 'static {}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjVertices<F>
where
	F: ObjMeshVecCompType {
	/// The `v` part of the VTN vertex
	pub position: TVec3<F>,

	/// The `vt` part of the VTN vertex
	pub texcoord: Option<TVec3<F>>,

	/// The `vn` part of the VTN vertex
	pub normal: Option<TVec3<F>>,

	/// The tangent of the vertex, not come from the OBJ mesh
	pub tangent: Option<TVec3<F>>,
}

#[derive(Default, Debug, Clone)]
pub struct ObjIndexedMesh<E>
where
	E: ObjMeshIndexType {
	/// The object name
	pub object_name: String,

	/// The group name
	pub group_name: String,

	/// The material name
	pub material_name: String,

	/// The smooth group
	pub smooth_group: i32,

	/// The face indices
	pub face_indices: Vec<(E, E, E)>,

	/// The line indices
	pub line_indices: Vec<Vec<E>>,
}

#[derive(Default, Debug, Clone)]
pub struct ObjUnindexedMesh<F>
where
	F: ObjMeshVecCompType {
	/// The object name
	pub object_name: String,

	/// The group name
	pub group_name: String,

	/// The material name
	pub material_name: String,

	/// The smooth group
	pub smooth_group: i32,

	/// The face indices
	pub faces: Vec<(ObjVertices<F>, ObjVertices<F>, ObjVertices<F>)>,

	/// The line indices
	pub lines: Vec<Vec<TVec3<F>>>,
}

#[derive(Default, Debug, Clone)]
pub struct ObjIndexedMeshSet<F, E>
where
	F: ObjMeshVecCompType,
	E: ObjMeshIndexType {
	/// The face vertices
	pub face_vertices: Vec<ObjVertices<F>>,

	/// The line vertices
	pub line_vertices: Vec<TVec3<F>>,

	/// The meshes
	pub meshes: Vec<ObjIndexedMesh<E>>,
}

/// The raw objmesh that's just the intepreted result of a obj file.
#[derive(Default, Debug, Clone)]
pub struct ObjMesh<F>
where
	F: ObjMeshVecCompType {
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

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct LineVert<F: ObjMeshVecCompType> {
	x: F,
	y: F,
	z: F,
}

impl<F> Hash for LineVert<F>
where F: ObjMeshVecCompType {
	fn hash<H: Hasher>(&self, state: &mut H) {
		match size_of::<F>() {
			1 => {let data: Vec<u8 > = unsafe {vec![*(&self.x as *const F as *const u8 ), *(&self.y as *const F as *const u8 ), *(&self.z as *const F as *const u8 )]}; state.write(unsafe {slice::from_raw_parts(data.as_ptr(), data.len())});}
			2 => {let data: Vec<u16> = unsafe {vec![*(&self.x as *const F as *const u16), *(&self.y as *const F as *const u16), *(&self.z as *const F as *const u16)]}; state.write(unsafe {slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)});}
			4 => {let data: Vec<u32> = unsafe {vec![*(&self.x as *const F as *const u32), *(&self.y as *const F as *const u32), *(&self.z as *const F as *const u32)]}; state.write(unsafe {slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)});}
			8 => {let data: Vec<u64> = unsafe {vec![*(&self.x as *const F as *const u64), *(&self.y as *const F as *const u64), *(&self.z as *const F as *const u64)]}; state.write(unsafe {slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)});}
			o => panic!("Invalid primitive type of `<F>`, the size of this type is `{o}`"),
		}
	}
}

fn get_face_vert_index<V, E>(face_vertices_map: &mut HashMap<V, E>, face_vert: &V) -> Result<E, ObjError>
where
	V: PartialEq + Eq + Hash + Clone + Copy + Sized,
	E: ObjMeshIndexType {
	if let Some(index) = face_vertices_map.get(face_vert) {
		Ok(*index)
	} else {
		let new_index = face_vertices_map.len();
		let new_ret = E::try_from(new_index).map_err(|_| ObjError::MeshIndicesOverflow)?;
		face_vertices_map.insert(*face_vert, new_ret);
		Ok(new_ret)
	}
}
fn get_line_vert_index<F, E>(line_vertices_map: &mut HashMap<LineVert<F>, E>, line_vert: &TVec3<F>) -> Result<E, ObjError>
where
	F: ObjMeshVecCompType,
	E: ObjMeshIndexType {
	let line_vert = LineVert {
		x: line_vert.x,
		y: line_vert.y,
		z: line_vert.z,
	};
	if let Some(index) = line_vertices_map.get(&line_vert) {
		Ok(*index)
	} else {
		let new_index = line_vertices_map.len();
		let new_ret = E::try_from(new_index).map_err(|_| ObjError::MeshIndicesOverflow)?;
		line_vertices_map.insert(line_vert, new_ret);
		Ok(new_ret)
	}
}

impl<F> ObjMesh<F>
where
	F: ObjMeshVecCompType {
	/// Parse an OBJ file.
	pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ObjError> {
		let reader = BufReader::new(File::open(&path)?);
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
		let mut last_smoothing_group = None;
		for (line_number, line) in reader.lines().enumerate() {
			let line = line?;
			let line = concentrate_line(&line);
			if line.is_empty() {
				continue;
			}
			if let Some(data) = line.strip_prefix("mtllib ") {
				let mtllib_fn = data.trim();
				let mut mtllib_path = PathBuf::from(path.as_ref());
				mtllib_path.set_file_name(mtllib_fn);
				match ObjMaterialLoader::from_file(&mtllib_path) {
					Ok(matlib) => {
						for (mat_name, mat_data) in matlib.iter() {
							materials.insert(mat_name.to_string(), mat_data.clone());
						}
					}
					Err(e) => {
						eprintln!("Parse material library `{mtllib_fn}` from `{}` failed: {e:?}", mtllib_path.display());
					}
				}
				continue;
			} else if line.starts_with("newmtl ") {
				if mat_loader.is_none() {
					mat_loader = Some(ObjMaterialLoader {
						path: PathBuf::from(path.as_ref()),
						materials: BTreeMap::new(),
						cur_material_name: String::default(),
						cur_material_fields: BTreeMap::new(),
					});
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
				texcoords.push(TVec3::new(x, y, z));
			} else if let Some(data) = line.strip_prefix("vn ") {
				let value = data.trim();
				let mut parts: Vec<&str> = value.split_whitespace().collect();
				while parts.len() < 3 {
					parts.push(" 0.0");
				}
				let x = parts[0].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[0])})?;
				let y = parts[1].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[1])})?;
				let z = parts[2].parse::<F>().ok().ok_or(ObjError::ParseError {line: line_number, what: format!("Could not parse `{}`", parts[2])})?;
				normals.push(TVec3::new(x, y, z));
			} else if line.starts_with("f ") || line.starts_with("l ") {
				let value = line["f ".len()..].trim();
				let parts: Vec<&str> = value.split_whitespace().collect();
				if parts.len() < 3 {
					eprintln!("Parse line `{line_number}` error: `{line}`: Insufficient index count for a face.");
					continue;
				}
				if last_smoothing_group.is_none() {
					let object = if let Some(object) = objects.get_mut(&object_name) {
						object
					} else {
						objects.insert(object_name.clone(), ObjObjects::default());
						objects.get_mut(&object_name).unwrap()
					};
					let group = if let Some(group) = object.groups.get_mut(&group_name) {
						group
					} else {
						object.groups.insert(group_name.clone(), ObjGroups::default());
						object.groups.get_mut(&group_name).unwrap()
					};
					let matgroup = if let Some(matgroup) = group.material_groups.get_mut(&material_name) {
						matgroup
					} else {
						group.material_groups.insert(material_name.clone(), ObjMaterialGroup::default());
						group.material_groups.get_mut(&material_name).unwrap()
					};
					let smthgroup = if let Some(smthgroup) = matgroup.smooth_groups.get_mut(&smoothing_group) {
						smthgroup
					} else {
						matgroup.smooth_groups.insert(smoothing_group, Arc::new(RwLock::new(ObjSmoothGroup::default())));
						matgroup.smooth_groups.get_mut(&smoothing_group).unwrap()
					};
					last_smoothing_group = Some(smthgroup.clone());
				}
				let mut group_lock = last_smoothing_group.as_ref().unwrap().write().unwrap();
				if line.starts_with("f ") {
					// Process as triangle strip
					let mut vtn_indices: Vec<ObjVTNIndex> = Vec::with_capacity(parts.len());
					for part in parts.iter() {
						vtn_indices.push(ObjVTNIndex::parse(line_number, part)?);
					}
					for i in 1..(parts.len() - 1) {
						group_lock.triangles.push((vtn_indices[0], vtn_indices[i], vtn_indices[i + 1]));
					}
				} else {
					// Process as line link
					let mut indices: Vec<u32> = Vec::with_capacity(parts.len());
					for part in parts.iter() {
						match part.parse() {
							Ok(index) => indices.push(index),
							Err(e) => {
								eprintln!("Parse line {line_number} error: `{line}`: {e:?}");
								continue;
							}
						}
					}
					group_lock.lines.push(indices);
				}
			} else if let Some(data) = line.strip_prefix("o ") {
				object_name = data.trim().to_string();
				last_smoothing_group = None;
			} else if let Some(data) = line.strip_prefix("g ") {
				group_name = data.trim().to_string();
				last_smoothing_group = None;
			} else if let Some(data) = line.strip_prefix("usemtl ") {
				material_name = data.trim().to_string();
				last_smoothing_group = None;
			} else if let Some(data) = line.strip_prefix("s ") {
				smoothing_group = data.trim().parse().unwrap_or(0);
				last_smoothing_group = None;
			} else {
				eprintln!("Ignoring line `{line_number}`: unknown `{line}`");
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

	/// Get the number of the mesh groups
	pub fn get_num_groups(&self) -> usize {
		let mut ret = 0;
		for object in self.objects.values() {
			for group in object.groups.values() {
				for matgroup in group.material_groups.values() {
					ret += matgroup.smooth_groups.len();
				}
			}
		}
		ret
	}

	/// Convert to VTN vertices and associated indices
	pub fn convert_to_indexed_meshes<E>(&self) -> Result<ObjIndexedMeshSet<F, E>, ObjError>
	where
		E: ObjMeshIndexType {
		let mut face_vertices_map: HashMap<ObjVTNIndex, E> = HashMap::new();
		let mut line_vertices_map: HashMap<LineVert<F>, E> = HashMap::new();
		let mut ret: ObjIndexedMeshSet<F, E> = ObjIndexedMeshSet::default();
		for (object_name, object) in self.objects.iter() {
			for (group_name, group) in object.groups.iter() {
				for (material_name, matgroup) in group.material_groups.iter() {
					for (smooth_group, smthgroup) in matgroup.smooth_groups.iter() {
						let lock = smthgroup.read().unwrap();
						let mut lines_vert_indices: Vec<Vec<E>> = Vec::with_capacity(lock.lines.len());
						let mut triangle_vert_indices: Vec<(E, E, E)> = Vec::with_capacity(lock.triangles.len());
						for line in lock.lines.iter() {
							let mut line_vert_indices: Vec<E> = Vec::with_capacity(line.len());
							for vert_idx in line.iter() {
								let vert = self.vertices[*vert_idx as usize - 1];
								line_vert_indices.push(get_line_vert_index(&mut line_vertices_map, &vert)?);
							}
							lines_vert_indices.push(line_vert_indices);
						}
						for triangle in lock.triangles.iter() {
							let vert1 = get_face_vert_index(&mut face_vertices_map, &triangle.0)?;
							let vert2 = get_face_vert_index(&mut face_vertices_map, &triangle.1)?;
							let vert3 = get_face_vert_index(&mut face_vertices_map, &triangle.2)?;
							triangle_vert_indices.push((vert1, vert2, vert3));
						}
						ret.meshes.push(ObjIndexedMesh {
							object_name: object_name.clone(),
							group_name: group_name.clone(),
							material_name: material_name.clone(),
							smooth_group: *smooth_group,
							face_indices: triangle_vert_indices,
							line_indices: lines_vert_indices,
						});
					}
				}
			}
		}
		ret.face_vertices.resize(face_vertices_map.len(), ObjVertices::default());
		ret.line_vertices.resize(line_vertices_map.len(), TVec3::default());
		for (vtn, vi) in face_vertices_map.iter() {
			let vi: usize = (*vi).try_into().map_err(|_| ObjError::MeshIndicesOverflow)?;
			ret.face_vertices[vi] = ObjVertices {
				position: if vtn.v == 0 {return Err(ObjError::MeshIndicesUnderflow)} else {self.vertices[vtn.v as usize - 1]},
				texcoord: if vtn.vt == 0 {None} else {Some(self.texcoords[vtn.vt as usize - 1])},
				normal: if vtn.vn == 0 {None} else {Some(self.normals[vtn.vn as usize - 1])},
				tangent: None,
			};
		}
		for (lv, li) in line_vertices_map.iter() {
			let li: usize = (*li).try_into().map_err(|_| ObjError::MeshIndicesOverflow)?;
			ret.line_vertices[li] = TVec3::new(lv.x, lv.y, lv.z);
		}
		Ok(ret)
	}

	/// Convert to VTN vertices without indices
	pub fn convert_to_unindexed_meshes(&self) -> Result<Vec<ObjUnindexedMesh<F>>, ObjError> {
		let mut ret: Vec<ObjUnindexedMesh<F>> = Vec::with_capacity(self.get_num_groups());
		fn get_face_vert<F>(vertices: &[TVec3<F>], texcoords: &[TVec3<F>], normals: &[TVec3<F>], vtn: &ObjVTNIndex) -> Result<ObjVertices<F>, ObjError>
		where
			F: ObjMeshVecCompType {
			Ok(ObjVertices {
				position: if vtn.v == 0 {return Err(ObjError::MeshIndicesUnderflow)} else {vertices[vtn.v as usize - 1]},
				texcoord: if vtn.vt == 0 {None} else {Some(texcoords[vtn.vt as usize - 1])},
				normal: if vtn.vn == 0 {None} else {Some(normals[vtn.vn as usize - 1])},
				tangent: None,
			})
		}
		for (object_name, object) in self.objects.iter() {
			for (group_name, group) in object.groups.iter() {
				for (material_name, matgroup) in group.material_groups.iter() {
					for (smooth_group, smthgroup) in matgroup.smooth_groups.iter() {
						let lock = smthgroup.read().unwrap();
						let mut lines: Vec<Vec<TVec3<F>>> = Vec::with_capacity(lock.lines.len());
						let mut faces: Vec<(ObjVertices<F>, ObjVertices<F>, ObjVertices<F>)> = Vec::with_capacity(lock.triangles.len());
						for line in lock.lines.iter() {
							let mut line_verts: Vec<TVec3<F>> = Vec::with_capacity(line.len());
							for vert_idx in line.iter() {
								let vert = self.vertices[*vert_idx as usize - 1];
								line_verts.push(vert);
							}
							lines.push(line_verts);
						}
						for triangle in lock.triangles.iter() {
							let vert1 = get_face_vert(&self.vertices, &self.texcoords, &self.normals, &triangle.0)?;
							let vert2 = get_face_vert(&self.vertices, &self.texcoords, &self.normals, &triangle.1)?;
							let vert3 = get_face_vert(&self.vertices, &self.texcoords, &self.normals, &triangle.2)?;
							faces.push((vert1, vert2, vert3));
						}
						ret.push(ObjUnindexedMesh {
							object_name: object_name.clone(),
							group_name: group_name.clone(),
							material_name: material_name.clone(),
							smooth_group: *smooth_group,
							faces,
							lines,
						});
					}
				}
			}
		}
		Ok(ret)
	}
}

fn float_is_zero_restrict<F>(f: F) -> bool
where
	F: ObjMeshVecCompType {
	match size_of::<F>() {
		1 => unsafe {*(&f as *const F as *const u8 ) == 0},
		2 => unsafe {*(&f as *const F as *const u16) == 0},
		4 => unsafe {*(&f as *const F as *const u32) == 0},
		8 => unsafe {*(&f as *const F as *const u64) == 0},
		o => panic!("Invalid primitive type of `<F>`, the size of this type is `{o}`"),
	}
}

impl<F> ObjVertices<F>
where
	F: ObjMeshVecCompType {
	/// Generate tangents per 3 vertices
	pub fn generate_tangents(v1: &mut Self, v2: &mut Self, v3: &mut Self) {
		let pos0 = v1.position;
		let pos1 = v2.position;
		let pos2 = v3.position;
		let uv0 = v1.texcoord.unwrap_or(TVec3::default()).xy();
		let uv1 = v2.texcoord.unwrap_or(TVec3::default()).xy();
		let uv2 = v3.texcoord.unwrap_or(TVec3::default()).xy();
		let normal0 = v1.normal.unwrap_or(TVec3::default());
		let normal1 = v2.normal.unwrap_or(TVec3::default());
		let normal2 = v3.normal.unwrap_or(TVec3::default());

		let delta_pos1 = pos1 - pos0;
		let delta_pos2 = pos2 - pos0;

		let delta_uv1 = uv1 - uv0;
		let delta_uv2 = uv2 - uv0;

		let f = F::from(1.0).unwrap() / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);

		let tangent = TVec3::new(
			f * (delta_uv2.y * delta_pos1.x - delta_uv1.y * delta_pos2.x),
			f * (delta_uv2.y * delta_pos1.y - delta_uv1.y * delta_pos2.y),
			f * (delta_uv2.y * delta_pos1.z - delta_uv1.y * delta_pos2.z),
		);

		let bitangent = TVec3::new(
			f * (-delta_uv2.x * delta_pos1.x + delta_uv1.x * delta_pos2.x),
			f * (-delta_uv2.x * delta_pos1.y + delta_uv1.x * delta_pos2.y),
			f * (-delta_uv2.x * delta_pos1.z + delta_uv1.x * delta_pos2.z),
		);

		let tangent0 = Self::orthogonalize_tangent(tangent, normal0);
		let tangent1 = Self::orthogonalize_tangent(tangent, normal1);
		let tangent2 = Self::orthogonalize_tangent(tangent, normal2);

		let tangent0 = Self::ensure_right_handed(tangent0, normal0, bitangent);
		let tangent1 = Self::ensure_right_handed(tangent1, normal1, bitangent);
		let tangent2 = Self::ensure_right_handed(tangent2, normal2, bitangent);

		v1.tangent = Some(tangent0);
		v2.tangent = Some(tangent1);
		v3.tangent = Some(tangent2);
	}

	/// Orthogonalize tangents using the Gram-Schmidt procedure
	fn orthogonalize_tangent(tangent: TVec3<F>, normal: TVec3<F>) -> TVec3<F> {
		let n_dot_t = normal.dot(&tangent);
		let projected = tangent - normal * n_dot_t;
		projected.normalize()
	}

	/// Make sure the tangent space is right-handed
	fn ensure_right_handed(tangent: TVec3<F>, normal: TVec3<F>, bitangent: TVec3<F>) -> TVec3<F> {
		let calculated_bitangent = normal.cross(&tangent);
		let dot_product = calculated_bitangent.dot(&bitangent);

		// If the dot product is negative, it means it is a left-handed system and the tangent needs to be flipped
		if dot_product < F::from(0.0).unwrap() {
			-tangent
		} else {
			tangent
		}
	}
}

impl<F> ObjUnindexedMesh<F>
where
	F: ObjMeshVecCompType {
	/// Get the dimension data of vertex position, texcoord, normal
	pub fn get_vert_dims(&self) -> (u32, u32, u32) {
		let mut max_position = 0;
		let mut max_texcoord = 0;
		let mut max_normal = 0;
		for vert in self.faces.iter().flat_map(|(a, b, c)| [a, b, c]) {
			match max_position {
				0 => {
					if !float_is_zero_restrict(vert.position.z) {max_position = max(max_position, 3)}
					else if !float_is_zero_restrict(vert.position.y) {max_position = max(max_position, 2)}
					else if !float_is_zero_restrict(vert.position.x) {max_position = max(max_position, 1)}
				}
				1 => {
					if !float_is_zero_restrict(vert.position.z) {max_position = max(max_position, 3)}
					else if !float_is_zero_restrict(vert.position.y) {max_position = max(max_position, 2)}
				}
				2 => {
					if !float_is_zero_restrict(vert.position.z) {max_position = max(max_position, 3)}
				}
				_ => break,
			}
		}
		for vert in self.faces.iter().flat_map(|(a, b, c)| [a, b, c]) {
			if let Some(texcoord) = vert.texcoord {
				match max_texcoord {
					0 => {
						if !float_is_zero_restrict(texcoord.z) {max_texcoord = max(max_texcoord, 3)}
						else if !float_is_zero_restrict(texcoord.y) {max_texcoord = max(max_texcoord, 2)}
						else if !float_is_zero_restrict(texcoord.x) {max_texcoord = max(max_texcoord, 1)}
					}
					1 => {
						if !float_is_zero_restrict(texcoord.z) {max_texcoord = max(max_texcoord, 3)}
						else if !float_is_zero_restrict(texcoord.y) {max_texcoord = max(max_texcoord, 2)}
					}
					2 => {
						if !float_is_zero_restrict(texcoord.z) {max_texcoord = max(max_texcoord, 3)}
					}
					_ => break,
				}
			}
		}
		for vert in self.faces.iter().flat_map(|(a, b, c)| [a, b, c]) {
			if let Some(normal) = vert.normal {
				match max_normal {
					0 => {
						if !float_is_zero_restrict(normal.z) {max_normal = max(max_normal, 3)}
						else if !float_is_zero_restrict(normal.y) {max_normal = max(max_normal, 2)}
						else if !float_is_zero_restrict(normal.x) {max_normal = max(max_normal, 1)}
					}
					1 => {
						if !float_is_zero_restrict(normal.z) {max_normal = max(max_normal, 3)}
						else if !float_is_zero_restrict(normal.y) {max_normal = max(max_normal, 2)}
					}
					2 => {
						if !float_is_zero_restrict(normal.z) {max_normal = max(max_normal, 3)}
					}
					_ => break,
				}
			}
		}
		(max_position, max_texcoord, max_normal)
	}

	/// Only the unindexed mesh could be able to generate tangent
	pub fn generate_tangents(&mut self) -> Result<(), ObjError> {
		let (_, tdim, ndim) = self.get_vert_dims();
		if tdim == 0 || ndim == 0 {
			Err(ObjError::NeedTexCoordAndNormal)
		} else {
			for (v1, v2, v3) in self.faces.iter_mut() {
				ObjVertices::generate_tangents(v1, v2, v3);
			}
			Ok(())
		}
	}

	/// Convert to the indexed mesh
	pub fn convert_to_indexed_meshes<E>(unindexed_meshes: &[Self]) -> Result<ObjIndexedMeshSet<F, E>, ObjError>
	where
		E: ObjMeshIndexType {
		let mut face_vertices_map: HashMap<ObjVertices<F>, E> = HashMap::new();
		let mut line_vertices_map: HashMap<LineVert<F>, E> = HashMap::new();
		let mut ret: ObjIndexedMeshSet<F, E> = ObjIndexedMeshSet::default();
		for uimesh in unindexed_meshes.iter() {
			let mut lines_vert_indices: Vec<Vec<E>> = Vec::with_capacity(uimesh.lines.len());
			let mut triangle_vert_indices: Vec<(E, E, E)> = Vec::with_capacity(uimesh.faces.len());
			for line in uimesh.lines.iter() {
				let mut line_vert_indices: Vec<E> = Vec::with_capacity(line.len());
				for vert in line.iter() {
					line_vert_indices.push(get_line_vert_index(&mut line_vertices_map, vert)?);
				}
				lines_vert_indices.push(line_vert_indices);
			}
			for triangle in uimesh.faces.iter() {
				let vert1 = get_face_vert_index(&mut face_vertices_map, &triangle.0)?;
				let vert2 = get_face_vert_index(&mut face_vertices_map, &triangle.1)?;
				let vert3 = get_face_vert_index(&mut face_vertices_map, &triangle.2)?;
				triangle_vert_indices.push((vert1, vert2, vert3));
			}
			ret.meshes.push(ObjIndexedMesh {
				object_name: uimesh.object_name.clone(),
				group_name: uimesh.group_name.clone(),
				material_name: uimesh.material_name.clone(),
				smooth_group: uimesh.smooth_group,
				face_indices: triangle_vert_indices,
				line_indices: lines_vert_indices,
			});
		}
		ret.face_vertices.resize(face_vertices_map.len(), ObjVertices::default());
		ret.line_vertices.resize(line_vertices_map.len(), TVec3::default());
		for (fv, vi) in face_vertices_map.iter() {
			let vi: usize = (*vi).try_into().map_err(|_| ObjError::MeshIndicesOverflow)?;
			ret.face_vertices[vi] = *fv;
		}
		for (lv, li) in line_vertices_map.iter() {
			let li: usize = (*li).try_into().map_err(|_| ObjError::MeshIndicesOverflow)?;
			ret.line_vertices[li] = TVec3::new(lv.x, lv.y, lv.z);
		}
		Ok(ret)
	}
}

impl<F, E> ObjIndexedMeshSet<F, E>
where
	F: ObjMeshVecCompType,
	E: ObjMeshIndexType {
	/// Get the dimension data of vertex position, texcoord, normal
	pub fn get_vert_dims(&self) -> (u32, u32, u32) {
		let mut max_position = 0;
		let mut max_texcoord = 0;
		let mut max_normal = 0;
		for vert in self.face_vertices.iter() {
			match max_position {
				0 => {
					if !float_is_zero_restrict(vert.position.z) {max_position = max(max_position, 3)}
					else if !float_is_zero_restrict(vert.position.y) {max_position = max(max_position, 2)}
					else if !float_is_zero_restrict(vert.position.x) {max_position = max(max_position, 1)}
				}
				1 => {
					if !float_is_zero_restrict(vert.position.z) {max_position = max(max_position, 3)}
					else if !float_is_zero_restrict(vert.position.y) {max_position = max(max_position, 2)}
				}
				2 => {
					if !float_is_zero_restrict(vert.position.z) {max_position = max(max_position, 3)}
				}
				_ => break,
			}
		}
		for vert in self.face_vertices.iter() {
			if let Some(texcoord) = vert.texcoord {
				match max_texcoord {
					0 => {
						if !float_is_zero_restrict(texcoord.z) {max_texcoord = max(max_texcoord, 3)}
						else if !float_is_zero_restrict(texcoord.y) {max_texcoord = max(max_texcoord, 2)}
						else if !float_is_zero_restrict(texcoord.x) {max_texcoord = max(max_texcoord, 1)}
					}
					1 => {
						if !float_is_zero_restrict(texcoord.z) {max_texcoord = max(max_texcoord, 3)}
						else if !float_is_zero_restrict(texcoord.y) {max_texcoord = max(max_texcoord, 2)}
					}
					2 => {
						if !float_is_zero_restrict(texcoord.z) {max_texcoord = max(max_texcoord, 3)}
					}
					_ => break,
				}
			}
		}
		for vert in self.face_vertices.iter() {
			if let Some(normal) = vert.normal {
				match max_normal {
					0 => {
						if !float_is_zero_restrict(normal.z) {max_normal = max(max_normal, 3)}
						else if !float_is_zero_restrict(normal.y) {max_normal = max(max_normal, 2)}
						else if !float_is_zero_restrict(normal.x) {max_normal = max(max_normal, 1)}
					}
					1 => {
						if !float_is_zero_restrict(normal.z) {max_normal = max(max_normal, 3)}
						else if !float_is_zero_restrict(normal.y) {max_normal = max(max_normal, 2)}
					}
					2 => {
						if !float_is_zero_restrict(normal.z) {max_normal = max(max_normal, 3)}
					}
					_ => break,
				}
			}
		}
		(max_position, max_texcoord, max_normal)
	}
}

/// The material component
#[derive(Clone)]
pub enum ObjMaterialComponent {
	Texture(PathBuf),
	Color(Vec4),
	Luminance(f32),
}

impl Debug for ObjMaterialComponent {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		match self {
			Self::Texture(filename) => write!(f, "Texture(\"{}\")", filename.display()),
			Self::Color(c) => write!(f, "Color({}, {}, {}, {})", c.x, c.y, c.z, c.w),
			Self::Luminance(lum) => write!(f, "Luminance({lum})"),
		}
	}
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
pub trait ObjMaterial: Debug + Any {
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
