#version 460

layout(std140, binding = 0) uniform v_scene {
	vec3 camera;
	mat4 view;
	mat4 proj;
};

// Vertex inputs
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

// Instance input
layout(location = 4) in mat4 transform;

// Output to next stage shaders
layout(location = 1) out vec2 v_texcoord;
layout(location = 2) out vec3 v_normal;
layout(location = 3) out vec3 v_tangent;
layout(location = 4) out vec3 v_binormal;
layout(location = 5) out vec3 v_position;

void main()
{
	v_texcoord = texcoord;
	mat3 rotation = mat3(transform);
	v_normal = rotation * normal;
	v_tangent = rotation * tangent;
	v_binormal = cross(v_normal, v_tangent);
	v_position = (transform * vec4(position, 1.0)).xyz;
	gl_Position = proj * view * transform * vec4(position, 1.0);
}
