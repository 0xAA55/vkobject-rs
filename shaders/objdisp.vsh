#version 460

layout(std140, binding = 0) uniform v_scene {
	vec3 light_dir;
	vec3 light_color;
	mat4 view;
	mat4 proj;
};

// Vertex inputs
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;

// Instance input
layout(location = 3) in mat4 transform;

// Output to next stage shaders
layout(location = 1) out vec2 v_texcoord;
layout(location = 2) out vec3 v_normal;

void main()
{
	v_texcoord = texcoord;
	v_normal = mat3(transform) * normal;
	gl_Position = proj * view * transform * vec4(position, 1.0);
}
