#version 460

layout(std140, binding = 0) uniform v_scene {
	vec3 light_dir;
	vec3 light_color;
	mat4 view;
	mat4 proj;
};

layout(std140, binding = 1) uniform v_object {
	vec3 obj_color;
	vec4 obj_specular;
};

layout(binding = 1) uniform sampler2D t_albedo;
layout(binding = 2) uniform sampler2D t_normal;

// Input from previous stage shaders
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;

// Output to attachments
layout(location = 0) out vec4 color;

void main()
{
	vec3 diffuse_color = obj_color * texture(t_albedo, texcoord).rgb;
	color = vec4(diffuse_color, 1.0);
}


