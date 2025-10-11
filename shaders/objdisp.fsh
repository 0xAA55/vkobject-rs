#version 460

layout(std140, binding = 0) uniform v_scene {
	mat4 view;
	mat4 proj;
	vec3 light_dir;
	vec3 light_color;
	vec3 ambient_color;
};

layout(binding = 1) uniform sampler2D t_albedo;
layout(binding = 2) uniform sampler2D t_normal;

// Input from previous stage shaders
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 binormal;

// Output to attachments
layout(location = 0) out vec4 color;

void main()
{
	mat3 surface_transform = mat3(tangent, binormal, normal);
	vec3 sampled_normal = normalize(texture(t_normal, texcoord).rgb * 2.0 - 1.0);
	vec3 normal = surface_transform * sampled_normal;
	vec3 diffuse = min(ambient_color + max(0.0, dot(normal, -light_dir)), vec3(1.0, 1.0, 1.0));
	vec3 diffuse_color = diffuse * light_color * texture(t_albedo, texcoord).rgb;
	color = vec4(diffuse_color, 1.0);
}


