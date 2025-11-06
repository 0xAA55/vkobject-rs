#version 460

layout(std140, binding = 0) uniform v_scene {
	vec3 camera;
	mat4 view;
	mat4 proj;
};

layout(binding = 1) uniform sampler2D t_albedo;
layout(binding = 2) uniform sampler2D t_normal;
layout(binding = 3) uniform sampler2D t_pano;

// Input from previous stage shaders
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 binormal;
layout(location = 5) in vec3 position;

// Output to attachments
layout(location = 0) out vec4 color;

void main()
{
	mat3 surface_transform = mat3(tangent, binormal, normal);
	vec3 sampled_normal = normalize(texture(t_normal, texcoord).rgb * 2.0 - 1.0);
	vec3 normal_world = normalize(surface_transform * sampled_normal);
	vec3 view_dir = normalize(camera - position);
	vec3 reflect_dir = reflect(-view_dir, normal_world);
	float longitude = atan(reflect_dir.z, reflect_dir.x);
	float latitude = asin(reflect_dir.y);
	float u = 0.5 + longitude / (2.0 * 3.14159265359);
	float v = 0.5 + latitude / 3.14159265359;
	vec4 pano_color = texture(t_pano, vec2(u, v));
	vec4 albedo_color = texture(t_albedo, texcoord);
	color = pano_color + albedo_color;
}


