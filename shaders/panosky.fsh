#version 460

vec2 frag_coord = gl_FragCoord.xy;

layout(std140, binding = 0) uniform v_scene {
	vec2 resolution;
	mat4 view;
	mat4 proj;
};

layout(binding = 1) uniform sampler2D t_pano;

// Output to attachments
layout(location = 0) out vec4 color;

void main()
{
	vec2 uv = frag_coord / resolution;

	// Convert UV to NDC [-1, 1]
	vec2 ndc = 2.0 * uv - 1.0;

	// Create clip space point at far plane (z=1.0)
	vec4 clip_pos = vec4(ndc, 1.0, 1.0);

	// Calculate view space position by applying inverse projection
	mat4 inv_proj = inverse(proj);
	vec4 view_pos = inv_proj * clip_pos;
	view_pos /= view_pos.w; // Perspective divide

	// Get view space direction and normalize (camera at origin)
	vec3 view_dir = normalize(view_pos.xyz);

	// Convert to world space direction using inverse view matrix
	mat4 inv_view = inverse(view);
	vec3 world_dir = normalize((inv_view * vec4(view_dir, 0.0)).xyz);

	// Convert world direction to equirectangular UV
	// Longitude from atan2(z, x), latitude from asin(y)
	float longitude = atan(world_dir.z, world_dir.x); // atan2(z, x) for azimuthal angle
	float latitude = asin(world_dir.y); // elevation angle

	// Map to [0,1] range
	float u = 0.5 + longitude / (2.0 * 3.14159265359);
	float v = 0.5 + latitude / 3.14159265359;

	// Sample the panorama texture
	color = texture(t_pano, vec2(u, v));
}
