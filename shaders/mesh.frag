#version 450

struct Light {
    vec4 position;
    vec4 color;
};

// Must match the block in mesh.vert exactly
layout(set = 0, binding = 0) uniform Scene {
    mat4 viewProj;
    vec4 cameraPos;
    Light lights[3];
    ivec4 info;      // x = number of active lights, y = lighting enabled (1) or flat albedo (0)
} scene;

// Set 1: the material's base color texture (a white 1x1 for untextured meshes). Sampled through an
// _SRGB image view, so the value arrives already linear.
layout(set = 1, binding = 0) uniform sampler2D baseColorTex;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 params;
} pc;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec2 fragUV;
layout(location = 4) flat in uint fragFlags;

layout(location = 0) out vec4 outColor;

void main() {
    // Interpolation shortens normals, so re-normalize (same as Phase 1)
    vec3 N = normalize(fragNormal);
    if (!gl_FrontFacing) N = -N;   // back face of a double-sided material: light the side the camera sees
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);

    // Our colors are authored as sRGB; lighting math must happen in linear space
    vec4 texel = texture(baseColorTex, fragUV);
    // Alpha-tested materials (leaves, fences, glTF "MASK"/"BLEND"): params.z is the cutoff, 0 = opaque
    if (pc.params.z > 0.0 && texel.a < pc.params.z) discard;
    vec3 albedo = pow(fragColor, vec3(2.2)) * texel.rgb;

    // Procedural checker for the floor, based on world position so squares stay square
    if (pc.params.x > 0.5) {
        float c = mod(floor(fragWorldPos.x * 4.0) + floor(fragWorldPos.z * 4.0), 2.0);
        albedo *= mix(0.25, 1.0, c);
    }

    // Selected in the editor: an orange glow, strongest at grazing angles (a rim), so the
    // outline reads even on a dark or flat-shaded surface
    vec3 glow = vec3(0.0);
    if ((fragFlags & 0x80000000u) != 0u) {
        float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);
        glow = vec3(1.0, 0.55, 0.1) * (0.20 + 1.2 * rim);
    }

    // Debug toggle: show the surface color with no lighting at all
    if (scene.info.y == 0) {
        outColor = vec4(albedo + glow, 1.0);
        return;
    }

    float shininess = pc.params.y;

    vec3 result = albedo * 0.12; // ambient

    for (int i = 0; i < scene.info.x; i++) {
        vec3 toLight = scene.lights[i].position.xyz - fragWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / dist;
        vec3 H = normalize(L + V);

        // Light fades with distance (point lights, unlike Phase 1's directional light)
        float atten = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
        vec3 radiance = scene.lights[i].color.rgb * scene.lights[i].color.a * atten;

        float diff = max(dot(N, L), 0.0);
        float spec = (diff > 0.0) ? pow(max(dot(N, H), 0.0), shininess) * 0.5 : 0.0;

        result += (albedo * diff + vec3(spec)) * radiance;
    }

    // The sRGB swapchain converts linear -> sRGB on write, so no manual gamma here
    outColor = vec4(result + glow, 1.0);
}
