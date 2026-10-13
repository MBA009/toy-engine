#version 450

struct Light {
    vec4 position;   // xyz = world position
    vec4 color;      // rgb = color, a = intensity
};

layout(set = 0, binding = 0) uniform Scene {
    mat4 viewProj;
    vec4 cameraPos;
    Light lights[3];
    ivec4 info;      // x = number of active lights
} scene;

// Since Step 3.4 the model matrix is no longer pushed: it arrives per instance (inModel below).
// The block is kept identical to the fragment shader's; only params is used, and only there.
layout(push_constant) uniform Push {
    mat4 model;
    vec4 params;     // x = checker pattern on/off, y = shininess
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inUV;
layout(location = 4) in mat4 inModel;   // per-instance; occupies locations 4, 5, 6 and 7
layout(location = 8) in uint inFlags;   // per-instance flags (bit 31 = selected in the editor)

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;
layout(location = 3) out vec2 fragUV;
layout(location = 4) flat out uint fragFlags;

void main() {
    vec4 world = inModel * vec4(inPos, 1.0);
    gl_Position = scene.viewProj * world;

    fragWorldPos = world.xyz;
    // mat3(model) is correct for rotation/translation/uniform scale.
    // Non-uniform scale would need the inverse-transpose instead.
    fragNormal = mat3(inModel) * inNormal;
    fragColor = inColor;
    fragUV = inUV;
    fragFlags = inFlags;
}
