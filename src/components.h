#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include "math3d.h"

// ---------------------------------------------------------------------------
// Components: plain data. No methods, no inheritance, no virtuals.
// ---------------------------------------------------------------------------

struct Transform {
    Vec3  position{ 0.0f, 0.0f, 0.0f };
    float yaw   = 0.0f;   // radians around Y
    float scale = 1.0f;   // uniform scale
};

struct Spin {
    float speed = 1.0f;   // radians per second around Y
};

struct Bob {
    float amplitude = 0.1f;
    float frequency = 1.0f;
    float baseY     = 0.0f;
    float phase     = 0.0f;
};

struct Velocity {
    Vec3 value{ 0.0f, 0.0f, 0.0f };   // units per second
};

struct MeshRenderer {
    uint32_t meshId     = 0;
    uint32_t materialId = 0;
};

// "Cold" data: needed by the editor later, never touched by the per-frame systems
struct Name {
    std::string value;
};

// One entry per drawable object, ready to feed to the renderer (push constants / instance data)
struct DrawItem {
    Mat4     model;
    uint32_t meshId;
    uint32_t materialId;
};

// Closed form of translate(position) * rotateY(yaw) * scale(s), without the matrix multiplies.
// (ecs_bench.cpp checks it against the composed version.)
inline Mat4 modelMatrix(const Transform& t) {
    float c = std::cos(t.yaw), s = std::sin(t.yaw), k = t.scale;
    Mat4 m = Mat4::identity();
    m.m[0][0] =  c * k;  m.m[0][2] = -s * k;
    m.m[1][1] =  k;
    m.m[2][0] =  s * k;  m.m[2][2] =  c * k;
    m.m[3][0] = t.position.x;
    m.m[3][1] = t.position.y;
    m.m[3][2] = t.position.z;
    return m;
}
