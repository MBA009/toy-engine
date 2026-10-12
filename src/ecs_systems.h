#pragma once
#include <cmath>
#include <vector>
#include "components.h"
#include "ecs.h"

// ---------------------------------------------------------------------------
// Systems: plain functions over component pools.
// Each lists its RAREST component first, since each<> walks the first pool.
// ---------------------------------------------------------------------------

inline void spinSystem(Registry& r, float dt) {
    r.each<Spin, Transform>([dt](Entity, Spin& s, Transform& t) {
        t.yaw += s.speed * dt;
    });
}

// Cheap, memory-bound system: touches only Velocity (12 B) and Transform.
inline void moveSystem(Registry& r, float dt) {
    r.each<Velocity, Transform>([dt](Entity, Velocity& v, Transform& t) {
        t.position.x += v.value.x * dt;
        t.position.z += v.value.z * dt;
    });
}

inline void bobSystem(Registry& r, float time) {
    r.each<Bob, Transform>([time](Entity, Bob& b, Transform& t) {
        t.position.y = b.baseY + b.amplitude * std::sin(time * b.frequency + b.phase);
    });
}

// Render preparation: one DrawItem per drawable entity. This is what the Vulkan
// renderer will consume in Step 3.4 (one push-constant model matrix per item).
inline void buildDrawList(Registry& r, std::vector<DrawItem>& out) {
    out.clear();
    r.each<MeshRenderer, Transform>([&out](Entity, MeshRenderer& m, Transform& t) {
        out.push_back(DrawItem{ modelMatrix(t), m.meshId, m.materialId });
    });
}
