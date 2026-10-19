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

// Rotates `current` toward `target` (radians) by at most maxStep, the short way round
inline float turnToward(float current, float target, float maxStep) {
    const float pi = 3.14159265358979f;
    float d = std::fmod(target - current, 2.0f * pi);
    if (d > pi) d -= 2.0f * pi;
    if (d < -pi) d += 2.0f * pi;
    if (std::fabs(d) <= maxStep) return target;
    return current + (d > 0.0f ? maxStep : -maxStep);
}

// Moves every walking agent along its path. Movement is measured on the ground plane (x,z); the height
// follows the path's own heights, so ramps work. The agent turns to face its direction of travel (a model's
// local +Z is its front, as in glTF). Reaching a corner snaps exactly onto it, and any leftover movement
// for this frame carries on to the next corner, so a big time step never overshoots or cuts corners.
inline void pathFollowSystem(Registry& r, float dt) {
    if (dt <= 0.0f) return;
    r.each<AIAgent, Transform>([dt](Entity, AIAgent& a, Transform& t) {
        float budget = a.moveSpeed * dt;
        while (a.walking() && budget > 0.0f) {
            const Vec3& corner = a.path[a.waypoint];
            const Vec3 target = { corner.x, corner.y + a.footOffset, corner.z };
            const float dx = target.x - t.position.x, dz = target.z - t.position.z;
            const float dist = std::sqrt(dx * dx + dz * dz);
            if (dist > 1e-6f) t.yaw = turnToward(t.yaw, std::atan2(dx, dz), a.turnSpeed * dt);
            if (dist <= budget) {                       // reach this corner and keep going with what is left
                t.position = target;
                budget -= dist;
                a.waypoint++;
            } else {                                    // partway along this leg
                const float f = budget / dist;
                t.position.x += dx * f;
                t.position.z += dz * f;
                t.position.y += (target.y - t.position.y) * f;
                budget = 0.0f;
            }
        }
    });
}

// Render preparation: one DrawItem per drawable entity. This is what the Vulkan
// renderer will consume in Step 3.4 (one push-constant model matrix per item).
// The selected entity (if any) gets DRAWFLAG_SELECTED so the shader can highlight it.
inline void buildDrawList(Registry& r, std::vector<DrawItem>& out, Entity selected = NULL_ENTITY) {
    out.clear();
    r.each<MeshRenderer, Transform>([&out, selected](Entity e, MeshRenderer& m, Transform& t) {
        out.push_back(DrawItem{ modelMatrix(t), m.meshId,
                                m.materialId | (e == selected ? DRAWFLAG_SELECTED : 0u) });
    });
}
