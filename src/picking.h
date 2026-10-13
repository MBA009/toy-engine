#pragma once
#include <cfloat>
#include <cmath>
#include <utility>
#include <vector>
#include "components.h"
#include "ecs.h"
#include "math3d.h"
#include "mesh_data.h"

// ---------------------------------------------------------------------------
// Mouse picking: click -> ray in world space -> closest entity it hits.
// ---------------------------------------------------------------------------

struct Ray {
    Vec3 origin;
    Vec3 dir;     // normalized
};

// The cube mesh spans -0.5..0.5 on every axis, so its bounding box is this half-extent.
inline constexpr float CUBE_HALF_EXTENT = 0.5f;

// mouseX/mouseY: window coordinates in pixels, origin at the top-left (what GLFW reports).
// invViewProj: inverse of (projection * view), using the OpenGL-style projection (before the Vulkan fix-up).
inline Ray screenToRay(const Mat4& invViewProj, float mouseX, float mouseY, float winW, float winH) {
    // Pixels -> normalized device coordinates: x in [-1,1] left..right, y in [-1,1] bottom..top
    float ndcX = 2.0f * mouseX / winW - 1.0f;
    float ndcY = 1.0f - 2.0f * mouseY / winH;

    // Unproject one point on the near plane (z = -1) and one on the far plane (z = +1).
    // The perspective divide by w brings them back to ordinary 3D coordinates.
    Vec4 nearH = invViewProj * Vec4(ndcX, ndcY, -1.0f, 1.0f);
    Vec4 farH  = invViewProj * Vec4(ndcX, ndcY,  1.0f, 1.0f);
    Vec3 nearP = nearH.xyz() * (1.0f / nearH.w);
    Vec3 farP  = farH.xyz()  * (1.0f / farH.w);
    return { nearP, (farP - nearP).normalized() };
}

// Ray vs. the box `bounds` (given in a mesh's own space) as placed by a Transform (position, yaw
// rotation, uniform scale). Instead of building a rotated box in world space, move the RAY into the
// mesh's own space, where the box is axis-aligned and the slab test is trivial. Mesh space is a
// rotation and a uniform scale away from world space, so distances along the ray don't change:
// the t we find is the same t in world space.
inline bool rayHitsBox(const Ray& ray, const Transform& tr, const Aabb& bounds, float& tHit) {
    float c = std::cos(tr.yaw), s = std::sin(tr.yaw), k = tr.scale;

    // world -> local: undo the translation, undo the yaw, undo the scale (see modelMatrix())
    float ox = (ray.origin.x - tr.position.x) / k, oy = (ray.origin.y - tr.position.y) / k, oz = (ray.origin.z - tr.position.z) / k;
    float dx = ray.dir.x / k, dy = ray.dir.y / k, dz = ray.dir.z / k;
    float o[3] = { c * ox - s * oz, oy, s * ox + c * oz };
    float d[3] = { c * dx - s * dz, dy, s * dx + c * dz };
    const float lo[3] = { bounds.min.x, bounds.min.y, bounds.min.z };
    const float hi[3] = { bounds.max.x, bounds.max.y, bounds.max.z };

    // Slab test: the ray is inside the box where it is inside all three pairs of parallel planes
    float tMin = -FLT_MAX, tMax = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(d[axis]) < 1e-8f) {                        // parallel to this pair of planes
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) return false;
        } else {
            float t1 = (lo[axis] - o[axis]) / d[axis];
            float t2 = (hi[axis] - o[axis]) / d[axis];
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tMin) tMin = t1;
            if (t2 < tMax) tMax = t2;
            if (tMin > tMax) return false;
        }
    }
    if (tMax < 0.0f) return false;                               // box is entirely behind the camera
    tHit = tMin >= 0.0f ? tMin : tMax;                           // tMin < 0: the ray starts inside the box
    return true;
}

// The unit cube the built-in cube mesh occupies
inline bool rayHitsCube(const Ray& ray, const Transform& tr, float& tHit) {
    return rayHitsBox(ray, tr, Aabb{}, tHit);
}

// The closest entity hit by the ray, or NULL_ENTITY. Each entity is tested against the bounding
// box of its mesh (meshBounds[meshId]); an unknown mesh id falls back to the unit cube.
// Tests every drawable entity (O(n)), which is fine for a click even at 100,000 cubes.
inline Entity pickEntity(Registry& reg, const Ray& ray, const std::vector<Aabb>& meshBounds, float* outDistance = nullptr) {
    static const Aabb unitCube{};
    Entity best = NULL_ENTITY;
    float bestT = FLT_MAX;
    reg.each<MeshRenderer, Transform>([&](Entity e, MeshRenderer& mr, Transform& tr) {
        const Aabb& box = mr.meshId < meshBounds.size() ? meshBounds[mr.meshId] : unitCube;
        float t;
        if (rayHitsBox(ray, tr, box, t) && t < bestT) { bestT = t; best = e; }
    });
    if (outDistance && best != NULL_ENTITY) *outDistance = bestT;
    return best;
}

// Everything is a unit cube
inline Entity pickEntity(Registry& reg, const Ray& ray, float* outDistance = nullptr) {
    static const std::vector<Aabb> none;
    return pickEntity(reg, ray, none, outDistance);
}
