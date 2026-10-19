#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
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

// Hit points (Step 7.4). Agents with a brain lose health while the player is touching them and recover it
// slowly otherwise; low health is what makes them run away.
struct Health {
    float current = 100.0f;
    float max = 100.0f;
    float fraction() const { return max > 0.0f ? current / max : 0.0f; }
};

// Tag: the entity the AI reacts to. Move it with the gizmo, or give it an AIAgent and use "Send to...".
struct Player {};

// Level markers (Step 8.4). Put them on an entity with a Transform (no mesh needed); the ring is drawn on the floor.
// Reaching the Goal with the Player wins the run; the player returns to the StartPad on a restart.
struct Goal { float radius = 0.4f; };
struct StartPad { float radius = 0.4f; };

// An agent's decision-making state (Step 7.4). The first group is tuning (edited live in the Inspector); the
// second is what the behavior tree remembers between frames. The tree itself is shared by all agents: it holds
// no per-agent data, which lives here.
struct AIBrain {
    // --- tuning ---
    float detectRange = 1.5f;        // sees the player inside this distance (ground plane)
    float contactRange = 0.25f;      // "touching": stops chasing, and the player hurts it
    float fleeBelow = 0.30f;         // runs away when health falls below this fraction of max...
    float recoverAt = 0.70f;         // ...and keeps running until it has recovered to this
    float contactDamage = 25.0f;     // health lost per second while touching the player
    float attackDamage = 20.0f;      // health the PLAYER loses per second while this agent touches it (if the player has Health)
    float regenPerSecond = 10.0f;    // health regained per second otherwise
    float patrolWait = 0.5f;         // pause at each patrol point, in seconds
    std::vector<Vec3> patrol;        // feet positions to walk between, in a loop

    // --- state ---
    enum Owner : uint8_t { OWNER_NONE, OWNER_PATROL, OWNER_CHASE, OWNER_FLEE, OWNER_IDLE };
    uint8_t pathOwner = OWNER_NONE;  // which behaviour set the agent's current path (so others know to replace it)
    uint32_t patrolIndex = 0;        // patrol point being walked to
    bool patrolStarted = false;      // a path to patrol[patrolIndex] has been issued
    float waitTimer = 0.0f;
    float repathTimer = 0.0f;
    bool chasing = false;            // currently has the player in sight (sight range is wider once it has seen them)
    bool fleeing = false;            // below fleeBelow and not yet recovered to recoverAt
    int fleeTarget = -1;             // patrol point being fled to
    std::vector<int8_t> trace;       // last tick, per tree node: -1 not visited, else the BTStatus returned (for Step 7.5)
    int lastLeaf = -1;               // id of the leaf that decided the last tick
};

// How an entity takes part in navmesh baking (Step 7.1). No NavSource = ignored by the navmesh.
enum NavMode : uint8_t {
    NAV_OBSTACLE = 1,   // a solid object: blocks agents, and the floor under it is not walkable
    NAV_GEOMETRY = 2,   // level geometry: its surfaces are used as they are (ramps, platforms, walls)
};
struct NavSource {
    uint8_t mode = NAV_OBSTACLE;
};

// A character that walks along a path (Step 7.3). The path is a list of corners on the navmesh (positions of
// the agent's FEET); PathFollowSystem moves the entity toward them one at a time. Later steps add the
// behaviour that decides where to go; this component only knows how to get there.
struct AIAgent {
    float moveSpeed = 0.6f;       // world units per second
    float turnSpeed = 10.0f;      // radians per second (how fast it swings to face where it is going)
    float footOffset = 0.0f;      // distance from the entity's origin down to its feet: origin = feet + this
    std::vector<Vec3> path;       // corners to walk through, in order
    uint32_t waypoint = 0;        // index of the corner being walked to; == path.size() means arrived (idle)

    bool walking() const { return waypoint < path.size(); }
    void stop() { path.clear(); waypoint = 0; }
    void setPath(std::vector<Vec3> corners) { path = std::move(corners); waypoint = 0; }
};

// One entry per drawable object, ready to feed to the renderer (push constants / instance data)
struct DrawItem {
    Mat4     model;
    uint32_t meshId;
    uint32_t materialId;
};

// Set in DrawItem::materialId for the entity currently selected in the editor
inline constexpr uint32_t DRAWFLAG_SELECTED = 1u << 31;

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

// The inverse of modelMatrix(): read position, yaw and uniform scale back out of a model matrix
// (what a gizmo hands back after a drag). Column 0 of modelMatrix is (cos*k, 0, -sin*k), so its
// length is the scale and atan2 recovers the full -180..180 degree yaw (Euler decomposition
// routines flip angles beyond +-90 degrees; this doesn't).
inline Transform transformFromMatrix(const Mat4& m, const Transform& fallback = Transform{}) {
    Transform t = fallback;
    t.position = { m.m[3][0], m.m[3][1], m.m[3][2] };
    float cx = m.m[0][0], cz = m.m[0][2];
    float k = std::sqrt(cx * cx + cz * cz);
    if (k > 1e-6f) {
        t.scale = k;
        t.yaw = std::atan2(-cz, cx);
    }
    return t;
}
