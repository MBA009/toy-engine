#pragma once
#include <cmath>
#include <string>
#include "ai_systems.h"

// ---------------------------------------------------------------------------
// The demo level (Step 8.4): a zig-zag course on the built-in floor (x -3..3, z 1..-7, y -0.5).
//
//        far (z -7)
//   +---------------------------+
//   |  tower      guard 3   GOAL|   top pocket, guarded
//   |=========== wall B ====    |   wall B has its gap on the LEFT
//   |   guard 4 / guard 2       |   middle corridor
//   |    ====== wall A ========|   wall A has its gap on the RIGHT
//   |   pyramid    guard 1      |   lower corridor
//   | START             tower   |
//   +---------------------------+
//        near (z 1)
//
// Everything is ordinary entities. Walls, pillars and tower/pyramid bases are navmesh obstacles; the stacked top
// layers, gems and beacon are just decoration. Only the built-in cube mesh (id 0) is used, so it needs no assets.
// ---------------------------------------------------------------------------

struct DemoLevelInfo {
    Entity player = NULL_ENTITY;
    int guards = 0;
    Vec3 start{}, goal{};
};

inline DemoLevelInfo buildDemoLevel(Registry& r) {
    const float FLOOR = -0.5f;
    DemoLevelInfo info;
    int serial = 0;

    auto cube = [&](const char* name, float x, float z, float size, float lift, bool obstacle) {
        Entity e = r.create();
        Transform t;
        t.scale = size;
        t.position = { x, FLOOR + lift + size * 0.5f, z };
        r.add<Transform>(e, t);
        r.add<MeshRenderer>(e, MeshRenderer{ 0, 0 });
        r.add<Name>(e, Name{ std::string(name) + " " + std::to_string(++serial) });
        if (obstacle) r.add<NavSource>(e, NavSource{ NAV_OBSTACLE });
        return e;
    };

    // Two long walls, each a row of cubes, with the gaps at opposite ends to force a zig-zag
    for (int i = 0; i < 10; i++) cube("Wall A", -2.8f + 0.4f * (float)i, -2.0f, 0.4f, 0.0f, true);   // spans x -3.0..1.0, gap on the right
    for (int i = 0; i < 10; i++) cube("Wall B", -0.7f + 0.4f * (float)i, -4.4f, 0.4f, 0.0f, true);   // spans x -0.9..3.1, gap on the left
    // Pillars that break up the open corridors
    cube("Pillar", 1.7f, -0.9f, 0.45f, 0.0f, true);
    cube("Pillar", -1.3f, -3.3f, 0.45f, 0.0f, true);
    cube("Pillar", 0.6f, -5.5f, 0.45f, 0.0f, true);
    // A stepped pyramid: only the base blocks agents, the layers above are for looks
    cube("Pyramid base", -0.3f, -0.9f, 0.7f, 0.0f, true);
    cube("Pyramid mid", -0.3f, -0.9f, 0.48f, 0.7f, false);
    cube("Pyramid top", -0.3f, -0.9f, 0.28f, 1.18f, false);
    // Corner towers
    cube("Tower base", 2.6f, 0.45f, 0.5f, 0.0f, true);
    cube("Tower mid", 2.6f, 0.45f, 0.36f, 0.5f, false);
    cube("Tower top", 2.6f, 0.45f, 0.22f, 0.86f, false);
    cube("Tower base", -2.65f, -6.55f, 0.5f, 0.0f, true);
    cube("Tower mid", -2.65f, -6.55f, 0.36f, 0.5f, false);
    cube("Tower top", -2.65f, -6.55f, 0.22f, 0.86f, false);

    // Floating gems: small cubes that spin and bob above the routes
    const float gems[][2] = { { -1.6f, -1.0f }, { 2.2f, -2.0f }, { 0.2f, -3.2f }, { -2.0f, -3.5f }, { 1.4f, -4.8f }, { -1.6f, -5.8f } };
    for (const auto& g : gems) {
        Entity e = cube("Gem", g[0], g[1], 0.1f, 0.35f, false);
        r.add<Spin>(e, Spin{ 1.4f });
        r.add<Bob>(e, Bob{ 0.06f, 2.0f, r.get<Transform>(e).position.y, g[0] * 3.0f });
    }

    // Start and goal pads (invisible entities; the editor draws a ring on the floor)
    info.start = { -2.3f, FLOOR, 0.15f };
    info.goal = { 2.3f, FLOOR, -6.2f };
    { Entity e = r.create(); r.add<Transform>(e, Transform{ info.start, 0, 1 }); r.add<Name>(e, Name{ "Start" }); r.add<StartPad>(e, StartPad{ 0.4f }); }
    { Entity e = r.create(); r.add<Transform>(e, Transform{ info.goal, 0, 1 }); r.add<Name>(e, Name{ "Goal" }); r.add<Goal>(e, Goal{ 0.4f }); }
    // A beacon over the goal: a column of spinning cubes you can spot from anywhere
    for (int i = 0; i < 4; i++) {
        const float size = 0.2f - 0.035f * (float)i;
        Entity e = cube("Beacon", info.goal.x, info.goal.z, size, 0.3f + 0.22f * (float)i, false);
        r.add<Spin>(e, Spin{ (i % 2 ? -2.0f : 2.0f) });
        r.add<Bob>(e, Bob{ 0.04f, 1.5f, r.get<Transform>(e).position.y, 0.5f * (float)i });
    }

    // Guards: each patrols a stretch of the course
    auto guard = [&](std::initializer_list<Vec3> route) {
        Entity e = r.create();
        Transform t;
        t.scale = 0.15f;
        AIAgent ag;
        ag.footOffset = 0.5f * t.scale;                       // the built-in cube is 1 high with its origin at the centre
        const Vec3 first = *route.begin();
        t.position = { first.x, first.y + ag.footOffset, first.z };
        r.add<Transform>(e, t);
        r.add<MeshRenderer>(e, MeshRenderer{ 0, 0 });
        r.add<Name>(e, Name{ "Guard " + std::to_string(++info.guards) });
        r.add<AIAgent>(e, ag);
        AIBrain br;
        br.patrol.assign(route.begin(), route.end());
        br.detectRange = 2.0f;                                 // a bit sharper than the default so the course is tense
        r.add<AIBrain>(e, br);
        r.add<Health>(e, Health{});
    };
    guard({ { -2.3f, FLOOR, -1.0f }, { 2.4f, FLOOR, -1.0f } });                                    // lower corridor
    guard({ { 2.5f, FLOOR, -3.2f }, { -2.5f, FLOOR, -3.2f } });                                    // middle corridor
    guard({ { -2.0f, FLOOR, -5.8f }, { -1.0f, FLOOR, -6.4f }, { 1.2f, FLOOR, -6.5f } });            // top pocket, near the goal
    guard({ { -2.4f, FLOOR, -4.0f }, { -2.4f, FLOOR, -5.4f } });                                   // watches the gap in wall B

    // The player
    {
        Entity e = r.create();
        Transform t;
        t.scale = 0.2f;
        AIAgent ag;
        ag.footOffset = 0.5f * t.scale;
        ag.moveSpeed = 0.8f;                                   // faster than the guards: it can outrun them
        t.position = { info.start.x, info.start.y + ag.footOffset, info.start.z };
        r.add<Transform>(e, t);
        r.add<MeshRenderer>(e, MeshRenderer{ 0, 0 });
        r.add<Name>(e, Name{ "Player" });
        r.add<AIAgent>(e, ag);
        r.add<Health>(e, Health{});
        r.add<Player>(e, Player{});
        info.player = e;
    }
    return info;
}
