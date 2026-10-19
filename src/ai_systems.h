#pragma once
#include <algorithm>
#include "bt.h"
#include "ecs_systems.h"

// ---------------------------------------------------------------------------
// AI systems (Step 7.4). Per frame the order is: healthSystem -> behaviorSystem -> pathFollowSystem.
// ---------------------------------------------------------------------------

// The first entity tagged Player that has a position, or NULL_ENTITY
inline Entity findPlayer(Registry& r) {
    Entity found = NULL_ENTITY;
    r.each<Player, Transform>([&](Entity e, Player&, Transform&) { if (found == NULL_ENTITY) found = e; });
    return found;
}

inline constexpr float PLAYER_REGEN = 5.0f;     // health per second the player regains when no guard is touching it

// Touching the player hurts the agent; being left alone heals it. The player (if it has Health) is hurt by every
// agent touching it that is not itself running away. (Agents without a brain are not affected.)
inline void healthSystem(Registry& r, float dt) {
    if (dt <= 0.0f) return;
    const Entity player = findPlayer(r);
    Vec3 playerPos{ 0, 0, 0 };
    if (player != NULL_ENTITY) playerPos = r.get<Transform>(player).position;
    r.each<AIBrain, Health, Transform>([&](Entity, AIBrain& b, Health& h, Transform& t) {
        const bool touching = player != NULL_ENTITY && bt::dist2D(t.position, playerPos) <= b.contactRange;
        h.current += (touching ? -b.contactDamage : b.regenPerSecond) * dt;
        h.current = std::clamp(h.current, 0.0f, h.max);
    });
    if (player != NULL_ENTITY && r.has<Health>(player)) {
        float damage = 0.0f;
        r.each<AIBrain, Transform>([&](Entity e, AIBrain& b, Transform& t) {
            if (e != player && !b.fleeing && bt::dist2D(t.position, playerPos) <= b.contactRange) damage += b.attackDamage;
        });
        Health& ph = r.get<Health>(player);
        ph.current = std::clamp(ph.current + (damage > 0.0f ? -damage : PLAYER_REGEN) * dt, 0.0f, ph.max);
    }
}

// Runs every agent's behavior tree, which decides what path it should be on
inline void behaviorSystem(Registry& r, float dt, const BehaviorTree& tree, const PathFn& findPath) {
    if (dt <= 0.0f) return;
    const Entity player = findPlayer(r);
    r.each<AIBrain, AIAgent, Transform>([&](Entity e, AIBrain& b, AIAgent& a, Transform& t) {
        BTContext ctx{ r, e, player == e ? NULL_ENTITY : player, dt, findPath, a, t, b, r.pool<Health>().find(e) };
        b.trace.assign((size_t)tree.nodeCount(), (int8_t)-1);
        b.lastLeaf = -1;
        tickNode(tree.root(), ctx);
    });
}

// ---------------------------------------------------------------------------
// The win/lose loop (Step 8.4): walk the player from the StartPad to the Goal without running out of health.
// ---------------------------------------------------------------------------
struct GameState {
    enum Phase : uint8_t { Playing, Won, Lost };
    Phase phase = Playing;
    float time = 0.0f;       // seconds spent in this run
    bool active = false;     // there is a player and a goal, so this is a game and not just a sandbox
};

inline Entity findGoal(Registry& r) {
    Entity found = NULL_ENTITY;
    r.each<Goal, Transform>([&](Entity e, Goal&, Transform&) { if (found == NULL_ENTITY) found = e; });
    return found;
}
inline Entity findStart(Registry& r) {
    Entity found = NULL_ENTITY;
    r.each<StartPad, Transform>([&](Entity e, StartPad&, Transform&) { if (found == NULL_ENTITY) found = e; });
    return found;
}

inline void updateGame(Registry& r, GameState& g, float dt) {
    const Entity player = findPlayer(r), goal = findGoal(r);
    g.active = player != NULL_ENTITY && goal != NULL_ENTITY;
    if (!g.active || g.phase != GameState::Playing) return;
    if (dt > 0.0f) g.time += dt;
    const Vec3& p = r.get<Transform>(player).position;
    const Vec3& gp = r.get<Transform>(goal).position;
    if (bt::dist2D(p, gp) <= r.get<Goal>(goal).radius) {
        g.phase = GameState::Won;
        if (r.has<AIAgent>(player)) r.get<AIAgent>(player).stop();
    } else if (r.has<Health>(player) && r.get<Health>(player).current <= 0.0f) {
        g.phase = GameState::Lost;
        if (r.has<AIAgent>(player)) r.get<AIAgent>(player).stop();
    }
}

// Puts everything back for another attempt: player on the start pad, guards at their first patrol point, full health
inline void resetRun(Registry& r, GameState& g) {
    const Entity player = findPlayer(r), start = findStart(r);
    if (player != NULL_ENTITY && start != NULL_ENTITY) {
        Transform& t = r.get<Transform>(player);
        const Vec3& sp = r.get<Transform>(start).position;
        const float foot = r.has<AIAgent>(player) ? r.get<AIAgent>(player).footOffset : 0.0f;
        t.position = { sp.x, sp.y + foot, sp.z };
    }
    if (player != NULL_ENTITY) {
        if (r.has<AIAgent>(player)) r.get<AIAgent>(player).stop();
        if (r.has<Health>(player)) r.get<Health>(player).current = r.get<Health>(player).max;
    }
    r.each<AIBrain, AIAgent, Transform>([&](Entity e, AIBrain& b, AIAgent& a, Transform& t) {
        a.stop();
        if (!b.patrol.empty()) t.position = { b.patrol[0].x, b.patrol[0].y + a.footOffset, b.patrol[0].z };
        b.pathOwner = AIBrain::OWNER_NONE; b.patrolIndex = 0; b.patrolStarted = false; b.waitTimer = 0.0f;
        b.repathTimer = 0.0f; b.chasing = false; b.fleeing = false; b.fleeTarget = -1;
        if (r.has<Health>(e)) r.get<Health>(e).current = r.get<Health>(e).max;
    });
    g = GameState{};
}
