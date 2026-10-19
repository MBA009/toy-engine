// Headless check of the demo level (Step 8.4): real navmesh built from the level's own entities, real AI systems.
#include <cstdio>
#include "demo_level.h"
#include "navmesh.h"
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)

// Same job as App::gatherNavInput for axis-aligned cubes (no yaw in this level)
static void addCube(NavInput& in, const Transform& t, bool solid) {
    float p[8][3];
    for (int i = 0; i < 8; i++) {
        p[i][0] = t.position.x + ((i & 1) ? .5f : -.5f) * t.scale;
        p[i][1] = t.position.y + ((i & 2) ? .5f : -.5f) * t.scale;
        p[i][2] = t.position.z + ((i & 4) ? .5f : -.5f) * t.scale;
    }
    const int q[6][4] = {{0,1,3,2},{4,6,7,5},{0,4,5,1},{2,3,7,6},{0,2,6,4},{1,5,7,3}};
    for (auto& f : q) { in.addTriangle(p[f[0]], p[f[1]], p[f[2]]); in.addTriangle(p[f[0]], p[f[2]], p[f[3]]); }
    if (solid) in.addSolid(&p[0][0], 8);
}
static void buildNav(Registry& r, NavMesh& nm) {
    NavInput in; const float Y = -0.5f;
    { const float a[3]={-3,Y,1}, b[3]={3,Y,1}, c[3]={3,Y,-7}, d[3]={-3,Y,-7}; in.addTriangle(a,b,c); in.addTriangle(a,c,d); }
    r.each<NavSource, Transform>([&](Entity, NavSource&, Transform& t) { addCube(in, t, true); });
    NavParams params; std::string err;
    if (!nm.build(in, params, err)) { std::printf("navmesh build failed: %s\n", err.c_str()); failures++; }
}
static PathFn makePathFn(NavMesh& nm) {
    return [&nm](const Vec3& from, const Vec3& to, std::vector<Vec3>& out) {
        const float a[3] = {from.x,from.y,from.z}, b[3] = {to.x,to.y,to.z};
        NavPath p; std::string e;
        if (!nm.findPath(a, b, p, e) || p.count() == 0) return false;
        out.clear(); for (size_t i = 0; i < p.count(); i++) out.push_back({p.point(i)[0], p.point(i)[1], p.point(i)[2]});
        return true; };
}
static void tick(Registry& r, GameState& g, const BehaviorTree& tree, const PathFn& f, float dt) {
    const float move = g.phase == GameState::Playing ? dt : 0.0f;
    healthSystem(r, move); behaviorSystem(r, move, tree, f); pathFollowSystem(r, move); updateGame(r, g, dt);
}

int main() {
    BehaviorTree tree = makeCombatTree();
    Registry r;
    DemoLevelInfo info = buildDemoLevel(r);
    CHECK(info.guards == 4 && info.player != NULL_ENTITY);
    CHECK(findGoal(r) != NULL_ENTITY && findStart(r) != NULL_ENTITY);
    NavMesh nm; buildNav(r, nm);
    CHECK(nm.valid());
    PathFn f = makePathFn(nm);

    // 1. The course is completable and really zig-zags: the start->goal route is much longer than the straight line
    { NavPath p; std::string e; const float a[3]={info.start.x,info.start.y,info.start.z}, b[3]={info.goal.x,info.goal.y,info.goal.z};
      CHECK(nm.findPath(a, b, p, e)); CHECK(!p.partial);
      const float straight = std::sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[2]-b[2])*(a[2]-b[2]));
      std::printf("start->goal: %zu waypoints, length %.2f (straight line %.2f)\n", p.count(), p.length, straight);
      CHECK(p.length > straight * 1.3f);
      // the route passes through both gaps: right of wall A (x > 1.0 at z about -2) and left of wall B (x < -0.9 at z about -4.4)
      bool gapA = false, gapB = false;
      for (size_t i = 0; i + 1 < p.count(); i++) {
          const float *u = p.point(i), *v = p.point(i+1);
          if ((u[2] - -2.0f) * (v[2] - -2.0f) <= 0) { float t = (u[2]==v[2]) ? 0 : (-2.0f-u[2])/(v[2]-u[2]); gapA |= (u[0]+(v[0]-u[0])*t) > 1.0f; }
          if ((u[2] - -4.4f) * (v[2] - -4.4f) <= 0) { float t = (u[2]==v[2]) ? 0 : (-4.4f-u[2])/(v[2]-u[2]); gapB |= (u[0]+(v[0]-u[0])*t) < -0.9f; }
      }
      CHECK(gapA && gapB); }

    // 2. Every patrol point is on walkable ground and every leg of every route has a full path
    r.each<AIBrain>([&](Entity, AIBrain& b) {
        for (size_t i = 0; i < b.patrol.size(); i++) {
            const Vec3& a = b.patrol[i]; const Vec3& n = b.patrol[(i+1) % b.patrol.size()];
            const float fa[3]={a.x,a.y,a.z}, fb[3]={n.x,n.y,n.z}; float snap[3]; NavPath p; std::string e;
            CHECK(nm.nearestPoint(fa, snap) && std::fabs(snap[0]-a.x) < 0.1f && std::fabs(snap[2]-a.z) < 0.1f);
            CHECK(nm.findPath(fa, fb, p, e) && !p.partial);
        }
    });

    // 3. Idle player: guards patrol for 30 s, never leave the floor or enter a wall, and nobody wins or loses
    GameState g;
    for (int i = 0; i < 600; i++) {
        tick(r, g, tree, f, 0.05f);
        r.each<AIBrain, Transform>([&](Entity, AIBrain&, Transform& t) {
            CHECK(t.position.x > -3 && t.position.x < 3 && t.position.z > -7 && t.position.z < 1);
            r.each<NavSource, Transform>([&](Entity, NavSource&, Transform& w) {
                CHECK(!(std::fabs(t.position.x - w.position.x) < w.scale*0.5f && std::fabs(t.position.z - w.position.z) < w.scale*0.5f));
            });
        });
    }
    CHECK(g.active && g.phase == GameState::Playing && g.time > 29.0f);

    // 4. Win: with the guards out of the way the player walks to the goal and the run ends
    { Registry r2; DemoLevelInfo i2 = buildDemoLevel(r2); NavMesh nm2; buildNav(r2, nm2); PathFn f2 = makePathFn(nm2);
      std::vector<Entity> gone; r2.each<AIBrain>([&](Entity e, AIBrain&) { gone.push_back(e); }); for (Entity e : gone) r2.destroy(e);
      GameState g2; std::vector<Vec3> c;
      CHECK(f2(i2.start, i2.goal, c)); r2.get<AIAgent>(i2.player).setPath(c);
      for (int i = 0; i < 1200 && g2.phase == GameState::Playing; i++) tick(r2, g2, tree, f2, 0.05f);
      CHECK(g2.phase == GameState::Won); std::printf("won after %.1f s\n", g2.time);
      CHECK(!r2.get<AIAgent>(i2.player).walking()); }

    // 5. Lose: a player caught by the whole guard team is worn down to zero
    { Registry r3; DemoLevelInfo i3 = buildDemoLevel(r3); NavMesh nm3; buildNav(r3, nm3); PathFn f3 = makePathFn(nm3);
      GameState g3; const Vec3 spot = r3.get<Transform>(i3.player).position;
      for (int i = 0; i < 400 && g3.phase == GameState::Playing; i++) {
          r3.each<AIBrain, Transform>([&](Entity, AIBrain&, Transform& t) { t.position.x = spot.x; t.position.z = spot.z; });   // the team closes in
          tick(r3, g3, tree, f3, 0.05f);
      }
      CHECK(g3.phase == GameState::Lost); std::printf("lost after %.1f s\n", g3.time);
      // restart puts everything back
      resetRun(r3, g3);
      CHECK(g3.phase == GameState::Playing && g3.time == 0.0f);
      CHECK(r3.get<Health>(i3.player).fraction() == 1.0f);
      const Vec3 sp = r3.get<Transform>(findStart(r3)).position;
      CHECK(bt::dist2D(r3.get<Transform>(i3.player).position, sp) < 0.001f);
      for (int i = 0; i < 100; i++) tick(r3, g3, tree, f3, 0.05f);
      CHECK(g3.phase == GameState::Playing); }

    // 5b. Informational: a player that simply walks the shortest route with every guard active
    { Registry r5; DemoLevelInfo i5 = buildDemoLevel(r5); NavMesh nm5; buildNav(r5, nm5); PathFn f5 = makePathFn(nm5);
      GameState g5; std::vector<Vec3> c; f5(i5.start, i5.goal, c); r5.get<AIAgent>(i5.player).setPath(c);
      for (int i = 0; i < 1200 && g5.phase == GameState::Playing; i++) tick(r5, g5, tree, f5, 0.05f);
      std::printf("naive run: %s after %.1f s, player health %.0f\n", g5.phase == GameState::Won ? "WON" : (g5.phase == GameState::Lost ? "LOST" : "unfinished"), g5.time, r5.get<Health>(i5.player).current); }

    // 6. A sandbox scene (no goal) is not a game
    { Registry r4; GameState g4; Entity e = r4.create(); r4.add<Transform>(e, Transform{}); r4.add<Player>(e, Player{});
      updateGame(r4, g4, 0.1f); CHECK(!g4.active && g4.phase == GameState::Playing); }

    std::printf(failures ? "FAILURES: %d\n" : "demo level ok\n", failures);
    return failures ? 1 : 0;
}
