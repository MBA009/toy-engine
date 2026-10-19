// Headless test for navmesh.h/.cpp (Recast + Detour).
//   g++ -std=c++17 -I. -Iexternal/recastnavigation/Recast/Include -Iexternal/recastnavigation/Detour/Include navmesh_test.cpp navmesh.cpp
//       external/recastnavigation/Recast/Source/*.cpp external/recastnavigation/Detour/Source/*.cpp -o navmesh_test
#include <cmath>
#include <cstdio>
#include "navmesh.h"
#include "DetourNavMesh.h"
#include "DetourNavMeshQuery.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

// A horizontal rectangle at height y, wound counter-clockwise seen from above (normal +y)
static void addFloor(NavInput& in, float x0, float z0, float x1, float z1, float y) {
    float a[3] = { x0, y, z1 }, b[3] = { x1, y, z1 }, c[3] = { x1, y, z0 }, d[3] = { x0, y, z0 };
    in.addTriangle(a, b, c);
    in.addTriangle(a, c, d);
}

// An axis-aligned box with outward-facing counter-clockwise faces
static void addBox(NavInput& in, float x0, float y0, float z0, float x1, float y1, float z1) {
    const float P[8][3] = { {x0,y0,z0},{x1,y0,z0},{x1,y1,z0},{x0,y1,z0},{x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1} };
    const int Q[6][4] = { {4,5,6,7}, {1,0,3,2}, {5,1,2,6}, {0,4,7,3}, {7,6,2,3}, {0,1,5,4} };   // +z, -z, +x, -x, +y, -y
    for (auto& q : Q) { in.addTriangle(P[q[0]], P[q[1]], P[q[2]]); in.addTriangle(P[q[0]], P[q[2]], P[q[3]]); }
}

// Area of the navmesh's debug triangles whose height is below maxY (projected onto the ground plane)
static double areaBelow(const NavMesh& nm, float maxY) {
    double a = 0;
    for (const NavDebugTri& t : nm.debugTriangles()) {
        if (t.a[1] > maxY || t.b[1] > maxY || t.c[1] > maxY) continue;
        a += 0.5 * std::fabs((t.b[0] - t.a[0]) * (t.c[2] - t.a[2]) - (t.c[0] - t.a[0]) * (t.b[2] - t.a[2]));
    }
    return a;
}

// Is there navmesh within 3 cm (horizontally) of this point? (findNearestPoly returns the nearest polygon
// in the search box even if it is far away, so the returned position has to be checked as well)
static bool hasPolyAt(const NavMesh& nm, float x, float y, float z) {
    dtNavMeshQuery* q = dtAllocNavMeshQuery();
    q->init(nm.detour(), 256);
    dtQueryFilter filter;
    float pos[3] = { x, y, z }, ext[3] = { 0.2f, 0.5f, 0.2f }, nearest[3] = { 0, 0, 0 };
    dtPolyRef ref = 0;
    q->findNearestPoly(pos, ext, &filter, &ref, nearest);
    dtFreeNavMeshQuery(q);
    return ref != 0 && std::fabs(nearest[0] - x) < 0.03f && std::fabs(nearest[2] - z) < 0.03f;
}

int main() {
    NavParams params;     // the engine's defaults: cell 0.05, agent radius 0.06 (rounds up to 2 cells = 0.1)

    std::printf("moveAlong\n");
    {
        NavInput in;
        addFloor(in, -5, -5, 5, 5, 0.0f);
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));
        const float from[3] = { 0, 0, 0 };
        float out[3];
        const float inside[3] = { 1.0f, 0, 0.5f };
        CHECK(nm.moveAlong(from, inside, out) && std::fabs(out[0] - 1.0f) < 0.01f && std::fabs(out[2] - 0.5f) < 0.01f);
        const float past[3] = { 20.0f, 0, 0.0f };                     // far beyond the edge: stops at the boundary
        CHECK(nm.moveAlong(from, past, out) && out[0] > 4.5f && out[0] < 5.0f && std::fabs(out[2]) < 0.05f);
        const float edge[3] = { 4.8f, 0, 0 }, diag[3] = { 9.0f, 0, 1.0f };   // pushing into the edge slides along it
        CHECK(nm.moveAlong(edge, diag, out) && out[0] < 5.0f && out[2] > 0.1f);
        const float nowhere[3] = { 50, 0, 50 };
        CHECK(!nm.moveAlong(nowhere, from, out));
    }

    std::printf("flat floor\n");
    {
        NavInput in;
        addFloor(in, -5, -5, 5, 5, 0.0f);
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));
        CHECK(nm.valid() && nm.stats().polygons > 0 && !nm.debugTriangles().empty() && !nm.debugEdges().empty());
        double area = areaBelow(nm, 0.5f), expect = 9.8 * 9.8;       // 10x10 pulled in 0.1 on each side
        std::printf("  walkable area %.2f (expected about %.2f), %d polygons, %.1f ms\n", area, expect, nm.stats().polygons, nm.stats().buildMs);
        CHECK(std::fabs(area - expect) < expect * 0.03);
        CHECK(hasPolyAt(nm, 0, 0, 0) && hasPolyAt(nm, 4.5f, 0, 4.5f));
        CHECK(!hasPolyAt(nm, 5.5f, 0, 0));                           // off the edge
        CHECK(!hasPolyAt(nm, 4.95f, 0, 0));                          // inside the agent-radius margin
        bool anyBorder = false, anyInner = false;
        for (auto& e : nm.debugEdges()) (e.border ? anyBorder : anyInner) = true;
        CHECK(anyBorder);
        (void)anyInner;
    }

    std::printf("a box in the middle blocks the floor\n");
    {
        NavInput in;
        addFloor(in, -5, -5, 5, 5, 0.0f);
        addBox(in, -1, 0, -1, 1, 1, 1);
        const float corners[8][3] = { {-1,0,-1},{1,0,-1},{1,1,-1},{-1,1,-1},{-1,0,1},{1,0,1},{1,1,1},{-1,1,1} };
        in.addSolid(&corners[0][0], 8);
        CHECK(in.solids.size() == 1 && in.solids[0].hullXZ.size() == 8);        // a square footprint: 4 corners
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));
        for (auto& tr : nm.debugTriangles()) CHECK(tr.a[1] < 0.5f);          // no roof island on top of the box
        double area = areaBelow(nm, 0.5f), expect = 9.8 * 9.8 - 2.2 * 2.2;   // box footprint grown by the agent radius
        std::printf("  walkable floor area %.2f (expected about %.2f)\n", area, expect);
        CHECK(std::fabs(area - expect) < expect * 0.04);
        CHECK(hasPolyAt(nm, 3, 0, 3));
        CHECK(!hasPolyAt(nm, 0, 0, 0));                              // inside the box
        CHECK(!hasPolyAt(nm, 1.05f, 0, 0));                          // against the box wall: within the margin
        CHECK(hasPolyAt(nm, 1.3f, 0, 0));
    }

    std::printf("solid hull of a rotated and a degenerate object\n");
    {
        NavInput in;
        const float diamond[8][3] = { {0,0,-1},{1,0,0},{0,0,1},{-1,0,0}, {0,1,-1},{1,1,0},{0,1,1},{-1,1,0} };
        in.addSolid(&diamond[0][0], 8);
        CHECK(in.solids.size() == 1 && in.solids[0].hullXZ.size() == 8);
        const float line[3][3] = { {0,0,0},{1,0,0},{2,0,0} };
        in.addSolid(&line[0][0], 3);
        CHECK(in.solids.size() == 1);                                        // collinear: ignored
    }

    std::printf("a slope steeper than the limit is not walkable\n");
    {
        NavInput in;
        addFloor(in, -5, -5, 5, 5, 0.0f);
        // a ramp rising 3 units over 1 unit of run (about 71 degrees), north of the floor
        float a[3] = { -1, 0, -5 }, b[3] = { 1, 0, -5 }, c[3] = { 1, 3, -6 }, d[3] = { -1, 3, -6 };
        in.addTriangle(a, b, c); in.addTriangle(a, c, d);
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));
        for (auto& t : nm.debugTriangles()) CHECK(t.a[1] < 0.5f);    // nothing on the ramp
    }

    std::printf("failures give a reason and leave the previous navmesh alone\n");
    {
        NavInput in;
        addFloor(in, -5, -5, 5, 5, 0.0f);
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));
        const int polys = nm.stats().polygons;

        NavInput empty;
        CHECK(!nm.build(empty, params, err) && !err.empty()); std::printf("  -> %s\n", err.c_str());
        CHECK(nm.valid() && nm.stats().polygons == polys);

        NavParams huge = params; huge.agentRadius = 20.0f;
        CHECK(!nm.build(in, huge, err) && !err.empty()); std::printf("  -> %s\n", err.c_str());
        CHECK(nm.valid() && nm.stats().polygons == polys);

        NavParams tiny = params; tiny.cellSize = 0.00001f;
        CHECK(!nm.build(in, tiny, err) && !err.empty()); std::printf("  -> %s\n", err.c_str());
        CHECK(nm.valid() && nm.stats().polygons == polys);

        NavParams bad = params; bad.cellSize = 0.0f;
        CHECK(!nm.build(in, bad, err) && !err.empty());

        nm.clear();
        CHECK(!nm.valid() && nm.debugTriangles().empty());
    }

    std::printf("path around a wall\n");
    {
        NavInput in;
        addFloor(in, -5, -5, 5, 5, 0.0f);
        addBox(in, -3, 0, -0.5f, 5, 1, 0.5f);                // a wall across the middle, open on the left (x < -3)
        const float wall[8][3] = { {-3,0,-0.5f},{5,0,-0.5f},{5,1,-0.5f},{-3,1,-0.5f},{-3,0,0.5f},{5,0,0.5f},{5,1,0.5f},{-3,1,0.5f} };
        in.addSolid(&wall[0][0], 8);
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));

        const float from[3] = { 2, 0, -4 }, to[3] = { 2, 0, 4 };
        NavPath path;
        CHECK(nm.findPath(from, to, path, err));
        std::printf("  %zu waypoints, length %.2f (straight line would be 8.00)\n", path.count(), path.length);
        CHECK(path.count() >= 3 && !path.partial);
        CHECK(path.length > 8.0f && path.length < 16.0f);   // longer than straight, but not absurdly so
        CHECK(std::fabs(path.point(0)[0] - 2) < 0.1f && std::fabs(path.point(0)[2] + 4) < 0.1f);
        const float* last = path.point(path.count() - 1);
        CHECK(std::fabs(last[0] - 2) < 0.1f && std::fabs(last[2] - 4) < 0.1f);
        // no corner of the path inside the wall, and no straight leg crossing it
        bool clear = true;
        for (size_t i = 0; i + 1 < path.count(); i++)
            for (int k = 0; k <= 100; k++) {
                float f = k / 100.0f, x = path.point(i)[0] + (path.point(i + 1)[0] - path.point(i)[0]) * f,
                      z = path.point(i)[2] + (path.point(i + 1)[2] - path.point(i)[2]) * f;
                if (x > -3 && x < 5 && z > -0.5f && z < 0.5f) clear = false;
            }
        CHECK(clear);
        float ymax = 0; for (size_t i = 0; i < path.count(); i++) ymax = std::fmax(ymax, path.point(i)[1]);
        CHECK(ymax < 0.3f);                                   // stayed on the floor

        std::printf("a route that needs no detour is a single straight leg\n");
        const float a[3] = { 2, 0, -4 }, b[3] = { 4, 0, -2 };
        CHECK(nm.findPath(a, b, path, err) && path.count() == 2 && std::fabs(path.length - std::sqrt(8.0f)) < 0.05f);

        std::printf("points off the navmesh snap, or fail with a reason\n");
        const float inWall[3] = { 1, 0, 0.3f };               // 0.3 inside the wall face: within snapping distance
        CHECK(nm.findPath(from, inWall, path, err));          // inside the wall: snaps to its edge
        const float far[3] = { 30, 0, 30 };
        CHECK(!nm.findPath(from, far, path, err) && !err.empty()); std::printf("  -> %s\n", err.c_str());
        float snapped[3];
        CHECK(nm.nearestPoint(inWall, snapped) && std::fabs(snapped[2]) > 0.5f);
        CHECK(!nm.nearestPoint(far, snapped));

        std::printf("ray cast hits the walkable surface\n");
        const float org[3] = { 3, 5, -3 }, dir[3] = { 0, -1, 0 };
        float th = 0;
        CHECK(nm.raycast(org, dir, th) && std::fabs(th - 5.0f) < 0.1f);
        const float up[3] = { 0, 1, 0 };
        CHECK(!nm.raycast(org, up, th));
        const float offmap[3] = { 30, 5, 30 };
        CHECK(!nm.raycast(offmap, dir, th));
    }

    std::printf("an unreachable goal gives a partial path\n");
    {
        NavInput in;
        addFloor(in, -5, -5, -1, 5, 0.0f);                   // two separate floors with a gap between them
        addFloor(in, 1, -5, 5, 5, 0.0f);
        NavMesh nm; std::string err;
        CHECK(nm.build(in, params, err));
        const float from[3] = { -3, 0, 0 }, to[3] = { 3, 0, 0 };
        NavPath path;
        CHECK(nm.findPath(from, to, path, err));
        CHECK(path.partial);
        const float* last = path.point(path.count() - 1);
        CHECK(last[0] < -0.5f);                               // ended on the start side, as close as possible
        std::printf("  ended at x = %.2f\n", last[0]);
    }

    std::printf("queries before any build\n");
    {
        NavMesh nm; NavPath p; std::string err;
        const float a[3] = { 0, 0, 0 };
        CHECK(!nm.findPath(a, a, p, err) && !err.empty());
        float o[3];
        CHECK(!nm.nearestPoint(a, o));
    }

    std::printf(fails ? "%d FAILED\n" : "all navmesh tests passed\n", fails);
    return fails != 0;
}
