#pragma once
#include <algorithm>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Navigation mesh (Step 7.1): turns triangle geometry into a set of convex walkable polygons that agents
// can path across (Recast does the building, Detour will do the path queries in Step 7.2).
//
// This header deliberately knows nothing about Vulkan, the ECS or the editor: triangles in, polygons out.
// ---------------------------------------------------------------------------

class dtNavMesh;        // Detour's runtime navmesh (defined in DetourNavMesh.h)
class dtNavMeshQuery;   // Detour's path query object

// All sizes are in world units. The defaults suit this engine's scale (a unit cube is 1 wide, the floor is
// a couple of units across, agents are small).
struct NavParams {
    float cellSize      = 0.05f;   // voxel width/depth: smaller = more accurate and slower
    float cellHeight    = 0.02f;   // voxel height
    float agentHeight   = 0.30f;   // clearance an agent needs overhead
    float agentRadius   = 0.06f;   // walkable area is pulled in this far from walls and ledges
    float agentMaxClimb = 0.05f;   // tallest step an agent can climb (smaller than a cube, so cubes block)
    float agentMaxSlope = 45.0f;   // steepest walkable slope, in degrees
    float regionMinSize   = 8.0f;  // islands smaller than this many cells are dropped
    float regionMergeSize = 20.0f;
    float edgeMaxLen      = 1.0f;  // longest polygon edge along a boundary
    float edgeMaxError    = 1.3f;  // how far the simplified boundary may drift from the real one (in cells)
    float detailSampleDist     = 6.0f;
    float detailSampleMaxError = 1.0f;
};

// Triangles to build from: positions as x,y,z floats, triangles as index triples (counter-clockwise from
// the walkable side, i.e. the top of a floor has its normal pointing up).
//
// "Solids" are obstacles that are filled, not hollow. Triangles alone only describe an object's surface, so
// the floor hidden INSIDE a closed box would still look walkable; a solid carves its whole footprint out of
// the walkable area (and its top, so agents never plan onto the roof of a crate). The footprint is the convex
// hull of the object, which over-blocks a little for L-shaped or hollow objects.
struct NavSolid { std::vector<float> hullXZ; float yMin = 0, yMax = 0; };    // hull as x,z pairs, counter-clockwise

struct NavInput {
    std::vector<float> verts;
    std::vector<int> tris;
    std::vector<NavSolid> solids;

    // Adds a solid from the object's world-space points (x,y,z triples)
    void addSolid(const float* pointsXYZ, int count) {
        if (count < 3) return;
        struct P { float x, z; };
        std::vector<P> pts((size_t)count);
        NavSolid s;
        s.yMin = s.yMax = pointsXYZ[1];
        for (int i = 0; i < count; ++i) {
            pts[(size_t)i] = { pointsXYZ[i * 3], pointsXYZ[i * 3 + 2] };
            s.yMin = std::min(s.yMin, pointsXYZ[i * 3 + 1]);
            s.yMax = std::max(s.yMax, pointsXYZ[i * 3 + 1]);
        }
        std::sort(pts.begin(), pts.end(), [](const P& a, const P& b) { return a.x < b.x || (a.x == b.x && a.z < b.z); });
        auto cross = [](const P& o, const P& a, const P& b) { return (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x); };
        std::vector<P> hull((size_t)count * 2);       // Andrew's monotone chain
        size_t k = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
            hull[k++] = pts[i];
        }
        for (size_t i = pts.size() - 1, t = k + 1; i > 0; --i) {
            while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i - 1]) <= 0) --k;
            hull[k++] = pts[i - 1];
        }
        hull.resize(k > 1 ? k - 1 : k);
        if (hull.size() < 3) return;                  // all points in a line: nothing to carve
        for (const P& p : hull) { s.hullXZ.push_back(p.x); s.hullXZ.push_back(p.z); }
        solids.push_back(std::move(s));
    }

    void addTriangle(const float* a, const float* b, const float* c) {
        int base = (int)(verts.size() / 3);
        for (const float* p : { a, b, c }) verts.insert(verts.end(), p, p + 3);
        tris.insert(tris.end(), { base, base + 1, base + 2 });
    }
    size_t triangleCount() const { return tris.size() / 3; }
};

// Debug geometry for drawing the navmesh as an overlay
struct NavDebugTri { float a[3], b[3], c[3]; unsigned poly; };
struct NavDebugEdge { float a[3], b[3]; bool border; };   // border = the edge of the walkable area (not shared with another polygon)

struct NavStats {
    int polygons = 0, vertices = 0, triangles = 0;
    double buildMs = 0.0;
    float boundsMin[3] = { 0, 0, 0 }, boundsMax[3] = { 0, 0, 0 };
};

// A route across the navmesh: the corners an agent has to walk through, in order, as x,y,z floats.
// Walking straight from each point to the next never leaves the walkable area.
struct NavPath {
    std::vector<float> points;
    float length = 0.0f;        // total length in world units
    bool partial = false;       // the end point was unreachable; the path goes to the closest reachable point instead
    size_t count() const { return points.size() / 3; }
    const float* point(size_t i) const { return &points[i * 3]; }
};

class NavMesh {
public:
    NavMesh() = default;
    ~NavMesh();
    NavMesh(const NavMesh&) = delete;
    NavMesh& operator=(const NavMesh&) = delete;

    // Builds a new navmesh. On failure returns false with the reason in `error`, and the previous
    // navmesh (if any) is left untouched.
    bool build(const NavInput& input, const NavParams& params, std::string& error);
    void clear();

    // ---- Path queries (Step 7.2) ----
    // Finds the shortest walkable route from start to end (world positions). Points that are not exactly on
    // the navmesh (clicked inside an obstacle, or hovering above the floor) are snapped to the nearest walkable
    // spot within about half a unit sideways and one unit vertically; further away is an error.
    bool findPath(const float start[3], const float end[3], NavPath& out, std::string& error);
    // Snaps a position to the navmesh (same search distance as findPath). False if nothing is near.
    bool nearestPoint(const float pos[3], float out[3]);
    // Walks from `start` toward `end` along the navmesh and stops (or slides along the edge) where the surface ends,
    // so a character moved with this can never leave the walkable area. `out` is the new position, with its height
    // taken from the surface. False if `start` is not near the navmesh.
    bool moveAlong(const float start[3], const float end[3], float out[3]);
    // First hit of a ray (origin + t * dir, dir normalized) with the walkable surface; t is the distance.
    bool raycast(const float origin[3], const float dir[3], float& tHit) const;

    bool valid() const { return nav_ != nullptr; }
    const dtNavMesh* detour() const { return nav_; }     // for path queries (Step 7.2)
    const NavStats& stats() const { return stats_; }
    const std::vector<NavDebugTri>& debugTriangles() const { return tris_; }
    const std::vector<NavDebugEdge>& debugEdges() const { return edges_; }

private:
    dtNavMesh* nav_ = nullptr;
    dtNavMeshQuery* query_ = nullptr;
    NavStats stats_;
    std::vector<NavDebugTri> tris_;
    std::vector<NavDebugEdge> edges_;
};
