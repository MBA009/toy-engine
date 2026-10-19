#include "navmesh.h"

#include <chrono>
#include <cmath>
#include <memory>

#include "Recast.h"
#include "DetourNavMesh.h"
#include "DetourNavMeshBuilder.h"
#include "DetourNavMeshQuery.h"

namespace {

enum PolyArea : unsigned char { AREA_GROUND = 0 };
enum PolyFlags : unsigned short { FLAG_WALK = 1 };

// Recast allocates its own structures; these wrappers free them on every exit path
struct HeightfieldDeleter { void operator()(rcHeightfield* p) const { rcFreeHeightField(p); } };
struct CompactDeleter { void operator()(rcCompactHeightfield* p) const { rcFreeCompactHeightfield(p); } };
struct ContourDeleter { void operator()(rcContourSet* p) const { rcFreeContourSet(p); } };
struct PolyMeshDeleter { void operator()(rcPolyMesh* p) const { rcFreePolyMesh(p); } };
struct DetailDeleter { void operator()(rcPolyMeshDetail* p) const { rcFreePolyMeshDetail(p); } };

// Recast reports problems through a context object; we just keep the last message
class Context : public rcContext {
public:
    std::string lastError;
protected:
    void doLog(const rcLogCategory category, const char* msg, const int) override {
        if (category == RC_LOG_ERROR) lastError = msg;
    }
};

}  // namespace

NavMesh::~NavMesh() { clear(); }

void NavMesh::clear() {
    if (query_) { dtFreeNavMeshQuery(query_); query_ = nullptr; }
    if (nav_) { dtFreeNavMesh(nav_); nav_ = nullptr; }
    tris_.clear();
    edges_.clear();
    stats_ = NavStats{};
}

bool NavMesh::build(const NavInput& input, const NavParams& p, std::string& error) {
    error.clear();
    const auto t0 = std::chrono::steady_clock::now();
    if (input.tris.empty()) { error = "no geometry to build from (is there a floor?)"; return false; }
    if (p.cellSize <= 0.0f || p.cellHeight <= 0.0f) { error = "cell size and cell height must be positive"; return false; }

    const int nverts = (int)(input.verts.size() / 3);
    const int ntris = (int)(input.tris.size() / 3);

    Context ctx;
    rcConfig cfg{};
    cfg.cs = p.cellSize;
    cfg.ch = p.cellHeight;
    cfg.walkableSlopeAngle = p.agentMaxSlope;
    // Recast works in whole voxels, so world sizes are rounded to cells (conservatively)
    cfg.walkableHeight = (int)std::ceil(p.agentHeight / cfg.ch);
    cfg.walkableClimb = (int)std::floor(p.agentMaxClimb / cfg.ch);
    cfg.walkableRadius = (int)std::ceil(p.agentRadius / cfg.cs);
    // Recast stores these in 8-bit fields internally; values outside the range silently produce garbage
    if (cfg.walkableHeight < 3 || cfg.walkableHeight > 250) {
        error = "agent height must be between 3 and 250 cells tall (" + std::to_string(cfg.walkableHeight) +
                " now); adjust the agent height or the cell height";
        return false;
    }
    if (cfg.walkableClimb < 0 || cfg.walkableClimb > 250) {
        error = "agent max climb must be between 0 and 250 cells (" + std::to_string(cfg.walkableClimb) + " now)";
        return false;
    }
    if (cfg.walkableRadius < 0 || cfg.walkableRadius > 100) {
        error = "agent radius must be at most 100 cells (" + std::to_string(cfg.walkableRadius) + " now); raise the cell size";
        return false;
    }
    cfg.maxEdgeLen = (int)(p.edgeMaxLen / p.cellSize);
    cfg.maxSimplificationError = p.edgeMaxError;
    cfg.minRegionArea = (int)(p.regionMinSize * p.regionMinSize);
    cfg.mergeRegionArea = (int)(p.regionMergeSize * p.regionMergeSize);
    cfg.maxVertsPerPoly = 6;
    cfg.detailSampleDist = p.detailSampleDist < 0.9f ? 0.0f : p.cellSize * p.detailSampleDist;
    cfg.detailSampleMaxError = p.cellHeight * p.detailSampleMaxError;

    rcCalcBounds(input.verts.data(), nverts, cfg.bmin, cfg.bmax);
    rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &cfg.width, &cfg.height);
    if (cfg.width <= 0 || cfg.height <= 0 || (long long)cfg.width * cfg.height > 16000000LL) {
        error = "the scene is too large for this cell size (" + std::to_string(cfg.width) + " x " +
                std::to_string(cfg.height) + " cells); raise the cell size";
        return false;
    }

    // 1. Voxelize: rasterize every triangle into a grid of solid spans, marking the walkable ones
    std::unique_ptr<rcHeightfield, HeightfieldDeleter> solid(rcAllocHeightfield());
    if (!solid || !rcCreateHeightfield(&ctx, *solid, cfg.width, cfg.height, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch)) {
        error = "out of memory creating the heightfield";
        return false;
    }
    std::vector<unsigned char> triAreas((size_t)ntris, 0);
    rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, input.verts.data(), nverts, input.tris.data(), ntris, triAreas.data());
    if (!rcRasterizeTriangles(&ctx, input.verts.data(), nverts, input.tris.data(), triAreas.data(), ntris, *solid, cfg.walkableClimb)) {
        error = "rasterizing the triangles failed";
        return false;
    }

    // 2. Remove spans an agent can't actually stand on
    rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *solid);
    rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *solid);
    rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *solid);

    // 3. Compact the open space, pull it in by the agent radius, and split it into regions
    std::unique_ptr<rcCompactHeightfield, CompactDeleter> chf(rcAllocCompactHeightfield());
    if (!chf || !rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *solid, *chf)) {
        error = "building the compact heightfield failed";
        return false;
    }
    solid.reset();
    // Carve the solid obstacles: everything inside their footprint (floor under them, their roof) becomes unwalkable
    for (const NavSolid& s : input.solids) {
        const int n = (int)(s.hullXZ.size() / 2);
        std::vector<float> poly((size_t)n * 3);
        for (int i = 0; i < n; ++i) {
            poly[(size_t)i * 3] = s.hullXZ[(size_t)i * 2];
            poly[(size_t)i * 3 + 1] = 0.0f;
            poly[(size_t)i * 3 + 2] = s.hullXZ[(size_t)i * 2 + 1];
        }
        rcMarkConvexPolyArea(&ctx, poly.data(), n, s.yMin - 2.0f * cfg.ch, s.yMax + 2.0f * cfg.ch, RC_NULL_AREA, *chf);
    }
    if (!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf)) { error = "eroding the walkable area failed"; return false; }
    if (!rcBuildDistanceField(&ctx, *chf)) { error = "building the distance field failed"; return false; }
    if (!rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea)) { error = "building regions failed"; return false; }

    // 4. Region outlines -> simplified contours -> convex polygons -> height detail
    std::unique_ptr<rcContourSet, ContourDeleter> cset(rcAllocContourSet());
    if (!cset || !rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset)) {
        error = "building contours failed";
        return false;
    }
    if (cset->nconts == 0) {
        error = "no walkable area found (the agent may be too large for the geometry, or there is no floor)";
        return false;
    }
    std::unique_ptr<rcPolyMesh, PolyMeshDeleter> pmesh(rcAllocPolyMesh());
    if (!pmesh || !rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *pmesh)) { error = "building the polygon mesh failed"; return false; }
    std::unique_ptr<rcPolyMeshDetail, DetailDeleter> dmesh(rcAllocPolyMeshDetail());
    if (!dmesh || !rcBuildPolyMeshDetail(&ctx, *pmesh, *chf, cfg.detailSampleDist, cfg.detailSampleMaxError, *dmesh)) {
        error = "building the detail mesh failed";
        return false;
    }
    chf.reset();
    cset.reset();

    // 5. Hand the polygons to Detour
    for (int i = 0; i < pmesh->npolys; ++i) {
        if (pmesh->areas[i] == RC_WALKABLE_AREA) pmesh->areas[i] = AREA_GROUND;
        pmesh->flags[i] = FLAG_WALK;
    }
    dtNavMeshCreateParams dp{};
    dp.verts = pmesh->verts;
    dp.vertCount = pmesh->nverts;
    dp.polys = pmesh->polys;
    dp.polyAreas = pmesh->areas;
    dp.polyFlags = pmesh->flags;
    dp.polyCount = pmesh->npolys;
    dp.nvp = pmesh->nvp;
    dp.detailMeshes = dmesh->meshes;
    dp.detailVerts = dmesh->verts;
    dp.detailVertsCount = dmesh->nverts;
    dp.detailTris = dmesh->tris;
    dp.detailTriCount = dmesh->ntris;
    dp.walkableHeight = p.agentHeight;
    dp.walkableRadius = p.agentRadius;
    dp.walkableClimb = p.agentMaxClimb;
    rcVcopy(dp.bmin, pmesh->bmin);
    rcVcopy(dp.bmax, pmesh->bmax);
    dp.cs = cfg.cs;
    dp.ch = cfg.ch;
    dp.buildBvTree = true;

    unsigned char* navData = nullptr;
    int navDataSize = 0;
    if (!dtCreateNavMeshData(&dp, &navData, &navDataSize)) { error = "Detour could not create the navmesh data"; return false; }
    dtNavMesh* nav = dtAllocNavMesh();
    if (!nav || dtStatusFailed(nav->init(navData, navDataSize, DT_TILE_FREE_DATA))) {
        dtFree(navData);
        if (nav) dtFreeNavMesh(nav);
        error = "Detour could not initialise the navmesh";
        return false;
    }

    dtNavMeshQuery* query = dtAllocNavMeshQuery();
    if (!query || dtStatusFailed(query->init(nav, 2048))) {
        if (query) dtFreeNavMeshQuery(query);
        dtFreeNavMesh(nav);
        error = "Detour could not create the path query";
        return false;
    }

    // Success: from here nothing can fail, so swap in the result
    clear();
    nav_ = nav;
    query_ = query;

    // Debug geometry, in world coordinates. Filled triangles come from the detail mesh (it follows the real
    // surface height); polygon outlines come from the polygon mesh, whose edges are voxel-quantized.
    tris_.reserve((size_t)dmesh->ntris);
    for (int i = 0; i < pmesh->npolys; ++i) {
        const unsigned* m = &dmesh->meshes[i * 4];
        const unsigned vertBase = m[0], triBase = m[2], triCount = m[3];
        for (unsigned t = 0; t < triCount; ++t) {
            const unsigned char* tri = &dmesh->tris[(triBase + t) * 4];
            NavDebugTri d;
            d.poly = (unsigned)i;
            for (int k = 0; k < 3; ++k) {
                const float* v = &dmesh->verts[(vertBase + tri[k]) * 3];
                float* out = k == 0 ? d.a : (k == 1 ? d.b : d.c);
                out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
            }
            tris_.push_back(d);
        }
    }

    const int nvp = pmesh->nvp;
    auto polyVert = [&](unsigned short vi, float* out) {
        const unsigned short* v = &pmesh->verts[vi * 3];
        out[0] = pmesh->bmin[0] + v[0] * pmesh->cs;
        out[1] = pmesh->bmin[1] + (v[1] + 1) * pmesh->ch;     // +1: the top of the voxel, where agents stand
        out[2] = pmesh->bmin[2] + v[2] * pmesh->cs;
    };
    for (int i = 0; i < pmesh->npolys; ++i) {
        const unsigned short* poly = &pmesh->polys[i * nvp * 2];
        int nv = 0;
        while (nv < nvp && poly[nv] != RC_MESH_NULL_IDX) ++nv;
        for (int j = 0; j < nv; ++j) {
            const unsigned short next = poly[nvp + j];
            NavDebugEdge e;
            polyVert(poly[j], e.a);
            polyVert(poly[(j + 1) % nv], e.b);
            e.border = (next == RC_MESH_NULL_IDX) || (next & 0x8000) != 0;
            // Interior edges are shared by two polygons: keep one copy (the one from the lower-numbered polygon)
            if (!e.border && next < (unsigned short)i) continue;
            edges_.push_back(e);
        }
    }

    stats_.polygons = pmesh->npolys;
    stats_.vertices = pmesh->nverts;
    stats_.triangles = dmesh->ntris;
    rcVcopy(stats_.boundsMin, pmesh->bmin);
    rcVcopy(stats_.boundsMax, pmesh->bmax);
    stats_.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// ---------------------------------------------------------------------------
// Path queries
// ---------------------------------------------------------------------------

namespace {
const float SEARCH_EXTENTS[3] = { 0.5f, 1.0f, 0.5f };      // how far from the navmesh a point may be and still snap to it
const int MAX_POLYS = 256;                                 // longest corridor of polygons a path may cross
}

bool NavMesh::moveAlong(const float start[3], const float end[3], float out[3]) {
    if (!query_) return false;
    dtQueryFilter filter;
    dtPolyRef ref = 0;
    float snap[3];
    if (dtStatusFailed(query_->findNearestPoly(start, SEARCH_EXTENTS, &filter, &ref, snap)) || ref == 0) return false;
    dtPolyRef visited[16];
    int nvisited = 0;
    float result[3];
    if (dtStatusFailed(query_->moveAlongSurface(ref, snap, end, &filter, result, visited, &nvisited, 16))) return false;
    float h = 0.0f;
    if (nvisited > 0 && dtStatusSucceed(query_->getPolyHeight(visited[nvisited - 1], result, &h))) result[1] = h;
    out[0] = result[0]; out[1] = result[1]; out[2] = result[2];
    return true;
}

bool NavMesh::nearestPoint(const float pos[3], float out[3]) {
    if (!query_) return false;
    dtQueryFilter filter;
    dtPolyRef ref = 0;
    float nearest[3];
    if (dtStatusFailed(query_->findNearestPoly(pos, SEARCH_EXTENTS, &filter, &ref, nearest)) || ref == 0) return false;
    out[0] = nearest[0]; out[1] = nearest[1]; out[2] = nearest[2];
    return true;
}

bool NavMesh::findPath(const float start[3], const float end[3], NavPath& out, std::string& error) {
    error.clear();
    out = NavPath{};
    if (!query_) { error = "no navmesh built yet"; return false; }

    dtQueryFilter filter;       // all polygons walkable, equal cost: the shortest path wins
    dtPolyRef startRef = 0, endRef = 0;
    float startPos[3], endPos[3];
    query_->findNearestPoly(start, SEARCH_EXTENTS, &filter, &startRef, startPos);
    if (!startRef) { error = "the start point is not near the navmesh"; return false; }
    query_->findNearestPoly(end, SEARCH_EXTENTS, &filter, &endRef, endPos);
    if (!endRef) { error = "the end point is not near the navmesh"; return false; }

    // 1. A corridor of polygons from start to end (A* over the polygon graph)
    dtPolyRef polys[MAX_POLYS];
    int npolys = 0;
    if (dtStatusFailed(query_->findPath(startRef, endRef, startPos, endPos, &filter, polys, &npolys, MAX_POLYS)) || npolys == 0) {
        error = "no route found";
        return false;
    }
    // If the end polygon wasn't reached, the corridor ends at the closest polygon; aim for the closest point on it
    if (polys[npolys - 1] != endRef) {
        out.partial = true;
        float closest[3];
        bool overPoly = false;
        if (dtStatusSucceed(query_->closestPointOnPoly(polys[npolys - 1], endPos, closest, &overPoly))) {
            endPos[0] = closest[0]; endPos[1] = closest[1]; endPos[2] = closest[2];
        }
    }

    // 2. Pull the corridor tight into the straight-line corners ("string pulling")
    float straight[MAX_POLYS * 3];
    unsigned char flags[MAX_POLYS];
    dtPolyRef refs[MAX_POLYS];
    int n = 0;
    if (dtStatusFailed(query_->findStraightPath(startPos, endPos, polys, npolys, straight, flags, refs, &n, MAX_POLYS)) || n == 0) {
        error = "could not straighten the path";
        return false;
    }
    out.points.assign(straight, straight + n * 3);
    for (int i = 1; i < n; ++i) {
        const float dx = straight[i * 3] - straight[(i - 1) * 3];
        const float dy = straight[i * 3 + 1] - straight[(i - 1) * 3 + 1];
        const float dz = straight[i * 3 + 2] - straight[(i - 1) * 3 + 2];
        out.length += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return true;
}

bool NavMesh::raycast(const float o[3], const float d[3], float& tHit) const {
    // Moeller-Trumbore ray/triangle test against the walkable surface
    bool hit = false;
    float best = 0.0f;
    for (const NavDebugTri& t : tris_) {
        const float e1[3] = { t.b[0] - t.a[0], t.b[1] - t.a[1], t.b[2] - t.a[2] };
        const float e2[3] = { t.c[0] - t.a[0], t.c[1] - t.a[1], t.c[2] - t.a[2] };
        const float p[3] = { d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0] };
        const float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (std::fabs(det) < 1e-9f) continue;                      // ray parallel to the triangle
        const float inv = 1.0f / det;
        const float s[3] = { o[0] - t.a[0], o[1] - t.a[1], o[2] - t.a[2] };
        const float u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
        if (u < 0.0f || u > 1.0f) continue;
        const float q[3] = { s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0] };
        const float v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
        if (v < 0.0f || u + v > 1.0f) continue;
        const float tt = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
        if (tt > 0.0f && (!hit || tt < best)) { hit = true; best = tt; }
    }
    if (hit) tHit = best;
    return hit;
}
