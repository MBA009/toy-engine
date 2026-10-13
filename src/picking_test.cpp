#include <cstdio>
#include "picking.h"
#include "camera.h"
static int fails = 0;
#define CHECK(c) do { if(!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while(0)

// project a world point to window pixels with the same matrices the renderer uses
static void project(const Mat4& vp, Vec3 p, float W, float H, float& sx, float& sy) {
    Vec4 c = vp * Vec4(p, 1.0f);
    float nx = c.x / c.w, ny = c.y / c.w;
    sx = (nx * 0.5f + 0.5f) * W;
    sy = (1.0f - (ny * 0.5f + 0.5f)) * H;
}

int main() {
    FlyCamera cam;                                  // the default start pose
    float W = 800, H = 600;
    Mat4 proj = Mat4::perspective(3.14159265f / 4.0f, W / H, 0.1f, 100.0f);
    Mat4 vp = proj * cam.view();
    Mat4 inv = vp.inverse();

    Registry reg;
    auto add = [&](Vec3 pos, float scale, float yaw) {
        Entity e = reg.create();
        Transform t; t.position = pos; t.scale = scale; t.yaw = yaw;
        reg.add<Transform>(e, t); reg.add<MeshRenderer>(e, MeshRenderer{0, 0});
        return e;
    };
    Entity a = add({ 0.0f, 0.0f, -1.5f }, 1.0f, 0.0f);      // like "Spinner"
    Entity b = add({ 1.4f, -0.15f, -3.0f }, 0.7f, 0.9f);    // off to the side and rotated: nothing in front of it
    Entity small = add({ -0.7f, -0.35f, -1.0f }, 0.3f, 2.0f);

    // Clicking where each cube's centre projects must pick that cube
    for (Entity e : { a, b, small }) {
        float sx, sy;
        project(vp, reg.get<Transform>(e).position, W, H, sx, sy);
        Ray r = screenToRay(inv, sx, sy, W, H);
        float t = 0;
        Entity hit = pickEntity(reg, r, &t);
        std::printf("entity %u centre -> pixel (%.1f, %.1f) -> hit %u at t=%.3f\n", e, sx, sy, hit, t);
        CHECK(hit == e);
    }

    // Occlusion: a cube hiding directly behind another along the view ray is not picked (closest hit wins).
    // (0.7, -0.15, -3) projects to a pixel whose ray passes through cube a first.
    {
        Registry occ;
        Entity front = occ.create(); Transform tf; tf.position = { 0.0f, 0.0f, -1.5f };
        occ.add<Transform>(front, tf); occ.add<MeshRenderer>(front, MeshRenderer{0,0});
        Entity back = occ.create(); Transform tb; tb.position = { 0.7f, -0.15f, -3.0f }; tb.scale = 0.7f;
        occ.add<Transform>(back, tb); occ.add<MeshRenderer>(back, MeshRenderer{0,0});
        float sx, sy; project(vp, tb.position, W, H, sx, sy);
        float t = 0;
        CHECK(pickEntity(occ, screenToRay(inv, sx, sy, W, H), &t) == front);
        occ.remove<MeshRenderer>(front);                       // take the occluder away: now the back cube is hit
        CHECK(pickEntity(occ, screenToRay(inv, sx, sy, W, H)) == back);
    }

    // The ray origin is on the near plane in front of the camera, pointing the way the camera looks
    Ray center = screenToRay(inv, W / 2, H / 2, W, H);
    Vec3 f = cam.forward();
    CHECK(center.dir.dot(f) > 0.999f);

    // A pixel in the top-left corner hits nothing
    Entity none = pickEntity(reg, screenToRay(inv, 5, 5, W, H));
    CHECK(none == NULL_ENTITY);

    // Closest wins: a ray down -Z through x=0.7 hits b only; a ray through both a's and a cube hidden behind it hits a
    Entity hiddenBehindA = add({ 0.0f, 0.0f, -4.0f }, 0.5f, 0.0f);
    Ray along = { { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -1.0f } };
    float t = 0; Entity hit = pickEntity(reg, along, &t);
    CHECK(hit == a); CHECK(std::fabs(t - 2.0f) < 1e-3f);      // a's front face is at z=-1.0, 2.0 from the origin
    (void)hiddenBehindA;

    // Rotation: a scale-1 cube turned 45 degrees reaches out to 0.707 along x at its corner
    Registry r2;
    Entity c = r2.create(); Transform tc; tc.yaw = 3.14159265f / 4; r2.add<Transform>(c, tc); r2.add<MeshRenderer>(c, MeshRenderer{0,0});
    Ray grazeCorner = { { 0.65f, 0.0f, 5.0f }, { 0.0f, 0.0f, -1.0f } };   // x = 0.65 > 0.5 would miss an unrotated cube
    CHECK(pickEntity(r2, grazeCorner) == c);
    Ray missBeyond = { { 0.75f, 0.0f, 5.0f }, { 0.0f, 0.0f, -1.0f } };     // x = 0.75 > 0.707 misses
    CHECK(pickEntity(r2, missBeyond) == NULL_ENTITY);

    // Scale: a scale-2 cube spans +-1
    Registry r3;
    Entity d = r3.create(); Transform td; td.scale = 2.0f; r3.add<Transform>(d, td); r3.add<MeshRenderer>(d, MeshRenderer{0,0});
    CHECK(pickEntity(r3, { { 0.95f, 0.0f, 5.0f }, { 0.0f, 0.0f, -1.0f } }) == d);
    CHECK(pickEntity(r3, { { 1.05f, 0.0f, 5.0f }, { 0.0f, 0.0f, -1.0f } }) == NULL_ENTITY);

    // A cube behind the ray origin is not hit
    CHECK(pickEntity(r3, { { 0.0f, 0.0f, 5.0f }, { 0.0f, 0.0f, 1.0f } }) == NULL_ENTITY);

    std::printf(fails ? "%d FAILED\n" : "all picking tests passed\n", fails);
    return fails != 0;
}
