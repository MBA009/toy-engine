// Phase 3 benchmark: the same scene updated + prepared for drawing by
//   (a) the data-oriented ECS  and  (b) a conventional virtual-dispatch OOP scene.
// Console only, no window, no GPU. Build in Release.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "components.h"
#include "ecs.h"
#include "ecs_systems.h"
#include "oop_scene.h"

// ---- one deterministic description of the scene, fed to both implementations ----
struct ObjectDesc {
    Transform t;
    int   kind;                       // 0 = static prop, 1 = spinner, 2 = bobbing spinner
    float speed, amp, freq, phase;
    Vec3  vel;
};

static std::vector<ObjectDesc> makeScene(int n) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    int side = (int)std::ceil(std::sqrt((float)n));
    std::vector<ObjectDesc> v(n);
    for (int i = 0; i < n; ++i) {
        ObjectDesc& d = v[i];
        d.t.position = { (float)(i % side), u(rng), (float)(i / side) };
        d.t.yaw   = u(rng) * 6.28f;
        d.t.scale = 0.5f + u(rng);
        float r = u(rng);
        d.kind  = r < 0.2f ? 0 : (r < 0.5f ? 1 : 2);    // 20% static, 30% spin, 50% bob+spin
        d.speed = 0.5f + u(rng);
        d.amp   = 0.1f + 0.4f * u(rng);
        d.freq  = 1.0f + 2.0f * u(rng);
        d.phase = u(rng) * 6.28f;
        d.vel   = { u(rng) - 0.5f, 0.0f, u(rng) - 0.5f };
    }
    return v;
}

static void buildEcs(Registry& reg, const std::vector<ObjectDesc>& scene) {
    for (size_t i = 0; i < scene.size(); ++i) {
        const ObjectDesc& d = scene[i];
        Entity e = reg.create();
        reg.add<Transform>(e, d.t);
        reg.add<MeshRenderer>(e, MeshRenderer{ (uint32_t)(i % 4), (uint32_t)(i % 8) });
        reg.add<Name>(e, Name{ "object_" + std::to_string(i) });
        reg.add<Velocity>(e, Velocity{ d.vel });
        if (d.kind >= 1) reg.add<Spin>(e, Spin{ d.speed });
        if (d.kind == 2) reg.add<Bob>(e, Bob{ d.amp, d.freq, d.t.position.y, d.phase });
    }
}

// shuffleLayout = false: objects allocated back-to-back (the OOP best case).
// shuffleLayout = true : allocation order differs from update order, like a scene
//                        that has had objects created and destroyed for a while.
static std::vector<std::unique_ptr<GameObject>>
buildOop(const std::vector<ObjectDesc>& scene, bool shuffleLayout) {
    std::vector<std::unique_ptr<GameObject>> objs;
    objs.reserve(scene.size());
    std::vector<size_t> allocOrder(scene.size());
    for (size_t i = 0; i < allocOrder.size(); ++i) allocOrder[i] = i;
    if (shuffleLayout) { std::mt19937 rng(777); std::shuffle(allocOrder.begin(), allocOrder.end(), rng); }

    std::vector<std::unique_ptr<GameObject>> byIndex(scene.size());
    for (size_t idx : allocOrder) {
        const ObjectDesc& d = scene[idx];
        std::string name = "object_" + std::to_string(idx);
        std::unique_ptr<GameObject> o;
        switch (d.kind) {
            case 0:  o = std::make_unique<StaticProp>(d.t, name); break;
            case 1:  o = std::make_unique<Spinner>(d.t, name, d.speed); break;
            default: o = std::make_unique<BobbingSpinner>(d.t, name, d.speed, d.amp, d.freq, d.phase); break;
        }
        o->velocity = d.vel;
        o->meshId = (uint32_t)(idx % 4);
        o->materialId = (uint32_t)(idx % 8);
        byIndex[idx] = std::move(o);
    }
    return byIndex;     // scene order, but heap addresses follow allocOrder
}

static double nowSec() {
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double>(C::now().time_since_epoch()).count();
}

// Runs frames for ~warmup then ~measure seconds; returns average ms per frame.
template <typename FrameFn>
static double timeIt(FrameFn&& frame, double warmup, double measure) {
    const float dt = 1.0f / 60.0f;
    float time = 0.0f;
    double t0 = nowSec();
    while (nowSec() - t0 < warmup) { time += dt; frame(dt, time); }
    int frames = 0;
    double start = nowSec(), end;
    do {
        for (int k = 0; k < 4; ++k) { time += dt; frame(dt, time); ++frames; }
        end = nowSec();
    } while (end - start < measure);
    return (end - start) * 1000.0 / frames;
}

static double checksum(const std::vector<DrawItem>& items) {
    double s = 0;
    for (const DrawItem& d : items) s += d.model.m[0][0] + d.model.m[3][0] + d.model.m[3][1] + d.model.m[3][2] + d.meshId;
    return s;
}

static bool nearlyEqual(const Mat4& a, const Mat4& b) {
    for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r)
        if (std::fabs(a.m[c][r] - b.m[c][r]) > 1e-4f) return false;
    return true;
}

// ---- correctness tests (real checks, not asserts: they run in Release too) ----
static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL: %s (line %d)\n", #cond, __LINE__); ++g_failures; } } while (0)

static void runTests() {
    std::printf("Running ECS tests...\n");
    {   // modelMatrix matches the composed matrices used in bench_scene.h
        Transform t; t.position = { 1.0f, 2.0f, -3.0f }; t.yaw = 0.7f; t.scale = 1.5f;
        Mat4 ref = Mat4::translate(t.position) * Mat4::rotateY(t.yaw) * Mat4::scale({ t.scale, t.scale, t.scale });
        CHECK(nearlyEqual(modelMatrix(t), ref));
    }
    {   // add / has / get / remove
        Registry r;
        Entity a = r.create(), b = r.create(), c = r.create();
        r.add<Transform>(a, Transform{ {1,0,0}, 0, 1 });
        r.add<Transform>(b, Transform{ {2,0,0}, 0, 1 });
        r.add<Transform>(c, Transform{ {3,0,0}, 0, 1 });
        r.add<Spin>(b, Spin{ 2.0f });
        CHECK(r.has<Transform>(a) && r.has<Spin>(b) && !r.has<Spin>(a));
        CHECK(r.get<Transform>(c).position.x == 3.0f);
        r.remove<Transform>(a);                       // swap-and-pop moves c into a's slot
        CHECK(!r.has<Transform>(a));
        CHECK(r.get<Transform>(c).position.x == 3.0f && r.get<Transform>(b).position.x == 2.0f);
        CHECK(r.pool<Transform>().size() == 2);
    }
    {   // each<> visits exactly the entities that have all listed components
        Registry r;
        int expect = 0;
        for (int i = 0; i < 100; ++i) {
            Entity e = r.create();
            r.add<Transform>(e);
            if (i % 2 == 0) r.add<Spin>(e);
            if (i % 4 == 0) { r.add<Bob>(e); ++expect; }
        }
        int visited = 0;
        r.each<Bob, Spin, Transform>([&](Entity, Bob&, Spin&, Transform&) { ++visited; });
        CHECK(visited == expect);
    }
    {   // destroy removes from every pool; the ID is reused
        Registry r;
        Entity a = r.create();
        r.add<Transform>(a); r.add<Spin>(a);
        r.destroy(a);
        CHECK(!r.alive(a) && !r.has<Transform>(a) && !r.has<Spin>(a));
        CHECK(r.pool<Transform>().size() == 0);
        Entity b = r.create();
        CHECK(b == a && r.alive(b) && !r.has<Spin>(b));
    }
    std::printf(g_failures ? "  %d FAILED\n" : "  all passed\n", g_failures);
}

// The four workloads. Each takes the scene implementation and runs one frame of it.
enum Workload { W_MOVE, W_UPDATE, W_DRAW, W_FRAME, W_COUNT };
static const char* WORKLOAD_NAME[W_COUNT] = {
    "move (position += velocity*dt)",
    "update (spin + bob, uses sin)",
    "draw list (model matrix per object)",
    "full frame (all of the above)"
};

using OopScene = std::vector<std::unique_ptr<GameObject>>;

static void ecsRun(Workload w, Registry& reg, std::vector<DrawItem>& out, float dt, float time) {
    if (w == W_MOVE || w == W_FRAME)   moveSystem(reg, dt);
    if (w == W_UPDATE || w == W_FRAME) { spinSystem(reg, dt); bobSystem(reg, time); }
    if (w == W_DRAW || w == W_FRAME)   buildDrawList(reg, out);
}

static void oopRun(Workload w, OopScene& objs, std::vector<DrawItem>& out, float dt, float time) {
    if (w == W_MOVE || w == W_FRAME)   for (auto& o : objs) o->move(dt);
    if (w == W_UPDATE || w == W_FRAME) for (auto& o : objs) o->update(dt, time);
    if (w == W_DRAW || w == W_FRAME) { out.clear(); for (auto& o : objs) o->render(out); }
}

int main(int argc, char** argv) {
    bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    runTests();
    if (g_failures) return 1;

    const int counts[] = { 1000, 10000, 100000, 1000000 };
    const double warmup = quick ? 0.1 : 0.5, measure = quick ? 0.3 : 2.0;

    std::printf("\nsizeof(GameObject)=%zu bytes (OOP, one heap allocation per object, plus subclass fields)\n", sizeof(GameObject));
    std::printf("ECS touches per object: move = Velocity %zu B + Transform %zu B; update = Spin/Bob + Transform\n",
                sizeof(Velocity), sizeof(Transform));
    std::printf("\nworkload,objects,ecs_ms,oop_ms,oop_shuffled_ms,oop_over_ecs,oop_shuffled_over_ecs\n");

    for (int n : counts) {
        auto scene = makeScene(n);
        Registry reg;
        buildEcs(reg, scene);
        OopScene oop     = buildOop(scene, false);
        OopScene oopShuf = buildOop(scene, true);

        std::vector<DrawItem> dEcs, dOop, dShuf;
        dEcs.reserve(n); dOop.reserve(n); dShuf.reserve(n);

        // Correctness: all three must agree after the same 10 full frames.
        {
            const float dt = 1.0f / 60.0f; float time = 0;
            for (int k = 0; k < 10; ++k) {
                time += dt;
                ecsRun(W_FRAME, reg, dEcs, dt, time);
                oopRun(W_FRAME, oop, dOop, dt, time);
                oopRun(W_FRAME, oopShuf, dShuf, dt, time);
            }
            double a = checksum(dEcs), b = checksum(dOop), c = checksum(dShuf);
            if (dEcs.size() != dOop.size() || std::fabs(a - b) > 1e-3 * std::fabs(a) + 1e-3 ||
                std::fabs(b - c) > 1e-3 * std::fabs(b) + 1e-3) {
                std::printf("MISMATCH at n=%d: ecs=%.4f oop=%.4f shuf=%.4f\n", n, a, b, c);
                return 1;
            }
        }

        for (int wi = 0; wi < W_COUNT; ++wi) {
            Workload w = (Workload)wi;
            double e = timeIt([&](float dt, float t) { ecsRun(w, reg, dEcs, dt, t); }, warmup, measure);
            double o = timeIt([&](float dt, float t) { oopRun(w, oop, dOop, dt, t); }, warmup, measure);
            double s = timeIt([&](float dt, float t) { oopRun(w, oopShuf, dShuf, dt, t); }, warmup, measure);
            std::printf("\"%s\",%d,%.4f,%.4f,%.4f,%.2f,%.2f\n", WORKLOAD_NAME[wi], n, e, o, s, o / e, s / e);
            std::fflush(stdout);
        }
    }
    return 0;
}
