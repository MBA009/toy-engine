// Headless test for scene_io.h: save -> load round trip and corrupt-file handling.
//   g++ -std=c++17 -I. -I<dir containing nlohmann/> scene_io_test.cpp -o scene_io_test && ./scene_io_test
#include <cstdio>
#include <map>
#include "scene_io.h"
using namespace sceneio;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static bool same(float a, float b) { return a == b; }   // exact: floats must survive the text round trip bit for bit

int main() {
    const std::map<uint32_t, std::string> keys = { { 0, "builtin:cube" }, { 2, "assets/Duck.glb" }, { 3, "assets/Box.glb" } };
    MeshKeyFn keyFn = [&](uint32_t id) { auto it = keys.find(id); return it == keys.end() ? std::string("builtin:cube") : it->second; };
    MeshResolveFn resolve = [&](const std::string& k) -> int {
        for (auto& kv : keys) if (kv.second == k) return (int)kv.first;
        return -1;
    };

    std::printf("round trip\n");
    Registry a;
    Entity e0 = a.create();
    a.add<Name>(e0, Name{ "Spinner \"quoted\" \xC3\xA9" });                       // quotes + UTF-8
    a.add<Transform>(e0, Transform{ { 0.1f, -2.3456789f, 1e-7f }, 3.14159274f, 0.3f });
    a.add<MeshRenderer>(e0, MeshRenderer{ 2, 5 });
    a.add<Spin>(e0, Spin{ -1.2f });
    a.add<Bob>(e0, Bob{ 0.25f, 3.0f, -0.4f, 1.7f });
    Entity e1 = a.create();
    a.add<Transform>(e1, Transform{});
    Entity e2 = a.create();                 // destroyed: must not be saved, and the gap must not matter
    a.add<Transform>(e2, Transform{});
    a.destroy(e2);
    Entity e3 = a.create();                 // reuses e2's id
    a.add<Transform>(e3, Transform{ { 5, 6, 7 }, 0.0f, 2.0f });
    a.add<MeshRenderer>(e3, MeshRenderer{ 3, 0 });
    a.add<Velocity>(e3, Velocity{ { 1.0f, 0.0f, -2.0f } });

    json doc = newSceneDoc();
    doc["entities"] = entitiesToJson(a, keyFn);
    CHECK(doc["entities"].size() == 3);
    std::string err;
    CHECK(writeSceneFile("tmp_scene_test/dir/s.json", doc, err));
    CHECK(!std::filesystem::exists("tmp_scene_test/dir/s.json.tmp"));

    json back;
    CHECK(readSceneFile("tmp_scene_test/dir/s.json", back, err));
    Registry b;
    std::vector<std::string> warn;
    entitiesFromJson(back["entities"], b, resolve, 1000, warn);
    CHECK(warn.empty());
    CHECK(b.size() == 3);
    // entities are written in id order, so the loaded ones are 0,1,2 = e0,e1,e3
    CHECK(b.get<Name>(0).value == a.get<Name>(e0).value);
    const Transform &t0a = a.get<Transform>(e0), &t0b = b.get<Transform>(0);
    CHECK(same(t0a.position.x, t0b.position.x) && same(t0a.position.y, t0b.position.y) && same(t0a.position.z, t0b.position.z));
    CHECK(same(t0a.yaw, t0b.yaw) && same(t0a.scale, t0b.scale));
    CHECK(b.get<MeshRenderer>(0).meshId == 2 && b.get<MeshRenderer>(0).materialId == 5);
    CHECK(same(b.get<Spin>(0).speed, -1.2f));
    const Bob &ba = a.get<Bob>(e0), &bb = b.get<Bob>(0);
    CHECK(same(ba.amplitude, bb.amplitude) && same(ba.frequency, bb.frequency) && same(ba.baseY, bb.baseY) && same(ba.phase, bb.phase));
    CHECK(!b.has<Name>(1) && !b.has<MeshRenderer>(1) && !b.has<Spin>(1) && !b.has<Bob>(1) && !b.has<Velocity>(1));
    CHECK(b.get<MeshRenderer>(2).meshId == 3 && same(b.get<Velocity>(2).value.z, -2.0f) && same(b.get<Transform>(2).scale, 2.0f));

    std::printf("unknown mesh falls back to the cube with a warning\n");
    {
        json d = newSceneDoc();
        d["entities"] = json::array({ json{ { "mesh", { { "asset", "assets/Gone.glb" } } } } });
        Registry r; std::vector<std::string> w;
        entitiesFromJson(d["entities"], r, resolve, 10, w);
        CHECK(r.size() == 1 && r.get<MeshRenderer>(0).meshId == 0 && w.size() == 1);
    }

    std::printf("entity cap\n");
    {
        json list = json::array();
        for (int i = 0; i < 5; i++) list.push_back(json::object());
        Registry r; std::vector<std::string> w;
        entitiesFromJson(list, r, resolve, 3, w);
        CHECK(r.size() == 3 && w.size() == 1);
    }

    std::printf("corrupt input is rejected, never crashes\n");
    auto expectThrow = [&](const json& list) {
        Registry r; std::vector<std::string> w; bool threw = false;
        try { entitiesFromJson(list, r, resolve, 100, w); } catch (const std::exception& ex) { threw = true; std::printf("  -> %s\n", ex.what()); }
        return threw;
    };
    CHECK(expectThrow(json::object()));                                                            // not an array
    CHECK(expectThrow(json::array({ 5 })));                                                        // entity not an object
    CHECK(expectThrow(json::array({ json{ { "transform", { { "position", json::array({ 1, 2 }) } } } } })));        // short vec
    CHECK(expectThrow(json::array({ json{ { "transform", { { "position", json::array({ 1, "x", 3 }) } } } } })));   // non-number
    CHECK(expectThrow(json::array({ json{ { "transform", { { "yaw", nullptr } } } } })));                           // NaN saved as null
    CHECK(expectThrow(json::array({ json{ { "name", 7 } } })));
    CHECK(expectThrow(json::array({ json{ { "spin", { { "speed", "fast" } } } } })));

    auto writeText = [](const char* p, const char* text) { std::ofstream f(p); f << text; };
    json d;
    writeText("tmp_scene_test/garbage.json", "{ this is not json");
    CHECK(!readSceneFile("tmp_scene_test/garbage.json", d, err)); std::printf("  -> %s\n", err.c_str());
    writeText("tmp_scene_test/other.json", "{\"hello\": 1}");
    CHECK(!readSceneFile("tmp_scene_test/other.json", d, err)); std::printf("  -> %s\n", err.c_str());
    writeText("tmp_scene_test/v2.json", "{\"format\":\"toy-engine-scene\",\"version\":2,\"entities\":[]}");
    CHECK(!readSceneFile("tmp_scene_test/v2.json", d, err)); std::printf("  -> %s\n", err.c_str());
    writeText("tmp_scene_test/empty.json", "");
    CHECK(!readSceneFile("tmp_scene_test/empty.json", d, err));
    CHECK(!readSceneFile("tmp_scene_test/nope.json", d, err)); std::printf("  -> %s\n", err.c_str());

    std::filesystem::remove_all("tmp_scene_test");
    std::printf(fails ? "%d FAILED\n" : "all scene IO tests passed\n", fails);
    return fails != 0;
}
