#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "components.h"
#include "ecs.h"

// ---------------------------------------------------------------------------
// Scene serialization (Step 5.6). The ECS has no list of "which components exist", so each known
// component type is written explicitly. Meshes are saved by NAME, not by number: mesh ids depend on
// the order things were imported in, so "builtin:cube" or the asset's file path is what survives a restart.
//
// Loading builds the entities into a scratch Registry first; the caller swaps it in only if the whole file
// was valid, so a broken file can never leave the editor half-loaded.
// ---------------------------------------------------------------------------

namespace sceneio {

using json = nlohmann::json;

inline constexpr int FORMAT_VERSION = 1;
inline constexpr const char* FORMAT_NAME = "toy-engine-scene";

// ---- small helpers: validated reads (a bad value throws, which loadScene turns into an error message) ----

inline float num(const json& j) {
    if (!j.is_number()) throw std::runtime_error("expected a number, got " + std::string(j.type_name()));
    float f = j.get<float>();
    if (!std::isfinite(f)) throw std::runtime_error("number is not finite");
    return f;
}

// JSON numbers are doubles, so a float like 0.1f would print as 0.10000000149011612. Pick the shortest
// decimal that still reads back as exactly the same float: files stay readable, and nothing is lost.
inline double f(float x) {
    char buf[32];
    for (int digits = 6; digits <= 9; ++digits) {
        std::snprintf(buf, sizeof buf, "%.*g", digits, (double)x);
        if (std::strtof(buf, nullptr) == x) break;
    }
    return std::strtod(buf, nullptr);
}

inline json toJson(const Vec3& v) { return json::array({ f(v.x), f(v.y), f(v.z) }); }

inline Vec3 toVec3(const json& j) {
    if (!j.is_array() || j.size() != 3) throw std::runtime_error("expected an array of 3 numbers");
    return { num(j[0]), num(j[1]), num(j[2]) };
}

inline float numOr(const json& obj, const char* key, float fallback) {
    auto it = obj.find(key);
    return it == obj.end() ? fallback : num(*it);
}

// ---- entities ----

// meshKey(meshId) -> stable name for the mesh ("builtin:cube", "assets/Duck.glb", ...)
using MeshKeyFn = std::function<std::string(uint32_t)>;
// meshResolve(key) -> mesh id (importing the file if needed), or -1 if it can't be found
using MeshResolveFn = std::function<int(const std::string&)>;

inline json entitiesToJson(Registry& reg, const MeshKeyFn& meshKey) {
    json list = json::array();
    for (size_t id = 0; id < reg.idCapacity(); ++id) {
        Entity e = (Entity)id;
        if (!reg.alive(e)) continue;
        json o = json::object();
        if (reg.has<Name>(e)) o["name"] = reg.get<Name>(e).value;
        if (reg.has<Transform>(e)) {
            const Transform& t = reg.get<Transform>(e);
            o["transform"] = { { "position", toJson(t.position) }, { "yaw", f(t.yaw) }, { "scale", f(t.scale) } };
        }
        if (reg.has<MeshRenderer>(e)) {
            const MeshRenderer& m = reg.get<MeshRenderer>(e);
            o["mesh"] = { { "asset", meshKey(m.meshId) }, { "material", m.materialId } };
        }
        if (reg.has<Spin>(e)) o["spin"] = { { "speed", f(reg.get<Spin>(e).speed) } };
        if (reg.has<Bob>(e)) {
            const Bob& b = reg.get<Bob>(e);
            o["bob"] = { { "amplitude", f(b.amplitude) }, { "frequency", f(b.frequency) }, { "baseY", f(b.baseY) }, { "phase", f(b.phase) } };
        }
        if (reg.has<Velocity>(e)) o["velocity"] = toJson(reg.get<Velocity>(e).value);
        if (reg.has<AIAgent>(e)) {      // the current path is not saved: agents come back idle
            const AIAgent& a = reg.get<AIAgent>(e);
            o["agent"] = { { "speed", f(a.moveSpeed) }, { "turnSpeed", f(a.turnSpeed) }, { "footOffset", f(a.footOffset) } };
        }
        if (reg.has<Health>(e)) o["health"] = { { "current", f(reg.get<Health>(e).current) }, { "max", f(reg.get<Health>(e).max) } };
        if (reg.has<Player>(e)) o["player"] = true;
        if (reg.has<Goal>(e)) o["goal"] = { { "radius", f(reg.get<Goal>(e).radius) } };
        if (reg.has<StartPad>(e)) o["start"] = { { "radius", f(reg.get<StartPad>(e).radius) } };
        if (reg.has<AIBrain>(e)) {
            const AIBrain& b = reg.get<AIBrain>(e);
            json patrol = json::array();
            for (const Vec3& p : b.patrol) patrol.push_back(toJson(p));
            o["brain"] = { { "detectRange", f(b.detectRange) }, { "contactRange", f(b.contactRange) }, { "fleeBelow", f(b.fleeBelow) },
                           { "recoverAt", f(b.recoverAt) }, { "contactDamage", f(b.contactDamage) }, { "attackDamage", f(b.attackDamage) }, { "regenPerSecond", f(b.regenPerSecond) },
                           { "patrolWait", f(b.patrolWait) }, { "patrol", patrol } };
        }
        if (reg.has<NavSource>(e)) o["nav"] = reg.get<NavSource>(e).mode == NAV_GEOMETRY ? "geometry" : "obstacle";
        list.push_back(std::move(o));
    }
    return list;
}

// Appends the entities of `list` to `out` (expected empty). Problems that don't stop the load
// (a mesh that can't be found) are added to `warnings`. Throws std::runtime_error on malformed data.
inline void entitiesFromJson(const json& list, Registry& out, const MeshResolveFn& meshResolve,
                             size_t maxEntities, std::vector<std::string>& warnings) {
    if (!list.is_array()) throw std::runtime_error("'entities' must be an array");
    size_t index = 0;
    for (const json& o : list) {
        const std::string where = "entity " + std::to_string(index++) + ": ";
        try {
            if (!o.is_object()) throw std::runtime_error("expected an object");
            if (out.size() >= maxEntities) {
                warnings.push_back("scene has more than " + std::to_string(maxEntities) + " entities; the rest were skipped");
                return;
            }
            Entity e = out.create();
            if (auto it = o.find("name"); it != o.end()) {
                if (!it->is_string()) throw std::runtime_error("'name' must be a string");
                out.add<Name>(e, Name{ it->get<std::string>() });
            }
            if (auto it = o.find("transform"); it != o.end()) {
                Transform t;
                if (auto p = it->find("position"); p != it->end()) t.position = toVec3(*p);
                t.yaw = numOr(*it, "yaw", 0.0f);
                t.scale = numOr(*it, "scale", 1.0f);
                out.add<Transform>(e, t);
            }
            if (auto it = o.find("mesh"); it != o.end()) {
                MeshRenderer m;
                std::string key = it->value("asset", std::string("builtin:cube"));
                int id = meshResolve(key);
                if (id < 0) {
                    warnings.push_back(where + "mesh '" + key + "' not found, using the cube");
                    id = 0;
                }
                m.meshId = (uint32_t)id;
                m.materialId = it->value("material", 0u);
                out.add<MeshRenderer>(e, m);
            }
            if (auto it = o.find("spin"); it != o.end()) out.add<Spin>(e, Spin{ numOr(*it, "speed", 1.0f) });
            if (auto it = o.find("bob"); it != o.end()) {
                Bob b;
                b.amplitude = numOr(*it, "amplitude", b.amplitude);
                b.frequency = numOr(*it, "frequency", b.frequency);
                b.baseY = numOr(*it, "baseY", b.baseY);
                b.phase = numOr(*it, "phase", b.phase);
                out.add<Bob>(e, b);
            }
            if (auto it = o.find("velocity"); it != o.end()) out.add<Velocity>(e, Velocity{ toVec3(*it) });
            if (auto it = o.find("agent"); it != o.end()) {
                AIAgent a;
                a.moveSpeed = numOr(*it, "speed", a.moveSpeed);
                a.turnSpeed = numOr(*it, "turnSpeed", a.turnSpeed);
                a.footOffset = numOr(*it, "footOffset", a.footOffset);
                out.add<AIAgent>(e, a);
            }
            if (auto it = o.find("health"); it != o.end()) {
                Health h;
                h.max = numOr(*it, "max", h.max);
                h.current = numOr(*it, "current", h.max);
                out.add<Health>(e, h);
            }
            if (auto it = o.find("goal"); it != o.end()) out.add<Goal>(e, Goal{ numOr(*it, "radius", 0.4f) });
            if (auto it = o.find("start"); it != o.end()) out.add<StartPad>(e, StartPad{ numOr(*it, "radius", 0.4f) });
            if (auto it = o.find("player"); it != o.end() && it->is_boolean() && it->get<bool>()) out.add<Player>(e, Player{});
            if (auto it = o.find("brain"); it != o.end()) {
                AIBrain br;
                br.detectRange = numOr(*it, "detectRange", br.detectRange);
                br.contactRange = numOr(*it, "contactRange", br.contactRange);
                br.fleeBelow = numOr(*it, "fleeBelow", br.fleeBelow);
                br.recoverAt = numOr(*it, "recoverAt", br.recoverAt);
                br.contactDamage = numOr(*it, "contactDamage", br.contactDamage);
                br.attackDamage = numOr(*it, "attackDamage", br.attackDamage);
                br.regenPerSecond = numOr(*it, "regenPerSecond", br.regenPerSecond);
                br.patrolWait = numOr(*it, "patrolWait", br.patrolWait);
                if (auto p = it->find("patrol"); p != it->end() && p->is_array())
                    for (const auto& v : *p) br.patrol.push_back(toVec3(v));
                out.add<AIBrain>(e, br);
            }
            if (auto it = o.find("nav"); it != o.end()) {
                const std::string mode = it->is_string() ? it->get<std::string>() : std::string();
                if (mode != "obstacle" && mode != "geometry") throw std::runtime_error("'nav' must be \"obstacle\" or \"geometry\"");
                out.add<NavSource>(e, NavSource{ mode == "geometry" ? (uint8_t)NAV_GEOMETRY : (uint8_t)NAV_OBSTACLE });
            }
        } catch (const json::exception& ex) {
            throw std::runtime_error(where + ex.what());
        } catch (const std::runtime_error& ex) {
            std::string msg = ex.what();
            if (msg.rfind("entity ", 0) == 0) throw;
            throw std::runtime_error(where + msg);
        }
    }
}

// ---- files ----

// Writes to "<path>.tmp" then renames over the target, so a crash mid-save can't destroy the previous file.
inline bool writeSceneFile(const std::string& path, const json& doc, std::string& error) {
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        fs::path tmp = p;
        tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) { error = "cannot write '" + tmp.string() + "'"; return false; }
            f << doc.dump(2) << '\n';
            f.flush();
            if (!f) { error = "write failed (disk full?)"; return false; }
        }
        std::error_code ec;
        fs::rename(tmp, p, ec);
        if (ec) { error = "cannot replace '" + path + "': " + ec.message(); return false; }
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}

// Reads and checks the header (format name + version); the caller reads the sections it wants.
inline bool readSceneFile(const std::string& path, json& doc, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "cannot open '" + path + "'"; return false; }
    try {
        doc = json::parse(f);
    } catch (const json::exception& ex) {
        error = std::string("not valid JSON: ") + ex.what();
        return false;
    }
    if (!doc.is_object() || doc.value("format", std::string()) != FORMAT_NAME) {
        error = "not a Toy Engine scene file";
        return false;
    }
    int version = doc.value("version", 0);
    if (version != FORMAT_VERSION) {
        error = "scene version " + std::to_string(version) + " is not supported (expected " + std::to_string(FORMAT_VERSION) + ")";
        return false;
    }
    return true;
}

inline json newSceneDoc() {
    json doc = json::object();
    doc["format"] = FORMAT_NAME;
    doc["version"] = FORMAT_VERSION;
    return doc;
}

}  // namespace sceneio
