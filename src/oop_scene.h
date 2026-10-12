#pragma once
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include "components.h"

// ---------------------------------------------------------------------------
// The conventional alternative: an inheritance-based scene of heap-allocated
// game objects with virtual Update()/Render(). It carries the sort of state a
// typical game object has (name, health, flags, velocity...), all stored inline.
// This is the baseline the ECS gets benchmarked against.
// ---------------------------------------------------------------------------

class GameObject {
public:
    GameObject(const Transform& t, std::string n) : transform(t), name(std::move(n)) {}
    virtual ~GameObject() = default;

    virtual void update(float dt, float time) = 0;

    // Per-object integration, as a game object would typically do it (virtual, like update)
    virtual void move(float dt) {
        transform.position.x += velocity.x * dt;
        transform.position.z += velocity.z * dt;
    }

    virtual void render(std::vector<DrawItem>& out) const {
        if (!visible) return;
        out.push_back(DrawItem{ modelMatrix(transform), meshId, materialId });
    }

    Transform transform;
    uint32_t  meshId = 0, materialId = 0;
    std::string name;
    float     health = 100.0f, maxHealth = 100.0f;
    uint32_t  flags = 0;
    Vec3      velocity{ 0.0f, 0.0f, 0.0f };
    void*     userData = nullptr;
    bool      visible = true;
};

class Spinner : public GameObject {
public:
    Spinner(const Transform& t, std::string n, float s)
        : GameObject(t, std::move(n)), speed(s) {}

    void update(float dt, float) override { transform.yaw += speed * dt; }

protected:
    float speed;
};

class BobbingSpinner : public Spinner {
public:
    BobbingSpinner(const Transform& t, std::string n, float s, float amp, float freq, float ph)
        : Spinner(t, std::move(n), s), amplitude(amp), frequency(freq), baseY(t.position.y), phase(ph) {}

    void update(float dt, float time) override {
        Spinner::update(dt, time);
        transform.position.y = baseY + amplitude * std::sin(time * frequency + phase);
    }

private:
    float amplitude, frequency, baseY, phase;
};

class StaticProp : public GameObject {
public:
    StaticProp(const Transform& t, std::string n) : GameObject(t, std::move(n)) {}
    void update(float, float) override {}
};
