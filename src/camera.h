#pragma once
#include <algorithm>
#include <cmath>
#include "math3d.h"

// A free-fly camera: a position plus two angles.
//   yaw   = rotation around the world Y axis (0 = looking down -Z, positive = turning right)
//   pitch = looking up (+) or down (-), clamped so the view can't flip over
struct FlyCamera {
    Vec3  position{ 0.0f, 0.8f, 1.0f };
    float yaw   = 0.0f;
    float pitch = -0.268f;      // atan2(-1.1, 4): looks at the same spot the old fixed camera did

    float moveSpeed = 2.0f;     // world units per second
    float lookSensitivity = 0.0015f;   // radians pe  
    Vec3 forward() const {
        float cp = std::cos(pitch);
        return { std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp };
    }
    Vec3 right() const { return { std::cos(yaw), 0.0f, std::sin(yaw) }; }

    // dxPixels > 0 = mouse moved right, dyPixels > 0 = mouse moved down
    void look(float dxPixels, float dyPixels) {
        yaw   += dxPixels * lookSensitivity;
        pitch -= dyPixels * lookSensitivity;
        pitch = std::clamp(pitch, -1.55f, 1.55f);
    }

    // forwardAmt / rightAmt / upAmt are in [-1, 1] (from the keys held)
    void move(float forwardAmt, float rightAmt, float upAmt, float speedScale, float dt) {
        Vec3 f = forward(), r = right();
        float d = moveSpeed * speedScale * dt;
        position.x += (f.x * forwardAmt + r.x * rightAmt) * d;
        position.y += (f.y * forwardAmt + upAmt) * d;
        position.z += (f.z * forwardAmt + r.z * rightAmt) * d;
    }

    Mat4 view() const {
        Vec3 f = forward();
        Vec3 target = { position.x + f.x, position.y + f.y, position.z + f.z };
        return Mat4::lookAt(position, target, { 0.0f, 1.0f, 0.0f });
    }
};
