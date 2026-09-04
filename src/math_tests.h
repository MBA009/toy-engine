#pragma once
#include "math3d.h"
#include <cstdio>
#include <cassert>

inline void runMathTests() {
    // Test 1: identity matrix leaves a point unchanged
    {
        Vec4 p(3.0f, -2.0f, 5.0f, 1.0f);
        Vec4 result = Mat4::identity() * p;
        assert(std::fabs(result.x - p.x) < 1e-5f);
        assert(std::fabs(result.y - p.y) < 1e-5f);
        assert(std::fabs(result.z - p.z) < 1e-5f);
        printf("[PASS] Identity matrix leaves point unchanged\n");
    }

    // Test 2: rotate (1,0,0) by 90 degrees around Y -> should land near (0,0,-1)
    {
        Vec4 p(1.0f, 0.0f, 0.0f, 1.0f);
        Mat4 rot = Mat4::rotateY(3.14159265f / 2.0f); // 90 degrees
        Vec4 result = rot * p;
        printf("[INFO] Rotated (1,0,0) by 90deg Y -> (%.4f, %.4f, %.4f)\n",
               result.x, result.y, result.z);
        assert(std::fabs(result.x - 0.0f) < 1e-4f);
        assert(std::fabs(result.z - (-1.0f)) < 1e-4f);
        printf("[PASS] 90-degree Y rotation matches hand calculation\n");
    }

    // Test 3: matrix * its inverse = identity
    {
        Mat4 m = Mat4::translate({1,2,3}) * Mat4::rotateY(0.7f);
        Mat4 inv = m.inverse();
        Mat4 result = m * inv;
        Mat4 id = Mat4::identity();
        bool ok = true;
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                if (std::fabs(result.m[c][r] - id.m[c][r]) > 1e-3f) ok = false;
        assert(ok);
        printf("[PASS] Matrix * inverse == identity\n");
    }

    printf("All math tests passed.\n\n");
}