#pragma once
#include "math3d.h"
#include <cmath>
inline constexpr int    BENCH_COUNTS[]      = { 1, 10, 100, 1000, 10000 };
inline constexpr int    BENCH_STAGES        = 5;
inline constexpr double BENCH_WARMUP_SEC    = 1.0;
inline constexpr double BENCH_MEASURE_SEC   = 3.0;

// Where cube i of n sits and how big it is (the grid layout used by every benchmark).
inline void benchCubeParams(int i, int n, float& x, float& y, float& z, float& s) {
    int side = (int)std::ceil(std::sqrt((float)n));
    float cell = 1.6f / side;
    int gx = i % side, gz = i / side;
    x = -0.8f + (gx + 0.5f) * cell;
    z = -0.8f - (gz + 0.5f) * (3.0f / side);
    s = cell * 0.7f;
    y = -0.5f + s * 0.5f;
}

// Full model matrix at time t (used by the software renderer, which has no ECS)
inline Mat4 benchCubeModel(int i, int n, float t) {
    float x, y, z, s;
    benchCubeParams(i, n, x, y, z, s);
    return Mat4::translate({ x, y, z }) * Mat4::rotateY(t + i * 0.1f) * Mat4::scale({ s, s, s });
}
