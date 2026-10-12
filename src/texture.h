#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "math3d.h"
#include <string>
#include <cstdio>
#include <algorithm>

struct Texture {
    int width = 0, height = 0, channels = 0;
    unsigned char* data = nullptr;
    bool ownsData = true;

    bool load(const std::string& path) {
        data = stbi_load(path.c_str(), &width, &height, &channels, 4);
        ownsData = true;
        if (!data) { printf("Failed to load texture: %s\n", path.c_str()); return false; }
        printf("Loaded texture %s (%dx%d)\n", path.c_str(), width, height);
        return true;
    }

    static Texture solidWhite() {
        static unsigned char whitePixel[4] = { 255, 255, 255, 255 };
        Texture t;
        t.data = whitePixel;
        t.width = 1;
        t.height = 1;
        t.ownsData = false;
        return t;
    }

    ~Texture() {
        if (data && ownsData) stbi_image_free(data);
    }

    Vec3 sample(float u, float v) const {
        if (!data) return { 1, 0, 1 }; // magenta = missing texture

        u = u - std::floor(u);
        v = v - std::floor(v);

        int x = std::min((int)(u * width), width - 1);
        int y = std::min((int)((1.0f - v) * height), height - 1);

        int idx = (y * width + x) * 4;
        return {
            data[idx + 0] / 255.0f,
            data[idx + 1] / 255.0f,
            data[idx + 2] / 255.0f
        };
    }

    Vec3 sampleBilinear(float u, float v) const {
    if (!data) return { 1, 0, 1 };

    u = u - std::floor(u);
    v = v - std::floor(v);

    float fx = u * width - 0.5f;
    float fy = (1.0f - v) * height - 0.5f;

    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = fx - x0;
    float ty = fy - y0;

    auto clampX = [&](int x) { return std::max(0, std::min(x, width - 1)); };
    auto clampY = [&](int y) { return std::max(0, std::min(y, height - 1)); };

    auto fetch = [&](int x, int y) -> Vec3 {
        int idx = (clampY(y) * width + clampX(x)) * 4;
        return { data[idx] / 255.0f, data[idx + 1] / 255.0f, data[idx + 2] / 255.0f };
    };

    Vec3 c00 = fetch(x0, y0), c10 = fetch(x1, y0);
    Vec3 c01 = fetch(x0, y1), c11 = fetch(x1, y1);

    Vec3 top = c00 * (1 - tx) + c10 * tx;
    Vec3 bottom = c01 * (1 - tx) + c11 * tx;
    return top * (1 - ty) + bottom * ty;
    }
};