#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "math_tests.h"
#include "math3d.h"


const int WIDTH = 800;
const int HEIGHT = 600;



struct Vertex {
    Vec3 position;
    Vec3 color;
};

// Triangle A - closer to camera
Vertex triangleA[3] = {
    { {  0.0f,  0.5f, 0.0f }, { 1, 0, 0 } }, // red
    { { -0.5f, -0.5f, 0.0f }, { 1, 0, 0 } },
    { {  0.5f, -0.5f, 0.0f }, { 1, 0, 0 } },
};

// Triangle B - further from camera, offset so it overlaps triangle A
Vertex triangleB[3] = {
    { {  0.3f,  0.3f, -1.0f }, { 0, 0, 1 } }, // blue, pushed back on Z
    { { -0.2f, -0.7f, -1.0f }, { 0, 0, 1 } },
    { {  0.8f, -0.7f, -1.0f }, { 0, 0, 1 } },
};

void rasterizeTriangle(Vertex tri[3], const Mat4& mvp,
                        std::vector<uint32_t>& framebuffer,
                        std::vector<float>& depthBuffer,
                        int width, int height) {
    struct ScreenVertex { float x, y, depth; Vec3 color; };
    ScreenVertex sv[3];

    for (int i = 0; i < 3; i++) {
        Vec4 clip = mvp * Vec4(tri[i].position, 1.0f);
        Vec3 ndc = { clip.x / clip.w, clip.y / clip.w, clip.z / clip.w };
        sv[i].x = (ndc.x * 0.5f + 0.5f) * width;
        sv[i].y = (1.0f - (ndc.y * 0.5f + 0.5f)) * height;
        sv[i].depth = ndc.z; // NDC depth, roughly [-1, 1] -> we'll compare this directly
        sv[i].color = tri[i].color;
    }

    int minX = std::max(0, (int)std::floor(std::min({sv[0].x, sv[1].x, sv[2].x})));
    int maxX = std::min(width - 1, (int)std::ceil(std::max({sv[0].x, sv[1].x, sv[2].x})));
    int minY = std::max(0, (int)std::floor(std::min({sv[0].y, sv[1].y, sv[2].y})));
    int maxY = std::min(height - 1, (int)std::ceil(std::max({sv[0].y, sv[1].y, sv[2].y})));

    auto edgeFunction = [](float ax, float ay, float bx, float by, float px, float py) {
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    };

    // Precompute the full-triangle area for barycentric normalization
    float area = edgeFunction(sv[0].x, sv[0].y, sv[1].x, sv[1].y, sv[2].x, sv[2].y);
    if (std::fabs(area) < 1e-6f) return; // degenerate triangle, skip

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float px = x + 0.5f, py = y + 0.5f;

            float w0 = edgeFunction(sv[1].x, sv[1].y, sv[2].x, sv[2].y, px, py);
            float w1 = edgeFunction(sv[2].x, sv[2].y, sv[0].x, sv[0].y, px, py);
            float w2 = edgeFunction(sv[0].x, sv[0].y, sv[1].x, sv[1].y, px, py);

            bool inside = (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);
            if (!inside) continue;

            // Normalize edge weights into barycentric coordinates
            float b0 = w0 / area;
            float b1 = w1 / area;
            float b2 = w2 / area;

            // Interpolate depth
            float depth = b0 * sv[0].depth + b1 * sv[1].depth + b2 * sv[2].depth;

            int idx = y * width + x;

            // Depth test: smaller NDC z = closer to camera. Only draw if closer than what's there.
            if (depth < depthBuffer[idx]) {
                depthBuffer[idx] = depth;

                Vec3 color = sv[0].color * b0 + sv[1].color * b1 + sv[2].color * b2;
                uint8_t r = (uint8_t)(color.x * 255);
                uint8_t g = (uint8_t)(color.y * 255);
                uint8_t b = (uint8_t)(color.z * 255);
                framebuffer[idx] = 0xFF000000 | (b << 16) | (g << 8) | r; // 0xAABBGGRR
            }
        }
    }
}



int main() {

    // runMathTests();

    

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Toy Engine", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Our CPU-side framebuffer: one uint32_t per pixel, packed as 0xAABBGGRR
    std::vector<uint32_t> framebuffer(WIDTH * HEIGHT);
    std::vector<float> depthBuffer(WIDTH * HEIGHT);

    Mat4 model = Mat4::identity(); // no transform yet, triangle stays at origin
    Mat4 view = Mat4::lookAt({0, 0, 3}, {0, 0, 0}, {0, 1, 0});
    Mat4 proj = Mat4::perspective(3.14159265f / 4.0f, (float)WIDTH / HEIGHT, 0.1f, 100.0f);
    Mat4 mvp = proj * view * model;



    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Clear color buffer
        uint32_t clearColor = 0xFF302010;
        std::fill(framebuffer.begin(), framebuffer.end(), clearColor);

        // Clear depth buffer to the far plane (NDC z = 1.0 is "farthest")
        std::fill(depthBuffer.begin(), depthBuffer.end(), 1.0f);

        // Draw both triangles - order doesn't matter now, depth test handles it
        rasterizeTriangle(triangleA, mvp, framebuffer, depthBuffer, WIDTH, HEIGHT);
        rasterizeTriangle(triangleB, mvp, framebuffer, depthBuffer, WIDTH, HEIGHT);

        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer.data());
        glfwSwapBuffers(window);
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}