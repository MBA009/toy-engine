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

Vertex triangle[3] = {
    { {  0.0f,  0.5f, 0.0f }, { 1, 0, 0 } }, // top, red
    { { -0.5f, -0.5f, 0.0f }, { 0, 1, 0 } }, // bottom-left, green
    { {  0.5f, -0.5f, 0.0f }, { 0, 0, 1 } }, // bottom-right, blue
};



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

    Mat4 model = Mat4::identity(); // no transform yet, triangle stays at origin
    Mat4 view = Mat4::lookAt({0, 0, 3}, {0, 0, 0}, {0, 1, 0});
    Mat4 proj = Mat4::perspective(3.14159265f / 4.0f, (float)WIDTH / HEIGHT, 0.1f, 100.0f);
    Mat4 mvp = proj * view * model;

    // Sanity check: transform each vertex and print screen-space coords
    for (int i = 0; i < 3; i++) {
        Vec4 clip = mvp * Vec4(triangle[i].position, 1.0f);
        // Perspective divide: clip space -> NDC
        Vec3 ndc = { clip.x / clip.w, clip.y / clip.w, clip.z / clip.w };
        // NDC [-1,1] -> screen space [0, WIDTH]/[0, HEIGHT]
        float screenX = (ndc.x * 0.5f + 0.5f) * WIDTH;
        float screenY = (1.0f - (ndc.y * 0.5f + 0.5f)) * HEIGHT; // flip Y: screen origin is top-left
        printf("Vertex %d: screen = (%.1f, %.1f)\n", i, screenX, screenY);
    }

while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Clear
    uint32_t clearColor = 0xFF302010;
    std::fill(framebuffer.begin(), framebuffer.end(), clearColor);

    // Transform vertices to screen space
    struct ScreenVertex { float x, y; Vec3 color; };
    ScreenVertex sv[3];
    for (int i = 0; i < 3; i++) {
        Vec4 clip = mvp * Vec4(triangle[i].position, 1.0f);
        Vec3 ndc = { clip.x / clip.w, clip.y / clip.w, clip.z / clip.w };
        sv[i].x = (ndc.x * 0.5f + 0.5f) * WIDTH;
        sv[i].y = (1.0f - (ndc.y * 0.5f + 0.5f)) * HEIGHT;
        sv[i].color = triangle[i].color;
    }

    // Bounding box of the triangle, clamped to screen bounds
    int minX = std::max(0, (int)std::floor(std::min({sv[0].x, sv[1].x, sv[2].x})));
    int maxX = std::min(WIDTH - 1, (int)std::ceil(std::max({sv[0].x, sv[1].x, sv[2].x})));
    int minY = std::max(0, (int)std::floor(std::min({sv[0].y, sv[1].y, sv[2].y})));
    int maxY = std::min(HEIGHT - 1, (int)std::ceil(std::max({sv[0].y, sv[1].y, sv[2].y})));

    // Edge function: positive if point is to the "left" of the edge a->b
    auto edgeFunction = [](float ax, float ay, float bx, float by, float px, float py) {
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    };

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float px = x + 0.5f, py = y + 0.5f; // sample at pixel center

            float w0 = edgeFunction(sv[1].x, sv[1].y, sv[2].x, sv[2].y, px, py);
            float w1 = edgeFunction(sv[2].x, sv[2].y, sv[0].x, sv[0].y, px, py);
            float w2 = edgeFunction(sv[0].x, sv[0].y, sv[1].x, sv[1].y, px, py);

            // Inside test: all three edge values must share the same sign
            bool inside = (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);

            if (inside) {
                // Flat white fill for now - color interpolation comes in Step 1.6
                uint32_t white = 0xFFFFFFFF;
                framebuffer[y * WIDTH + x] = white;
            }
        }
    }

    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer.data());
    glfwSwapBuffers(window);  

    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}