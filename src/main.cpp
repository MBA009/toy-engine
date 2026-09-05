#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "math_tests.h"
#include "math3d.h"
#include "texture.h"


const int WIDTH = 800;
const int HEIGHT = 600;



struct Vertex {
    Vec3 position;
    Vec3 color;
    float u, v;
};

// A floor quad (two triangles) receding away from the camera into the distance.
// UVs span 0-1 across the whole floor, so the checkerboard pattern makes warping obvious.
Vertex floorQuad[6] = {
    // Triangle 1
    { { -1.0f, -0.5f, -0.5f }, {1,1,1}, 0.0f, 0.0f },
    { { -1.0f, -0.5f, -5.0f }, {1,1,1}, 0.0f, 1.0f },
    { {  1.0f, -0.5f, -5.0f }, {1,1,1}, 1.0f, 1.0f },
    // Triangle 2
    { { -1.0f, -0.5f, -0.5f }, {1,1,1}, 0.0f, 0.0f },
    { {  1.0f, -0.5f, -5.0f }, {1,1,1}, 1.0f, 1.0f },
    { {  1.0f, -0.5f, -0.5f }, {1,1,1}, 1.0f, 0.0f },
};

Vec3 sampleCheckerboard(float u, float v) {
    int checkSize = 8;
    int cx = (int)(u * checkSize);
    int cy = (int)(v * checkSize);
    bool isWhite = ((cx + cy) % 2) == 0;
    return isWhite ? Vec3{1, 1, 1} : Vec3{0.1f, 0.1f, 0.1f};
}


void rasterizeTriangle(Vertex tri[3], const Mat4& mvp,
                        std::vector<uint32_t>& framebuffer,
                        std::vector<float>& depthBuffer,
                        int width, int height,
                        const Texture& texture,
                        bool perspectiveCorrect = true) {
    struct ScreenVertex { float x, y, depth, invW; Vec3 color; float u, v; };
    ScreenVertex sv[3];

    for (int i = 0; i < 3; i++) {
        Vec4 clip = mvp * Vec4(tri[i].position, 1.0f);
        Vec3 ndc = { clip.x / clip.w, clip.y / clip.w, clip.z / clip.w };
        sv[i].x = (ndc.x * 0.5f + 0.5f) * width;
        sv[i].y = (1.0f - (ndc.y * 0.5f + 0.5f)) * height;
        sv[i].depth = ndc.z;
        sv[i].invW = 1.0f / clip.w;   // <-- key value for perspective correction
        sv[i].color = tri[i].color;
        sv[i].u = tri[i].u;
        sv[i].v = tri[i].v;
    }

    int minX = std::max(0, (int)std::floor(std::min({sv[0].x, sv[1].x, sv[2].x})));
    int maxX = std::min(width - 1, (int)std::ceil(std::max({sv[0].x, sv[1].x, sv[2].x})));
    int minY = std::max(0, (int)std::floor(std::min({sv[0].y, sv[1].y, sv[2].y})));
    int maxY = std::min(height - 1, (int)std::ceil(std::max({sv[0].y, sv[1].y, sv[2].y})));

    auto edgeFunction = [](float ax, float ay, float bx, float by, float px, float py) {
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    };

    float area = edgeFunction(sv[0].x, sv[0].y, sv[1].x, sv[1].y, sv[2].x, sv[2].y);
    if (std::fabs(area) < 1e-6f) return;

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float px = x + 0.5f, py = y + 0.5f;

            float w0 = edgeFunction(sv[1].x, sv[1].y, sv[2].x, sv[2].y, px, py);
            float w1 = edgeFunction(sv[2].x, sv[2].y, sv[0].x, sv[0].y, px, py);
            float w2 = edgeFunction(sv[0].x, sv[0].y, sv[1].x, sv[1].y, px, py);

            bool inside = (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);
            if (!inside) continue;

            // Screen-space barycentric weights (correct for interpolating already-projected
            // quantities like screen position and NDC depth, but NOT for UV/color directly).
            float b0 = w0 / area;
            float b1 = w1 / area;
            float b2 = w2 / area;

            float depth = b0 * sv[0].depth + b1 * sv[1].depth + b2 * sv[2].depth;

            int idx = y * width + x;
            if (depth >= depthBuffer[idx]) continue;

            float u, v;
            Vec3 color;

            if (perspectiveCorrect) {
                // Perspective-correct interpolation:
                // interpolate (attribute * invW) linearly in screen space, then divide
                // by the interpolated invW itself.
                float interpInvW = b0 * sv[0].invW + b1 * sv[1].invW + b2 * sv[2].invW;

                float uOverW = b0 * (sv[0].u * sv[0].invW) + b1 * (sv[1].u * sv[1].invW) + b2 * (sv[2].u * sv[2].invW);
                float vOverW = b0 * (sv[0].v * sv[0].invW) + b1 * (sv[1].v * sv[1].invW) + b2 * (sv[2].v * sv[2].invW);
                u = uOverW / interpInvW;
                v = vOverW / interpInvW;

                Vec3 cOverW = (sv[0].color * sv[0].invW) * b0 + (sv[1].color * sv[1].invW) * b1 + (sv[2].color * sv[2].invW) * b2;
                color = cOverW * (1.0f / interpInvW);
            } else {
                // Naive (incorrect) interpolation - straight barycentric average in screen space.
                // Kept here deliberately so you can compare against perspectiveCorrect=true.
                u = b0 * sv[0].u + b1 * sv[1].u + b2 * sv[2].u;
                v = b0 * sv[0].v + b1 * sv[1].v + b2 * sv[2].v;
                color = sv[0].color * b0 + sv[1].color * b1 + sv[2].color * b2;
            }

            depthBuffer[idx] = depth;

            Vec3 checker = texture.sampleBilinear(u, v);
            Vec3 finalColor = { color.x * checker.x, color.y * checker.y, color.z * checker.z };

            uint8_t r = (uint8_t)(std::min(finalColor.x, 1.0f) * 255);
            uint8_t g = (uint8_t)(std::min(finalColor.y, 1.0f) * 255);
            uint8_t b = (uint8_t)(std::min(finalColor.z, 1.0f) * 255);
            framebuffer[idx] = 0xFF000000 | (b << 16) | (g << 8) | r;
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
    Mat4 view = Mat4::lookAt({0, 0.8f, 1.0f}, {0, -0.3f, -3.0f}, {0, 1, 0});
    Mat4 proj = Mat4::perspective(3.14159265f / 4.0f, (float)WIDTH / HEIGHT, 0.1f, 100.0f);
    Mat4 mvp = proj * view * model;


    Texture floorTexture;
    if (!floorTexture.load("assets/floor.png")) {
        return -1;
    }
    while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    std::fill(framebuffer.begin(), framebuffer.end(), 0xFF302010);
    std::fill(depthBuffer.begin(), depthBuffer.end(), 1.0f);

    Vertex tri1[3] = { floorQuad[0], floorQuad[1], floorQuad[2] };
    Vertex tri2[3] = { floorQuad[3], floorQuad[4], floorQuad[5] };

    rasterizeTriangle(tri1, mvp, framebuffer, depthBuffer, WIDTH, HEIGHT, floorTexture, true);
    rasterizeTriangle(tri2, mvp, framebuffer, depthBuffer, WIDTH, HEIGHT, floorTexture, true);

    glPixelZoom(1.0f, -1.0f);      // flip vertical draw direction
    glRasterPos2f(-1.0f, 1.0f);    // start from the top-left corner of the window
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer.data());

    
    glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}