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
    Vec3 normal;
    float u, v;
};

// A floor quad (two triangles) receding away from the camera into the distance.
// UVs span 0-1 across the whole floor, so the checkerboard pattern makes warping obvious.
Vertex floorQuad[6] = {
    // Triangle 1
    { { -1.0f, -0.5f, -0.5f }, {1,1,1}, {0,1,0}, 0.0f, 0.0f },
    { { -1.0f, -0.5f, -5.0f }, {1,1,1}, {0,1,0}, 0.0f, 1.0f },
    { {  1.0f, -0.5f, -5.0f }, {1,1,1}, {0,1,0}, 1.0f, 1.0f },
    // Triangle 2
    { { -1.0f, -0.5f, -0.5f }, {1,1,1}, {0,1,0}, 0.0f, 0.0f },
    { {  1.0f, -0.5f, -5.0f }, {1,1,1}, {0,1,0}, 1.0f, 1.0f },
    { {  1.0f, -0.5f, -0.5f }, {1,1,1}, {0,1,0}, 1.0f, 0.0f },
};

Vertex cube[36] = {
    // Front face (+Z)
    { {-0.5f,-0.5f, 0.5f}, {1,1,1}, {0,0,1}, 0.0f,0.0f },
    { { 0.5f,-0.5f, 0.5f}, {1,1,1}, {0,0,1}, 1.0f,0.0f },
    { { 0.5f, 0.5f, 0.5f}, {1,1,1}, {0,0,1}, 1.0f,1.0f },
    { {-0.5f,-0.5f, 0.5f}, {1,1,1}, {0,0,1}, 0.0f,0.0f },
    { { 0.5f, 0.5f, 0.5f}, {1,1,1}, {0,0,1}, 1.0f,1.0f },
    { {-0.5f, 0.5f, 0.5f}, {1,1,1}, {0,0,1}, 0.0f,1.0f },

    // Back face (-Z)
    { { 0.5f,-0.5f,-0.5f}, {1,1,1}, {0,0,-1}, 0.0f,0.0f },
    { {-0.5f,-0.5f,-0.5f}, {1,1,1}, {0,0,-1}, 1.0f,0.0f },
    { {-0.5f, 0.5f,-0.5f}, {1,1,1}, {0,0,-1}, 1.0f,1.0f },
    { { 0.5f,-0.5f,-0.5f}, {1,1,1}, {0,0,-1}, 0.0f,0.0f },
    { {-0.5f, 0.5f,-0.5f}, {1,1,1}, {0,0,-1}, 1.0f,1.0f },
    { { 0.5f, 0.5f,-0.5f}, {1,1,1}, {0,0,-1}, 0.0f,1.0f },

    // Right face (+X)
    { { 0.5f,-0.5f, 0.5f}, {1,1,1}, {1,0,0}, 0.0f,0.0f },
    { { 0.5f,-0.5f,-0.5f}, {1,1,1}, {1,0,0}, 1.0f,0.0f },
    { { 0.5f, 0.5f,-0.5f}, {1,1,1}, {1,0,0}, 1.0f,1.0f },
    { { 0.5f,-0.5f, 0.5f}, {1,1,1}, {1,0,0}, 0.0f,0.0f },
    { { 0.5f, 0.5f,-0.5f}, {1,1,1}, {1,0,0}, 1.0f,1.0f },
    { { 0.5f, 0.5f, 0.5f}, {1,1,1}, {1,0,0}, 0.0f,1.0f },

    // Left face (-X)
    { {-0.5f,-0.5f,-0.5f}, {1,1,1}, {-1,0,0}, 0.0f,0.0f },
    { {-0.5f,-0.5f, 0.5f}, {1,1,1}, {-1,0,0}, 1.0f,0.0f },
    { {-0.5f, 0.5f, 0.5f}, {1,1,1}, {-1,0,0}, 1.0f,1.0f },
    { {-0.5f,-0.5f,-0.5f}, {1,1,1}, {-1,0,0}, 0.0f,0.0f },
    { {-0.5f, 0.5f, 0.5f}, {1,1,1}, {-1,0,0}, 1.0f,1.0f },
    { {-0.5f, 0.5f,-0.5f}, {1,1,1}, {-1,0,0}, 0.0f,1.0f },

    // Top face (+Y)
    { {-0.5f, 0.5f, 0.5f}, {1,1,1}, {0,1,0}, 0.0f,0.0f },
    { { 0.5f, 0.5f, 0.5f}, {1,1,1}, {0,1,0}, 1.0f,0.0f },
    { { 0.5f, 0.5f,-0.5f}, {1,1,1}, {0,1,0}, 1.0f,1.0f },
    { {-0.5f, 0.5f, 0.5f}, {1,1,1}, {0,1,0}, 0.0f,0.0f },
    { { 0.5f, 0.5f,-0.5f}, {1,1,1}, {0,1,0}, 1.0f,1.0f },
    { {-0.5f, 0.5f,-0.5f}, {1,1,1}, {0,1,0}, 0.0f,1.0f },

    // Bottom face (-Y)
    { {-0.5f,-0.5f,-0.5f}, {1,1,1}, {0,-1,0}, 0.0f,0.0f },
    { { 0.5f,-0.5f,-0.5f}, {1,1,1}, {0,-1,0}, 1.0f,0.0f },
    { { 0.5f,-0.5f, 0.5f}, {1,1,1}, {0,-1,0}, 1.0f,1.0f },
    { {-0.5f,-0.5f,-0.5f}, {1,1,1}, {0,-1,0}, 0.0f,0.0f },
    { { 0.5f,-0.5f, 0.5f}, {1,1,1}, {0,-1,0}, 1.0f,1.0f },
    { {-0.5f,-0.5f, 0.5f}, {1,1,1}, {0,-1,0}, 0.0f,1.0f },
};


struct Light {
    Vec3 direction; // for a directional light, pointing FROM the light TO the scene
    Vec3 color;
    float intensity;
};

Light mainLight = {
    { -0.4f, -1.0f, -0.3f }, // direction (will be normalized)
    { 1.0f, 1.0f, 0.95f },   // slightly warm white
    1.2f
};

Vec3 cameraPos = { 0, 0.8f, 1.0f }; // matches your lookAt eye position from Step 1.6


static Texture solidWhite() {
    Texture t;
    static unsigned char whitePixel[4] = { 255, 255, 255, 255 };
    t.data = whitePixel;
    t.width = 1;
    t.height = 1;
    return t;
}

Vec3 sampleCheckerboard(float u, float v) {
    int checkSize = 8;
    int cx = (int)(u * checkSize);
    int cy = (int)(v * checkSize);
    bool isWhite = ((cx + cy) % 2) == 0;
    return isWhite ? Vec3{1, 1, 1} : Vec3{0.1f, 0.1f, 0.1f};
}


void rasterizeTriangle(Vertex tri[3], const Mat4& mvp, const Mat4& model,
                        std::vector<uint32_t>& framebuffer,
                        std::vector<float>& depthBuffer,
                        int width, int height,
                        const Texture& texture,
                        const Light& light,
                        Vec3 cameraPos,
                        bool perspectiveCorrect = true) {
    struct ScreenVertex { float x, y, depth, invW; Vec3 color, normal, worldPos; float u, v; };
    ScreenVertex sv[3];

    for (int i = 0; i < 3; i++) {
        Vec4 worldPos4 = model * Vec4(tri[i].position, 1.0f);
        Vec4 clip = mvp * Vec4(tri[i].position, 1.0f);
        Vec3 ndc = { clip.x / clip.w, clip.y / clip.w, clip.z / clip.w };
        sv[i].x = (ndc.x * 0.5f + 0.5f) * width;
        sv[i].y = (1.0f - (ndc.y * 0.5f + 0.5f)) * height;
        sv[i].depth = ndc.z;
        sv[i].invW = 1.0f / clip.w;
        sv[i].color = tri[i].color;
        sv[i].normal = tri[i].normal;
        sv[i].worldPos = worldPos4.xyz();
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

            float b0 = w0 / area, b1 = w1 / area, b2 = w2 / area;
            float depth = b0 * sv[0].depth + b1 * sv[1].depth + b2 * sv[2].depth;

            int idx = y * width + x;
            if (depth >= depthBuffer[idx]) continue;

            float interpInvW = b0 * sv[0].invW + b1 * sv[1].invW + b2 * sv[2].invW;

            auto perspInterp = [&](Vec3 a0, Vec3 a1, Vec3 a2) {
                Vec3 sum = (a0 * sv[0].invW) * b0 + (a1 * sv[1].invW) * b1 + (a2 * sv[2].invW) * b2;
                return sum * (1.0f / interpInvW);
            };

            float uOverW = b0 * (sv[0].u * sv[0].invW) + b1 * (sv[1].u * sv[1].invW) + b2 * (sv[2].u * sv[2].invW);
            float vOverW = b0 * (sv[0].v * sv[0].invW) + b1 * (sv[1].v * sv[1].invW) + b2 * (sv[2].v * sv[2].invW);
            float u = uOverW / interpInvW;
            float v = vOverW / interpInvW;

            Vec3 normal = perspInterp(sv[0].normal, sv[1].normal, sv[2].normal).normalized();
            Vec3 worldPos = perspInterp(sv[0].worldPos, sv[1].worldPos, sv[2].worldPos);
            Vec3 baseColor = perspInterp(sv[0].color, sv[1].color, sv[2].color);

            depthBuffer[idx] = depth;

            // --- Blinn-Phong ---
            Vec3 N = normal;
            Vec3 L = (light.direction * -1.0f).normalized(); // vector FROM surface TO light
            Vec3 V = (cameraPos - worldPos).normalized();     // vector FROM surface TO camera
            Vec3 H = (L + V).normalized();                    // halfway vector

            float ambientStrength = 0.15f;
            float diffuseTerm = std::max(N.dot(L), 0.0f);
            float specStrength = 0.5f;
            float shininess = 32.0f;
            float specTerm = std::pow(std::max(N.dot(H), 0.0f), shininess) * specStrength;

            Vec3 ambient = light.color * ambientStrength;
            Vec3 diffuse = light.color * (diffuseTerm * light.intensity);
            Vec3 specular = light.color * (specTerm * light.intensity);

            Vec3 texColor = texture.sample(u, v);
            Vec3 albedo = { baseColor.x * texColor.x, baseColor.y * texColor.y, baseColor.z * texColor.z };

            Vec3 litColor = {
                albedo.x * (ambient.x + diffuse.x) + specular.x,
                albedo.y * (ambient.y + diffuse.y) + specular.y,
                albedo.z * (ambient.z + diffuse.z) + specular.z
            };

            uint8_t r = (uint8_t)(std::min(litColor.x, 1.0f) * 255);
            uint8_t g = (uint8_t)(std::min(litColor.y, 1.0f) * 255);
            uint8_t b = (uint8_t)(std::min(litColor.z, 1.0f) * 255);
            framebuffer[idx] = 0xFF000000 | (b << 16) | (g << 8) | r;
        }
    }
}





int main() {

    // runMathTests();

    Texture whiteTexture = Texture::solidWhite();

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
    Mat4 cubeBasePosition = Mat4::translate({0.0f, 0.0f, -1.5f});


    Texture floorTexture;
    if (!floorTexture.load("assets/floor.png")) {
        return -1;
    }
    while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    float time = (float)glfwGetTime();

    std::fill(framebuffer.begin(), framebuffer.end(), 0xFF302010);
    std::fill(depthBuffer.begin(), depthBuffer.end(), 1.0f);

    // Floor triangles, pulled from floorQuad
    Vertex tri1[3] = { floorQuad[0], floorQuad[1], floorQuad[2] };
    Vertex tri2[3] = { floorQuad[3], floorQuad[4], floorQuad[5] };

    // Floor stays static
    Mat4 floorModel = Mat4::identity();
    Mat4 floorMvp = proj * view * floorModel;
    rasterizeTriangle(tri1, floorMvp, floorModel, framebuffer, depthBuffer, WIDTH, HEIGHT, floorTexture, mainLight, cameraPos, true);
    rasterizeTriangle(tri2, floorMvp, floorModel, framebuffer, depthBuffer, WIDTH, HEIGHT, floorTexture, mainLight, cameraPos, true);

    // Cube rotates continuously around Y, offset to sit on the floor
    Mat4 cubeModel = cubeBasePosition * Mat4::rotateY(time);
    Mat4 cubeMvp = proj * view * cubeModel;

    for (int i = 0; i < 12; i++) {
        Vertex cubeTri[3] = { cube[i*3 + 0], cube[i*3 + 1], cube[i*3 + 2] };
        rasterizeTriangle(cubeTri, cubeMvp, cubeModel, framebuffer, depthBuffer, WIDTH, HEIGHT, whiteTexture, mainLight, cameraPos, true);
    }

    glPixelZoom(1.0f, -1.0f);
    glRasterPos2f(-1.0f, 1.0f);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer.data());
    glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}