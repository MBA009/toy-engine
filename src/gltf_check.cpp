// Loads each glTF/GLB given on the command line and prints what the engine would see: counts, texture sizes, warnings.
// g++ -std=c++17 -O2 -I. -Iexternal/cgltf -Iexternal/stb gltf_check.cpp gltf_loader.cpp -o gltf_check  &&  ./gltf_check assets/*.glb
#include <chrono>
#include <cstdio>
#include "gltf_loader.h"
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        MeshData m; std::string e;
        auto t0 = std::chrono::steady_clock::now();
        bool ok = loadGltf(argv[i], m, e);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s: %s %.0f ms\n", argv[i], ok ? "ok" : "FAILED", ms);
        if (!ok) { std::printf("  error: %s\n", e.c_str()); continue; }
        std::printf("  %zu verts, %zu tris, %zu materials, %zu submeshes, %zu images", m.vertices.size(), m.indices.size() / 3, m.materials.size(), m.submeshes.size(), m.images.size());
        for (auto& im : m.images) std::printf(" [%dx%d]", im.width, im.height);
        std::printf("\n");
        for (auto& w : m.warnings) std::printf("  warning: %s\n", w.c_str());
    }
}
