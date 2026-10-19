#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "math3d.h"

// One vertex as the renderer's vertex buffer holds it (must match the pipeline's attribute layout)
struct MeshVertex {
    Vec3 pos;
    Vec3 color;     // sRGB-authored base color (the fragment shader linearizes it)
    Vec3 normal;
    float u, v;
};

struct Aabb {
    Vec3 min{ -0.5f, -0.5f, -0.5f };
    Vec3 max{  0.5f,  0.5f,  0.5f };
};

// A decoded image: 8-bit RGBA, top row first (glTF and Vulkan agree on the UV origin, so no flip)
struct ImageData {
    std::string name;
    int width = 0, height = 0;
    std::vector<uint8_t> rgba;
};

// What the renderer needs from a glTF material (for now: just the base color texture)
struct MaterialData {
    std::string name;
    int baseColorImage = -1;     // index into MeshData::images, or -1 = untextured (white)
    bool doubleSided = false;    // draw back faces too (leaves, cloth, thin sheets)
    float alphaCutoff = 0.0f;    // > 0: pixels whose texture alpha is below this are discarded (glTF "MASK"; "BLEND" is approximated by 0.5)
};

// A run of indices drawn with one material. A mesh with N materials is drawn as N draw calls.
struct Submesh {
    uint32_t firstIndex = 0, indexCount = 0;   // local to the mesh's index list
    int material = -1;                         // index into MeshData::materials, or -1 = default
};

// A mesh on the CPU, before it is appended to the renderer's big vertex/index buffers.
// Indices are LOCAL to this mesh (0 = its first vertex); the renderer draws it with a vertex offset.
struct MeshData {
    std::string name;
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
    Aabb bounds;
    std::vector<ImageData> images;
    std::vector<MaterialData> materials;
    std::vector<Submesh> submeshes;            // covers all of `indices`, in order
    std::vector<std::string> dependencies;     // every file the import read (the model, its .bin, external images): what hot reload watches
    std::vector<std::string> warnings;         // non-fatal import problems (an image that would not decode...)
};