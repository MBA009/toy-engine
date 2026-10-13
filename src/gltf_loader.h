#pragma once
#include <string>
#include "mesh_data.h"

struct GltfImportOptions {
    // Rescale and recenter so the model's largest dimension is 1 and its bounding box is centred
    // on the origin: the same footprint as the built-in unit cube, so Transform::scale then means
    // "size in world units" whatever units the file was authored in.
    bool normalize = true;
};

// Loads a .gltf (with .bin / embedded data) or .glb file into ONE mesh: every triangle primitive of
// every node, with each node's world transform baked into the vertices.
// Handles: positions, normals (generated when the file has none), first UV set, indexed or
// non-indexed triangles, base color factor (written to the vertex color), base color textures
// (PNG/JPEG, embedded or external) and one submesh per material.
// Ignores: sampler settings, UV transforms, a second UV set, normal/metallic/emissive maps, alpha
// modes, skins, animations, cameras, lights, non-triangle primitives.
// Returns false and fills `error` on failure.
bool loadGltf(const std::string& path, MeshData& out, std::string& error,
              const GltfImportOptions& options = GltfImportOptions{});
