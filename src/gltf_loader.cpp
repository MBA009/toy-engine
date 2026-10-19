#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

#include "gltf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {

// glTF stores colors as linear values. Our vertex colors are authored as sRGB (the shader applies
// pow(color, 2.2) to linearize them), so convert on the way in; the round trip then lands back on
// the linear value the artist chose.
float linearToSrgbApprox(float c) {
    return std::pow(std::clamp(c, 0.0f, 1.0f), 1.0f / 2.2f);
}

// cgltf gives node matrices as column-major float[16], which is exactly how Mat4 stores m[col][row].
Mat4 toMat4(const float* m16) {
    Mat4 r;
    std::memcpy(&r.m[0][0], m16, sizeof(float) * 16);
    return r;
}

Vec3 transformPoint(const Mat4& m, const Vec3& p) {
    return (m * Vec4(p, 1.0f)).xyz();
}

// Gets the encoded bytes of a glTF image (embedded in the .glb / a buffer view, a base64 data: URI,
// or a file next to the model) and decodes them to RGBA8.
bool decodeImage(const cgltf_options& opts, const cgltf_image* img, const std::filesystem::path& baseDir,
                 ImageData& out, std::string& error, std::vector<std::string>* dependencies) {
    std::vector<uint8_t> file;
    const uint8_t* bytes = nullptr;
    size_t size = 0;
    void* b64 = nullptr;

    if (img->buffer_view) {
        bytes = (const uint8_t*)cgltf_buffer_view_data(img->buffer_view);
        size = img->buffer_view->size;
    } else if (img->uri && std::strncmp(img->uri, "data:", 5) == 0) {
        const char* comma = std::strchr(img->uri, ',');
        if (!comma) { error = "malformed data URI"; return false; }
        const char* payload = comma + 1;
        size_t len = std::strlen(payload);
        size_t pad = (len >= 1 && payload[len - 1] == '=') + (len >= 2 && payload[len - 2] == '=');
        size = len / 4 * 3 - pad;
        if (cgltf_load_buffer_base64(&opts, size, payload, &b64) != cgltf_result_success) {
            error = "bad base64 image data";
            return false;
        }
        bytes = (const uint8_t*)b64;
    } else if (img->uri) {
        std::string rel = img->uri;
        cgltf_decode_uri(rel.data());                       // turns %20 etc. back into characters
        rel.resize(std::strlen(rel.c_str()));
        std::ifstream f(baseDir / rel, std::ios::binary);
        if (!f) { error = "cannot open image file '" + rel + "'"; return false; }
        if (dependencies) dependencies->push_back((baseDir / rel).generic_string());
        file.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        bytes = file.data();
        size = file.size();
    } else {
        error = "image has no data";
        return false;
    }

    int w = 0, h = 0, comps = 0;
    stbi_uc* px = stbi_load_from_memory(bytes, (int)size, &w, &h, &comps, 4);   // always 4 channels out
    if (b64) std::free(b64);
    if (!px) { error = std::string("cannot decode image (PNG and JPEG are supported): ") + stbi_failure_reason(); return false; }
    out.width = w;
    out.height = h;
    out.rgba.assign(px, px + (size_t)w * h * 4);
    stbi_image_free(px);
    return true;
}

}  // namespace

bool loadGltf(const std::string& path, MeshData& out, std::string& error, const GltfImportOptions& options) {
    out = MeshData{};
    error.clear();

    cgltf_options opts{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&opts, path.c_str(), &data);
    if (res != cgltf_result_success) {
        error = "could not parse '" + path + "' (cgltf error " + std::to_string((int)res) + ")";
        return false;
    }
    // Reads the .bin file next to a .gltf (or decodes base64 / the .glb binary chunk)
    res = cgltf_load_buffers(&opts, data, path.c_str());
    if (res != cgltf_result_success) {
        error = "could not load the binary data for '" + path + "' (cgltf error " + std::to_string((int)res) + ")";
        cgltf_free(data);
        return false;
    }
    if (cgltf_validate(data) != cgltf_result_success) {
        error = "'" + path + "' is not a valid glTF file";
        cgltf_free(data);
        return false;
    }

    // Extensions that change how geometry is stored: without support the file can't be read at all
    for (cgltf_size i = 0; i < data->extensions_required_count; ++i) {
        const std::string ext = data->extensions_required[i];
        if (ext == "KHR_draco_mesh_compression" || ext == "EXT_meshopt_compression") {
            error = "'" + path + "' uses " + ext + " mesh compression, which is not supported "
                    "(re-export it uncompressed, or run: gltf-transform copy in.glb out.glb)";
            cgltf_free(data);
            return false;
        }
    }
    {   // Extensions that only change the look: the model still loads, but not exactly as its author intended
        std::string ignored;
        for (cgltf_size i = 0; i < data->extensions_used_count; ++i) {
            const std::string ext = data->extensions_used[i];
            if (ext.rfind("KHR_materials_", 0) == 0 && ext != "KHR_materials_unlit" && ext != "KHR_materials_emissive_strength")
                ignored += (ignored.empty() ? "" : ", ") + ext;
        }
        if (!ignored.empty())
            out.warnings.push_back("material extensions ignored (drawn as plain base color): " + ignored);
    }

    const std::filesystem::path baseDir = std::filesystem::path(path).parent_path();
    out.dependencies.push_back(path);
    for (cgltf_size b = 0; b < data->buffers_count; ++b) {          // an external .bin next to a .gltf
        const char* uri = data->buffers[b].uri;
        if (!uri || std::strncmp(uri, "data:", 5) == 0) continue;
        std::string rel = uri;
        cgltf_decode_uri(rel.data());
        rel.resize(std::strlen(rel.c_str()));
        out.dependencies.push_back((baseDir / rel).generic_string());
    }
    std::vector<int> imageMap(data->images_count, -2);       // glTF image -> out.images (-2 = not tried, -1 = failed)
    std::vector<int> materialMap(data->materials_count, -2); // glTF material -> out.materials

    // Creates (once) the MaterialData for a glTF material, decoding its base color image on first use
    auto materialFor = [&](const cgltf_material* mat) -> int {
        if (!mat) return -1;
        size_t mi = (size_t)cgltf_material_index(data, mat);
        if (materialMap[mi] != -2) return materialMap[mi];
        MaterialData md;
        md.name = mat->name ? mat->name : "material " + std::to_string(mi);
        md.doubleSided = mat->double_sided != 0;
        if (mat->alpha_mode == cgltf_alpha_mode_mask) md.alphaCutoff = std::max(mat->alpha_cutoff, 0.01f);   // glTF default is 0.5
        else if (mat->alpha_mode == cgltf_alpha_mode_blend) {
            md.alphaCutoff = 0.5f;      // real blending needs back-to-front sorting; alpha-test is a decent stand-in
            out.warnings.push_back("material '" + md.name + "' is alpha-blended; drawn alpha-tested instead");
        }
        if (mat->has_pbr_metallic_roughness) {
            const cgltf_texture_view& tv = mat->pbr_metallic_roughness.base_color_texture;
            if (tv.texture && !tv.texture->image && (tv.texture->basisu_image || tv.texture->webp_image))
                out.warnings.push_back("material '" + md.name + "': texture is KTX2/WebP only, which is not supported");
            if (tv.texture && tv.texture->image) {
                size_t ii = (size_t)cgltf_image_index(data, tv.texture->image);
                if (imageMap[ii] == -2) {
                    ImageData img;
                    std::string err;
                    img.name = tv.texture->image->name ? tv.texture->image->name : "image " + std::to_string(ii);
                    if (decodeImage(opts, tv.texture->image, baseDir, img, err, &out.dependencies)) {
                        imageMap[ii] = (int)out.images.size();
                        out.images.push_back(std::move(img));
                    } else {
                        imageMap[ii] = -1;
                        out.warnings.push_back("texture '" + img.name + "' skipped: " + err);
                    }
                }
                md.baseColorImage = imageMap[ii];
                if (tv.texcoord != 0)
                    out.warnings.push_back("material '" + md.name + "' uses UV set " + std::to_string(tv.texcoord) +
                                           "; only set 0 is read, so its texture will look wrong");
            }
        }
        materialMap[mi] = (int)out.materials.size();
        out.materials.push_back(std::move(md));
        return materialMap[mi];
    };

    for (cgltf_size n = 0; n < data->nodes_count; ++n) {
        const cgltf_node* node = &data->nodes[n];
        if (!node->mesh) continue;

        float world16[16];
        cgltf_node_transform_world(node, world16);            // this node's transform times all its parents'
        const Mat4 world = toMat4(world16);
        // Normals need the inverse-transpose so non-uniform scales in the file still light correctly
        const Mat4 normalMat = world.inverse().transposed();

        for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p) {
            const cgltf_primitive& prim = node->mesh->primitives[p];
            if (prim.type != cgltf_primitive_type_triangles) continue;

            const cgltf_accessor* posAcc = nullptr;
            const cgltf_accessor* nrmAcc = nullptr;
            const cgltf_accessor* uvAcc = nullptr;
            for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
                const cgltf_attribute& attr = prim.attributes[a];
                if (attr.type == cgltf_attribute_type_position) posAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) nrmAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uvAcc = attr.data;
            }
            if (!posAcc || posAcc->count == 0) continue;

            Vec3 color = { 1.0f, 1.0f, 1.0f };
            if (prim.material && prim.material->has_pbr_metallic_roughness) {
                const float* f = prim.material->pbr_metallic_roughness.base_color_factor;
                color = { linearToSrgbApprox(f[0]), linearToSrgbApprox(f[1]), linearToSrgbApprox(f[2]) };
            }

            // KHR_texture_transform (offset / rotation / scale of the base color texture) is baked into the UVs:
            // uv' = offset + R(rotation) * (scale * uv), exactly the spec's translation * rotation * scale
            bool uvTransform = false;
            float uvOff[2] = { 0, 0 }, uvScale[2] = { 1, 1 }, uvCos = 1.0f, uvSin = 0.0f;
            if (prim.material && prim.material->has_pbr_metallic_roughness) {
                const cgltf_texture_view& tv = prim.material->pbr_metallic_roughness.base_color_texture;
                if (tv.texture && tv.has_transform) {
                    uvTransform = true;
                    uvOff[0] = tv.transform.offset[0]; uvOff[1] = tv.transform.offset[1];
                    uvScale[0] = tv.transform.scale[0]; uvScale[1] = tv.transform.scale[1];
                    uvCos = std::cos(tv.transform.rotation);
                    uvSin = std::sin(tv.transform.rotation);
                }
            }

            const uint32_t base = (uint32_t)out.vertices.size();
            const cgltf_size count = posAcc->count;
            for (cgltf_size i = 0; i < count; ++i) {
                float p3[3] = { 0, 0, 0 }, n3[3] = { 0, 0, 0 }, uv[2] = { 0, 0 };
                cgltf_accessor_read_float(posAcc, i, p3, 3);
                if (nrmAcc) cgltf_accessor_read_float(nrmAcc, i, n3, 3);
                if (uvAcc) cgltf_accessor_read_float(uvAcc, i, uv, 2);

                MeshVertex v;
                v.pos = transformPoint(world, { p3[0], p3[1], p3[2] });
                v.normal = nrmAcc ? (normalMat * Vec4(n3[0], n3[1], n3[2], 0.0f)).xyz().normalized() : Vec3{ 0, 0, 0 };
                v.color = color;
                if (uvTransform) {
                    float sx = uv[0] * uvScale[0], sy = uv[1] * uvScale[1];
                    v.u = uvCos * sx - uvSin * sy + uvOff[0];
                    v.v = uvSin * sx + uvCos * sy + uvOff[1];
                } else {
                    v.u = uv[0];
                    v.v = uv[1];
                }
                out.vertices.push_back(v);
            }

            const cgltf_size indexCount = prim.indices ? prim.indices->count : count;
            const size_t firstIndex = out.indices.size();
            for (cgltf_size i = 0; i < indexCount; ++i) {
                cgltf_size idx = prim.indices ? cgltf_accessor_read_index(prim.indices, i) : i;
                out.indices.push_back(base + (uint32_t)idx);
            }
            // A mirrored node (negative scale) flips the winding; flip it back so faces still point outward
            if (world.m[0][0] * (world.m[1][1] * world.m[2][2] - world.m[2][1] * world.m[1][2])
              - world.m[1][0] * (world.m[0][1] * world.m[2][2] - world.m[2][1] * world.m[0][2])
              + world.m[2][0] * (world.m[0][1] * world.m[1][2] - world.m[1][1] * world.m[0][2]) < 0.0f) {
                for (size_t i = firstIndex; i + 2 < out.indices.size(); i += 3)
                    std::swap(out.indices[i + 1], out.indices[i + 2]);
            }

            // One submesh per material run (neighbouring primitives with the same material merge)
            {
                int mat = materialFor(prim.material);
                uint32_t first = (uint32_t)firstIndex, cnt = (uint32_t)(out.indices.size() - firstIndex);
                if (!out.submeshes.empty() && out.submeshes.back().material == mat &&
                    out.submeshes.back().firstIndex + out.submeshes.back().indexCount == first)
                    out.submeshes.back().indexCount += cnt;
                else
                    out.submeshes.push_back(Submesh{ first, cnt, mat });
            }

            // No normals in the file: use the average of the face normals around each vertex (smooth shading)
            if (!nrmAcc) {
                for (size_t i = firstIndex; i + 2 < out.indices.size(); i += 3) {
                    MeshVertex& a = out.vertices[out.indices[i]];
                    MeshVertex& b = out.vertices[out.indices[i + 1]];
                    MeshVertex& c = out.vertices[out.indices[i + 2]];
                    Vec3 faceN = (b.pos - a.pos).cross(c.pos - a.pos);   // length = twice the triangle's area, so big faces weigh more
                    a.normal = a.normal + faceN;
                    b.normal = b.normal + faceN;
                    c.normal = c.normal + faceN;
                }
                for (cgltf_size i = 0; i < count; ++i) {
                    MeshVertex& v = out.vertices[base + i];
                    v.normal = v.normal.length() > 0.0f ? v.normal.normalized() : Vec3{ 0, 1, 0 };
                }
            }
        }
    }
    cgltf_free(data);

    if (out.vertices.empty() || out.indices.empty()) {
        error = "'" + path + "' contains no triangle meshes";
        return false;
    }

    // Bounds of everything, in the file's own units
    Vec3 lo = out.vertices[0].pos, hi = lo;
    for (const MeshVertex& v : out.vertices) {
        lo = { std::min(lo.x, v.pos.x), std::min(lo.y, v.pos.y), std::min(lo.z, v.pos.z) };
        hi = { std::max(hi.x, v.pos.x), std::max(hi.y, v.pos.y), std::max(hi.z, v.pos.z) };
    }

    if (options.normalize) {
        Vec3 centre = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
        float largest = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z });
        float s = largest > 1e-12f ? 1.0f / largest : 1.0f;
        for (MeshVertex& v : out.vertices) v.pos = (v.pos - centre) * s;   // uniform scale: normals are unaffected
        lo = (lo - centre) * s;
        hi = (hi - centre) * s;
    }
    out.bounds.min = lo;
    out.bounds.max = hi;
    out.name = std::filesystem::path(path).stem().string();
    return true;
}