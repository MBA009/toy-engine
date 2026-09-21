# Toy Engine

A small game engine built from scratch in C++, developed as a portfolio project to demonstrate engine/systems programming: rendering (software and GPU), a data-oriented ECS, an in-engine editor, an asset pipeline, and AI pathfinding/behavior — all integrated into one cohesive tool rather than separate demos.

## Why this project exists

Each phase below intentionally mirrors a layer of a real production engine (Unity, Unreal, and Godot all share this same underlying stack), built up from first principles so the *why* behind each layer (rendering pipelines, real-time performance constraints, GPU programming, systems architecture) is actually understood, not just used.

## Project goals

- Build a working understanding of the full rendering pipeline, from software rasterization up to a modern GPU API (Vulkan)
- Demonstrate data-oriented design with a real, benchmarked comparison against naive object-oriented architecture
- Build real developer tooling (an in-engine editor, asset pipeline) rather than only end-user-facing features
- Integrate AI pathfinding and behavior systems using industry-standard techniques and libraries
- Produce a single, cohesive demo where all of the above work together — not eight disconnected proofs of concept

## Architecture overview

```
┌─────────────────────────────────────────────┐
│              In-Engine Editor                │  (ImGui, gizmos, inspector)
├─────────────────────────────────────────────┤
│         AI (Navmesh, Pathfinding,             │  (Recast/Detour, behavior trees)
│              Behavior Trees)                  │
├─────────────────────────────────────────────┤
│           Asset Pipeline (glTF,               │  (cgltf, hot-reload)
│         textures, hot-reload)                 │
├─────────────────────────────────────────────┤
│      Entity Component System (ECS)            │  (data-oriented, benchmarked)
├─────────────────────────────────────────────┤
│         Renderer (Vulkan / GPU)                │  (pipeline, shaders, lighting)
├─────────────────────────────────────────────┤
│      Software Rasterizer (CPU, foundational)   │  (built first, to learn the pipeline)
└─────────────────────────────────────────────┘
```

Each layer is built directly on top of the one below it, and later phases (editor, AI) are wired into earlier ones (ECS, renderer) rather than existing as isolated systems.

## Development roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Project setup, windowing (GLFW) | ✅ Done |
| 1 | Software rasterizer (CPU rendering pipeline) | 🚧 In progress |
| 2 | GPU renderer (Vulkan) | ⬜ Not started |
| 3 | Data-oriented ECS + OOP benchmark comparison | ⬜ Not started |
| 4 | Debug tooling foundation (ImGui overlay) | ⬜ Not started |
| 5 | In-engine editor (picking, gizmos, inspector) | ⬜ Not started |
| 6 | Asset pipeline (glTF import, hot-reload) | ⬜ Not started |
| 7 | AI: navmesh pathfinding + behavior trees | ⬜ Not started |
| 8 | Full integration pass | ⬜ Not started |
| 9 | Portfolio packaging (demo video, docs) | ⬜ Not started |

*(Update this table as phases are completed.)*

## Key technical decisions

- **Software rasterizer before GPU rendering:** building the rendering pipeline on the CPU first (transform chain, rasterization, depth testing, perspective-correct interpolation) forces a real understanding of the math and pipeline stages before a graphics API abstracts them away.
- **Vulkan over OpenGL/DirectX:** more relevant to current engine job postings, and its explicitness forces a real understanding of GPU synchronization rather than hiding it behind driver magic.
- **Hand-rolled ECS over inheritance-based OOP:** components are stored in contiguous arrays rather than scattered heap-allocated objects, benchmarked directly against a naive OOP scene graph to demonstrate — not just assert — the performance difference at scale.
- **Recast/Detour for navmesh and pathfinding:** industry-standard libraries used by real engines, chosen deliberately over a from-scratch implementation to demonstrate the more realistic skill of integrating production tooling correctly.

## Tech stack

- **Language:** C++17
- **Build system:** CMake + Ninja
- **Windowing:** GLFW
- **Graphics API:** Vulkan (Phase 2 onward)
- **Debug UI / Editor:** Dear ImGui, ImGuizmo
- **Asset loading:** cgltf / tinygltf, stb_image
- **AI:** Recast & Detour (navmesh + pathfinding)
- **Serialization:** nlohmann/json

## Building the project

```bash
git clone --recurse-submodules <repo-url>
cd toy-engine
cmake -G Ninja -S . -B build
cmake --build build
./build/ToyEngine
```

> `--recurse-submodules` is required — third-party libraries (GLFW, ImGui, etc.) are included as git submodules, not vendored copies.

## Benchmark results

*(To be added once Phase 3's ECS vs. naive-OOP benchmark is complete — this section will include frame-time comparisons at 1k/10k/100k entities with an explanation of the results.)*

## Demo

*(A short demo video will be linked here once Phase 9 is complete.)*
