# Toy Engine — Detailed Step-by-Step Development Guide

**Goal:** Build a small but complete engine demonstrating rendering, systems architecture, tools, and AI — in that order, so each phase builds on the last.

**Total estimated time:** ~12–16 weeks part-time, ~6–8 weeks full-time focus.

---

## Phase 0 — Setup (2–3 days)

### Step 0.1 — Repo & project structure
- `git init`, create a `.gitignore` (use GitHub's standard C++ template)
- Suggested folder layout:
  ```
  /src          -- your engine source
  /external     -- third-party libs (as git submodules where possible)
  /assets       -- textures, models, shaders
  /shaders      -- .vert/.frag/.comp source
  CMakeLists.txt
  README.md
  ```
- Add libraries as **git submodules** (`git submodule add <url> external/<name>`) rather than copy-pasting source — keeps versions traceable and updates easy

### Step 0.2 — Toolchain
- Install a C++17-capable compiler (MSVC on Windows, GCC/Clang on Linux/Mac)
- Install CMake (3.20+) and Ninja (faster builds than default Make)
- Install the Vulkan SDK from LunarG (you'll need this by Phase 2, easier to set up now)

### Step 0.3 — Windowing
- Add GLFW as a submodule, wire it into `CMakeLists.txt` via `add_subdirectory(external/glfw)` + `target_link_libraries`
- Write `main.cpp`: create a GLFW window, run a loop that polls events until the window closes

### Step 0.4 — Confirm the pipeline
- Build and run. You should get a blank window you can close.
- Commit: "Initial project skeleton with window creation"

**Milestone check:** `cmake --build .` succeeds from a clean clone on a second machine (or a fresh folder) with no manual steps beyond installing the SDK.

---

## Phase 1 — Software Rasterizer (1.5–2 weeks)

### Step 1.1 — Framebuffer
- Allocate a `std::vector<uint32_t>` (or similar) sized `width * height` for pixel color (RGBA packed into 32 bits)
- Each frame: clear it to a background color, then blit it to the screen — easiest way is an SDL2 texture you update with `SDL_UpdateTexture`, or an OpenGL fullscreen quad if you'd rather avoid a second dependency
- Get a solid-color clear working and visible before writing a single triangle

### Step 1.2 — Math library
- Implement (or vendor in, e.g. GLM, but writing your own once is valuable for understanding):
  - `Vec3`, `Vec4`, `Mat4`
  - Operations: multiply, transpose, inverse, `perspective()`, `lookAt()`
- Write small unit tests: multiply a point by an identity matrix and confirm it's unchanged; multiply a known point by a known rotation and check against hand-calculated output
- This step is boring but skipping it means debugging rendering bugs that are actually math bugs — much harder to find later

### Step 1.3 — Define a triangle
- Hardcode 3 vertices in model space (position, color or UV) directly in code
- Write the transform chain: model space → world space → view space → clip space → screen space
- Print/log the transformed screen-space coordinates for a known input and sanity-check them by hand

### Step 1.4 — Rasterize the triangle
- Implement the edge-function test: for each pixel in the triangle's bounding box, compute the three edge functions; if all three agree in sign, the pixel is inside
- Fill inside pixels with a flat color first — get a visible filled triangle before adding anything else

### Step 1.5 — Depth buffer
- Allocate a `float` buffer same size as the framebuffer, initialize to infinity (or 1.0 in NDC)
- During rasterization, compute interpolated depth per pixel, compare against the depth buffer, write and update only if closer
- Test with two overlapping triangles at different depths — confirm correct occlusion

### Step 1.6 — Barycentric interpolation + perspective correction
- Compute barycentric weights per pixel (naturally falls out of the edge-function values)
- Interpolate UV/normal/color using these weights
- Apply the perspective-correction divide (interpolate `attribute/w`, then divide by interpolated `1/w`) — verify by texturing a triangle that's angled away from the camera and checking the texture doesn't visibly warp incorrectly

### Step 1.7 — Texturing
- Use `stb_image.h` to load a PNG/JPG into a CPU buffer
- Sample it using the interpolated UVs from Step 1.6 (nearest-neighbor first, bilinear filtering as a stretch goal)

### Step 1.8 — Lighting
- Add a normal per vertex, interpolate it (Step 1.6 machinery already supports this)
- Implement Blinn-Phong: ambient + diffuse (N·L) + specular (N·H) terms
- Add a single hardcoded directional or point light first, confirm shading looks correct, then generalize to a small array of lights

### Step 1.9 — Assemble the scene
- Replace your hardcoded single triangle with a full cube (12 triangles)
- Animate a rotation over time to prove your transform pipeline handles motion correctly

**Milestone check:** A textured, lit, spinning cube rendered entirely on CPU.
**Deliverable:** Short GIF/video + README section explaining each pipeline stage in your own words.

---

## Phase 2 — Move to GPU: Vulkan (2–3 weeks)

### Step 2.1 — Boilerplate (expect this to take the longest single chunk of the whole project)
- Follow vulkan-tutorial.com sequentially: instance → validation layers → physical/logical device → surface → swapchain
- Do not skip the validation layers setup — they will save you days of silent-failure debugging later
- Commit after each major milestone (instance created, device selected, swapchain working) so you can bisect if something breaks

### Step 2.2 — Render pass & pipeline
- Create a render pass with one color attachment (+ depth attachment once you get there)
- Write a minimal vertex + fragment shader pair in GLSL, compile to SPIR-V (`glslc`, ships with the Vulkan SDK)
- Build the graphics pipeline object (this is the step with the most boilerplate — expect a large, mostly-copy-once struct-filling function)

### Step 2.3 — Get a triangle on screen
- Hardcode vertex data in the vertex shader itself first (no vertex buffer yet) — proves the pipeline works before adding buffer complexity
- Once that renders, move vertex data into an actual vertex buffer + staging buffer upload

### Step 2.4 — Recreate the Phase 1 cube scene
- Add an index buffer for the cube
- Add a uniform buffer for MVP matrices, update it per frame
- Add a depth attachment + depth testing to the pipeline (mirrors Phase 1's depth buffer, now GPU-native)

### Step 2.5 — Lighting on GPU
- Port your Blinn-Phong math from Phase 1 into the fragment shader
- Pass light data via a uniform buffer or push constants; extend to 2–3 point lights via a small uniform array

### Step 2.6 — Benchmark
- Instrument frame time (CPU-side timer around your render loop is enough at this scale)
- Render the same scene at increasing triangle counts (subdivide the cube, or instance many cubes) in both Phase 1 (software) and Phase 2 (Vulkan) and record frame time at each count
- Put this table/graph directly in your README — it's a strong, concrete artifact

**Milestone check:** Same visual scene as Phase 1, GPU-accelerated, with a documented performance comparison.

---

## Phase 3 — Data-Oriented ECS (1–2 weeks)

### Step 3.1 — Define components
- Start minimal: `Transform { position, rotation, scale }`, `MeshRenderer { meshId, materialId }`, `Light { color, intensity, type }`
- Keep components as plain structs — no methods, no inheritance

### Step 3.2 — Entity storage
- Simplest viable version: `Entity` is just a `uint32_t` ID
- Store each component type in its own `std::vector`, indexed in parallel by entity ID (a "sparse set" or simple parallel-array approach is enough — don't reach for a full archetype table unless you want the extra complexity)

### Step 3.3 — Systems
- Write plain functions, e.g. `void RenderSystem(ECS& ecs)` that iterates entities having both `Transform` and `MeshRenderer`, and issues draw calls
- Write `void LightSystem(ECS& ecs)` that gathers all `Light` components into the uniform buffer data from Phase 2.5

### Step 3.4 — Migrate your scene
- Replace the hardcoded cube(s) from Phase 2 with entities created through the ECS
- Confirm the scene still renders identically — this is a refactor, not a feature change, so nothing should visually differ yet

### Step 3.5 — Build the comparison benchmark
- Write a second, deliberately naive `GameObject` class with virtual `Update()`/`Render()`, using inheritance for different object types
- Spawn matching scenes (1k / 10k / 100k objects) in both systems
- Measure and graph update+render time for each; include cache-miss intuition in your writeup (why contiguous arrays beat scattered heap allocations)

**Milestone check:** ECS-driven scene, plus a benchmark chart with a short written explanation of *why* the ECS wins.

---

## Phase 4 — Debug Tooling Foundation (3–5 days)

### Step 4.1 — Integrate Dear ImGui
- Add as a submodule, use the Vulkan+GLFW backend files ImGui ships with
- Get a single "Hello ImGui" window rendering on top of your existing scene

### Step 4.2 — Stats overlay
- Display: FPS, frame time (ms), entity count, draw call count
- Track draw calls with a simple counter incremented in your render system

### Step 4.3 — Toggles
- Wireframe mode (swap pipeline or use `VK_POLYGON_MODE_LINE`)
- Disable lighting (a shader uniform flag)
- Freeze/unfreeze camera movement

**Milestone check:** Live debug overlay running alongside your GPU renderer, all toggles functional.

---

## Phase 5 — In-Engine Editor (2–3 weeks)

### Step 5.1 — Camera controls
- Implement a free-fly or orbit camera controllable with mouse+keyboard — needed before picking makes sense to test

### Step 5.2 — Object picking
- Convert mouse click → normalized device coordinates → world-space ray using your inverse view-projection matrix
- Test the ray against each entity's bounding box (start with axis-aligned bounding boxes, computed from mesh bounds)
- Select the closest hit entity

### Step 5.3 — Hierarchy panel
- ImGui list of all entities (by name — add a `Name` component if you don't have one)
- Clicking an entry selects it (shared selection state with picking from 5.2)

### Step 5.4 — Inspector panel
- For the selected entity, render editable ImGui widgets for each of its components (position/rotation/scale sliders, color pickers for light color, etc.)
- Changes should apply live to the ECS data immediately

### Step 5.5 — Transform gizmos
- Integrate ImGuizmo
- Wire it to the selected entity's `Transform` component — dragging the gizmo should update position/rotation/scale live

### Step 5.6 — Scene serialization
- Pick a format (JSON via nlohmann/json is the easiest starting point)
- Write a `SaveScene()` that dumps all entities + components to JSON
- Write a `LoadScene()` that reconstructs the ECS state from that JSON
- Test: save, close the app, reopen, load — confirm the scene is identical

**Milestone check:** Click an object, drag it with a gizmo, edit its properties in a panel, save, restart the app, reload — everything matches.

---

## Phase 6 — Asset Pipeline (1–2 weeks)

### Step 6.1 — glTF loading
- Add `cgltf` (or `tinygltf`) as a submodule
- Write a loader that parses a `.gltf`/`.glb` file into your engine's mesh format (vertex positions, normals, UVs, indices)
- Load one small test model and confirm it renders correctly through your Phase 2 pipeline

### Step 6.2 — Materials & textures from glTF
- Parse the material/texture references in the glTF file, load referenced images via `stb_image.h`
- Wire the loaded material into your `MeshRenderer` component

### Step 6.3 — Import-time processing
- Generate mipmaps for loaded textures (either CPU-side downsampling or GPU blit-based generation)
- Optional: basic texture compression if you want to go further

### Step 6.4 — Hot-reloading
- Poll file modification timestamps for watched assets (shaders, textures) each frame or on an interval
- On change: recompile the shader (re-run `glslc`, reload the SPIR-V, rebuild the pipeline) or reload the texture, without restarting the app

### Step 6.5 — Real asset test
- Download a CC-licensed glTF model (Sketchfab has many), drop it in `/assets`, confirm it loads, appears in the editor hierarchy, and is fully editable

**Milestone check:** Drag a glTF file into the assets folder, it's usable in the editor with no restart required.

---

## Phase 7 — AI: Pathfinding and Behavior (2 weeks)

### Step 7.1 — Navmesh generation
- Add Recast as a submodule
- Feed your scene's static geometry (floor/walls) into Recast's build pipeline to generate a navmesh
- Visualize the generated navmesh as a debug overlay (reuses Phase 4 tooling) to confirm it looks correct before using it

### Step 7.2 — Pathfinding queries
- Add Detour, use it to query a path between two points on the navmesh
- Debug-draw the returned path as a line strip

### Step 7.3 — Agent component & movement
- Add an `AIAgent` component (current path, current waypoint index, move speed)
- Write a `PathFollowSystem` that moves agents along their current path each frame

### Step 7.4 — Behavior tree
- Hand-roll a minimal behavior tree: `Selector`, `Sequence`, and `Leaf` node types are enough
- Implement leaves: `IsPlayerInRange`, `MoveToPlayer`, `Patrol`, `Flee`
- Compose: Selector( Sequence(IsPlayerInRange, MoveToPlayer), Patrol ) as a first working tree
- Add a `Health` field and a `Flee` branch triggered below a threshold

### Step 7.5 — Debug visualization
- Draw each agent's current behavior tree node (as text above their head, or in the ImGui overlay) so you can watch decisions happen in real time
- This is what makes AI demos convincing in a portfolio video — visible reasoning, not just movement

**Milestone check:** Multiple agents patrol, detect a "player" entity, chase, and flee at low health, with visible state.

---

## Phase 8 — Integration Pass (1 week)

### Step 8.1 — Place agents from the editor
- Extend Phase 5's picking/placement flow to support spawning an `AIAgent` entity at a clicked location

### Step 8.2 — Expose AI parameters in the Inspector
- Detection range, flee health threshold, patrol waypoints — all editable live via ImGui, same pattern as Step 5.4

### Step 8.3 — Full loop test
- Import a model (Phase 6) → place it and an AI agent in the editor (Phase 5) → watch it path and react (Phase 7) → rendered via Vulkan (Phase 2) → managed via ECS (Phase 3) → all visible through debug overlays (Phase 4)
- Fix any friction points where phases don't compose cleanly — this pass often surfaces small integration bugs

**Milestone check:** One cohesive demo scene, not eight disconnected demos.

---

## Phase 9 — Portfolio Packaging (1 week)

### Step 9.1 — Record the demo
- 3–5 minutes: open editor → place object → place AI agent → show it patrol/chase/flee → toggle debug views → show the ECS benchmark chart on screen

### Step 9.2 — Write the README
- Architecture diagram (boxes and arrows is fine)
- Key decisions and why (Vulkan over OpenGL, hand-rolled ECS over inheritance, Recast over custom navmesh)
- ECS benchmark results front and center, with the graph embedded
- Build instructions verified on a clean checkout

### Step 9.3 — Clean up
- Squash noisy WIP commits if the history is messy, but keep the overall progression visible (per-phase commits are good — don't squash everything into one)
- Tag releases per phase if you want reviewers to see incremental snapshots (`git tag phase-1-rasterizer`, etc.)

### Step 9.4 — Publish
- Push to GitHub, pin it on your profile, link the demo video in the README

---

## Suggested pacing (part-time)

| Weeks | Phase |
|---|---|
| 1 | Phase 0 + start Phase 1 |
| 2–3 | Finish Phase 1 |
| 4–6 | Phase 2 |
| 7–8 | Phase 3 |
| 8 | Phase 4 |
| 9–11 | Phase 5 |
| 12–13 | Phase 6 |
| 14–15 | Phase 7 |
| 16 | Phase 8 + 9 |

## Key resources by phase

- **Phase 1:** Sebastian Lague's rasterizer video, scratchapixel.com
- **Phase 2:** vulkan-tutorial.com, GPU Gems (free online)
- **Phase 3:** "Data-Oriented Design" by Richard Fabian (free online)
- **Phase 5:** Dear ImGui examples repo, ImGuizmo repo
- **Phase 6:** cgltf/tinygltf docs, stb_image.h
- **Phase 7:** Recast & Detour docs, *Game AI Pro* (free volumes online)

## Practical notes

- Don't perfect one phase before moving on — get each to "good enough, works, committed," then move forward. Polish happens in Phase 9.
- If you get stuck for more than 2–3 days on a Vulkan bug, that's normal — its verbosity means small mistakes are common. Check validation layer output first, always.
- If time runs short, Phases 0–4 alone are already a solid portfolio piece. Phases 5–8 push it from "good" to "clearly a systems thinker."
- Commit at the end of every numbered step above, not just every phase — smaller commits make it much easier to bisect bugs and show real progression in your history.