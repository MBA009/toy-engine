# Architecture notes

A tour of how the engine fits together, in the order a frame runs. For the one-page overview see the [README](../README.md).

## Frame order (`vk_main.cpp`)

1. **Input and camera.** Right-mouse fly camera; picking, gizmo and path tools read the mouse; hot-reload polling runs every 0.5 s.
2. **Simulation** (`simulateAndUpload`), with `dt` clamped to 0.1 s and zero while paused or dragging a gizmo:
   `spinSystem` → `bobSystem` → `healthSystem` → `behaviorSystem` → `pathFollowSystem` → `playerKeyboardMove` → `updateGame`.
   When a run is won or lost the AI systems get `dt = 0`, so agents freeze while decorations keep animating.
3. **Draw list.** `buildDrawList` walks `Transform + MeshRenderer` and writes one `DrawItem` (model matrix + flags + mesh/material IDs) per entity. The list is sorted by mesh only if it is not already, then split into draw ranges.
4. **Upload.** The list is copied into this frame-in-flight's host-visible instance buffer.
5. **Record and submit.** One `vkCmdDrawIndexed` per (mesh, material) range with an instance count. The floor is a separate instance at the end of the buffer.
6. **ImGui.** Panels edit components directly; overlays (navmesh, paths, AI labels, rings) use ImGui's background draw list and the last frame's `proj * view`.

## The ECS (`ecs.h`)

Each component type has a `Pool<T>`: a dense array of components, a parallel array of entity IDs, and a sparse array mapping entity ID to dense index. Removal swaps with the last element, so the dense array stays packed. `Registry::each<A, B, ...>` iterates the pool of the first type and looks the others up, so the rarest component should be listed first.

Why it is faster than the OOP baseline: a system touches only the bytes it needs (the move system reads 32 bytes per object from two packed arrays; the OOP version loads a 112-byte object through a pointer and a virtual call), and iteration order does not depend on allocation history.

Limits that are deliberate: no entity generations, and no structural changes (add/remove component) inside an `each()` callback. The editor applies structural changes after it has finished drawing a panel.

## Meshes, materials and assets

- Mesh 0 is the built-in cube, mesh 1 the floor, 2+ are imported glTF files. All meshes share one vertex buffer and one index buffer; each mesh records its `vertexOffset`, `firstIndex` and a list of submeshes (one per material).
- `loadGltf` returns a renderer-independent `MeshData` (vertices, indices, submeshes, decoded images, materials, warnings, dependencies). It bakes node transforms and `KHR_texture_transform` into the vertices, merges primitives that share a material, and refuses compressed meshes with a clear message.
- Textures are uploaded as `_SRGB` images with a GPU-generated mip chain; materials are descriptor sets (set 1) holding one combined image sampler.
- **Hot reload.** A polling `FileWatcher` (with debounce) tracks shader sources and imported assets. A changed shader is recompiled with `glslc` and the pipelines are rebuilt and swapped in after `vkDeviceWaitIdle`; if compilation fails the old pipeline stays and the error is shown. A changed asset is re-imported, entities are re-pointed to the new mesh, and the old one is retired (its space in the shared buffers is reclaimed on restart).

## Scene files (`scene_io.h`)

Plain JSON (`toy-engine-scene`, version 1). Meshes are saved by key (`builtin:cube` or the asset path), not by runtime ID. Loading builds the new world in a scratch `Registry` and only swaps it in if everything parsed, so a corrupt file can never half-replace the scene. Writes go to a temporary file and are renamed over the old one.

## Navigation (`navmesh.*`)

`gatherNavInput` collects the floor plus every entity marked as *level geometry* or *obstacle*, at its current pose. Obstacles also contribute their convex hull as a "solid", which is carved out of the heightfield so that a floor trapped inside a closed box is not left walkable. The Recast pipeline then runs: rasterize triangles, filter low and ledge spans, compact heightfield, erode by agent radius, partition into regions, contours, polygon mesh, detail mesh, and `dtCreateNavMeshData`.

Parameters are in world units (cell 0.05, agent radius 0.06, height 0.30 by default because the scene is small) and are validated against Recast's internal 8-bit limits before building, since out-of-range values produce a corrupt navmesh rather than an error.

Queries: `findPath` (nearest-poly snap, `findPath`, `findStraightPath`), `nearestPoint`, `raycast` (for clicking on the navmesh) and `moveAlong` (keyboard movement that slides along edges).

## AI

- **`AIAgent`** is the body: speed, turn speed, foot offset (distance from the entity origin to its feet) and the current list of waypoints. `pathFollowSystem` moves along it with a per-frame movement budget, so a long `dt` still turns corners correctly.
- **`AIBrain`** is the mind: tuning values plus the state the tree needs between frames. `trace` records, for each tree node, what it returned on the last tick, which is what the debug panel draws.
- **Behavior tree** (`bt.h`): nodes are `Selector`, `Sequence` and `Leaf`. `tickNode` evaluates from the root every tick. Leaves use `takeOver(owner)` to cancel a path that a different behavior set, so, for example, a patrol path never keeps running after the agent starts chasing.
- **Hysteresis** keeps behavior from flickering: an agent that has noticed the player keeps tracking until the distance exceeds 1.3x its detect range, and a fleeing agent keeps fleeing until it has recovered to the `recoverAt` fraction, not just above `fleeBelow`.
- **Game loop** (`updateGame`, `resetRun`): the player wins by getting inside the `Goal` radius and loses at zero health. Guards hurt the player while touching it (unless they are fleeing) and the player regains health otherwise.

## Testing approach

Everything above the Vulkan calls is written so it can run without a GPU: the ECS, systems, behavior tree, scene I/O, navmesh and the demo level are plain C++ with injected dependencies (the tree gets a `PathFn`, the scene serializer gets mesh key/resolve callbacks). That is what makes the headless tests possible, and they are the reason most of the integration bugs found during development were caught before reaching the renderer. The renderer and editor UI are checked by running them.
