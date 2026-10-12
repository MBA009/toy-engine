## ECS-driven rendering and GPU instancing

The Vulkan renderer now draws an ECS world: every cube is an entity (Transform + Spin + MeshRenderer). Each frame the spin system runs, a draw list of model matrices is built from the world, and that list is uploaded to a per-frame instance buffer. The same scene is drawn two ways: one `vkCmdDrawIndexed` per cube, or a single instanced draw for all cubes.

![CPU time per frame: one draw call per cube vs. instanced](docs/phase3_instancing.png)

RTX 5060 Laptop GPU, 800x600, Release, vsync off. Times in ms per frame. Raw numbers: [`phase3_instancing.csv`](docs/phase3_instancing.csv).

| Cubes | CPU, per-draw | CPU, instanced | CPU speedup | Record + submit, per-draw | Record + submit, instanced | GPU, per-draw | GPU, instanced |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.066 | 0.079 | 0.84x | 0.065 | 0.077 | 0.014 | 0.015 |
| 100 | 0.072 | 0.078 | 0.92x | 0.067 | 0.073 | 0.016 | 0.022 |
| 1,000 | 0.127 | 0.109 | 1.17x | 0.093 | 0.073 | 0.029 | 0.025 |
| 10,000 | 0.686 | 0.367 | 1.87x | 0.330 | 0.075 | 0.199 | 0.131 |
| 100,000 | 4.776 | 2.713 | 1.76x | 1.972 | 0.092 | 1.291 | 0.914 |

"Record + submit" is total CPU time minus `sim_ms` (the ECS update, draw-list build and upload).

### What the numbers show

- **Instancing removes the draw-call cost.** At 100,000 cubes, recording and submitting the commands drops from 1.97 ms to 0.09 ms (21x). Per-draw submission costs roughly 20-30 ns per draw call at scale.
- **The total improves less (1.8x) because the ECS work is now the bottleneck.** With instancing, `sim_ms` is 2.62 ms of the 2.71 ms CPU frame at 100,000 cubes (97%). That is about 26 ns per entity for the spin system, the draw list (a `sin`/`cos` pair and a 72-byte matrix per entity) and a 7.2 MB copy to the GPU-visible buffer.
- **GPU time also fell** (1.29 ms to 0.91 ms at 100,000 cubes), plausibly because the GPU's command-processing front end handles one draw instead of 100,000, though this run does not isolate the reason.
- **Below about 1,000 cubes there is no benefit**, and the instanced mode is a few microseconds slower on the CPU (0.079 vs 0.066 ms at 1 cube). That difference is near the noise floor of single runs.
- Wall-clock frame time sat at the display's 144 Hz (6.944 ms) in every row, so even 100,000 cubes is display-bound, not CPU- or GPU-bound, on this machine.

### Caveats

- Single run, one machine; variance not measured. CPU times under 0.1 ms include fixed per-frame work (fence wait, uniform update, submit) that is the same in both modes.
- The per-draw mode here is leaner than the Phase 2 renderer: the model matrix now comes from the instance buffer, so there is no per-draw push constant. Its per-draw CPU cost is therefore not comparable to the 108 ns per draw measured in Phase 2.
- Fragment work is identical in both modes (same triangles); at 800x600 with tiny cubes the GPU numbers mostly reflect vertex and primitive processing.

### Next bottleneck

Instanced rendering of 100,000 cubes is limited by the CPU-side ECS pass, not by Vulkan. The obvious next steps are caching sin/cos in the transform, rebuilding only changed entities, or running the systems across threads.
