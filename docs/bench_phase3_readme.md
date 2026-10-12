## Data-oriented ECS vs. object-oriented scene

The same simulated scene (objects that move, spin, bob and produce a model matrix for drawing) run through two implementations: a sparse-set ECS (`ecs.h`) and a conventional heap-allocated `GameObject` hierarchy with virtual `update()`/`move()`/`render()` (`oop_scene.h`). CPU only, no rendering. Release build, MSVC `/O2`.

![Full-frame cost vs. object count, ECS vs. OOP](docs/phase3_ecs_vs_oop.png)

Time per frame in ms (OOP time / ECS time in brackets; above 1.0 means the ECS is faster). "OOP shuffled" is the same OOP scene with its allocation order scrambled relative to update order, like a scene that has had objects created and destroyed for a while.

| Workload | Objects | ECS | OOP | OOP shuffled |
|---|---:|---:|---:|---:|
| **Move** (`position += velocity*dt`) | 1,000 | 0.0015 | 0.0013 (0.87x) | 0.0013 (0.83x) |
| | 10,000 | 0.0155 | 0.0169 (1.09x) | 0.0202 (1.31x) |
| | 100,000 | 0.167 | 0.291 (1.74x) | 0.591 (3.53x) |
| | 1,000,000 | 1.85 | 10.68 (5.77x) | 15.84 (8.56x) |
| **Update** (spin + bob, uses `sin`) | 1,000 | 0.0039 | 0.0051 (1.31x) | 0.0054 (1.38x) |
| | 10,000 | 0.0445 | 0.0785 (1.76x) | 0.0897 (2.02x) |
| | 100,000 | 0.598 | 1.117 (1.87x) | 1.686 (2.82x) |
| | 1,000,000 | 6.62 | 19.68 (2.97x) | 29.49 (4.45x) |
| **Draw list** (model matrix per object) | 1,000 | 0.0222 | 0.0227 (1.02x) | 0.0227 (1.02x) |
| | 10,000 | 0.225 | 0.232 (1.03x) | 0.237 (1.05x) |
| | 100,000 | 2.26 | 3.88 (1.71x) | 5.40 (2.38x) |
| | 1,000,000 | 22.7 | 55.8 (2.46x) | 117.1 (5.16x) |
| **Full frame** (all three) | 1,000 | 0.0276 | 0.0290 (1.05x) | 0.0290 (1.05x) |
| | 10,000 | 0.287 | 0.340 (1.19x) | 0.354 (1.23x) |
| | 100,000 | 3.07 | 5.04 (1.64x) | 7.51 (2.44x) |
| | 1,000,000 | 53.7 | 99.2 (1.85x) | 151.8 (2.83x) |

Raw numbers: [`phase3_ecs_bench.csv`](docs/phase3_ecs_bench.csv).

### What the numbers show

- **Small scenes: no meaningful difference.** At 1,000 objects the whole frame costs about 0.03 ms either way, and the ECS is even slightly slower on the cheapest workload (0.87x). The layout only matters once the data stops fitting in cache.
- **The gap grows with scene size** (1.05x at 1k, 1.85x at 1M for the full frame) and is widest on the cheapest per-object work: the move system is 5.8x faster at 1M objects. It reads 32 bytes per object from two packed arrays; the OOP version loads a 112-byte object through a pointer and a virtual call to do the same two adds.
- **Heap layout hurts OOP, not the ECS.** Scrambling allocation order raises the OOP full frame from 99 ms to 152 ms at 1M objects. The ECS iterates arrays in order, so its time doesn't depend on allocation history.
- **Where work per object is heavy, the layout matters less.** The draw list is dominated by `sin`/`cos` and writing a 72-byte matrix per object, and shows the smallest gap (about 1.0x up to 10k objects).

### Method

- One deterministic scene description (fixed RNG seed) builds all three implementations: 20% static, 30% spinning, 50% bobbing and spinning; every object has a position, a mesh/material ID, a name, and a velocity.
- Before timing, the three scenes run 10 frames and their draw lists are compared; the benchmark aborts on a mismatch. The ECS also has unit tests (add/remove/swap-and-pop, multi-component queries, destroy and ID reuse, matrix correctness) that run in Release.
- Each measurement: 0.5 s of warmup, then 2 s of timed frames, reported as the average per frame.

### Caveats

- Single runs on one machine; run-to-run variance was not measured.
- The result depends on the compiler: on GCC `-O2` the draw-list step was roughly even or slightly slower in the ECS up to 100k objects (the MSVC numbers above show the ECS ahead). Treat the sub-1.3x differences as noise-level.
- The OOP baseline is a deliberately plain one: virtual calls, `std::unique_ptr` per object, 112-byte objects. A leaner OOP design (data-oriented members, devirtualized calls, contiguous storage) would narrow the gap, which is the point of the data-oriented approach.
- The ECS has no generation counter on entity IDs and does not allow adding or removing components inside an `each()` callback.

### Reproduce

```powershell
cmake --build build-release --target EcsBench
.\build-release\EcsBench.exe > docs\phase3_ecs_bench.csv
```
