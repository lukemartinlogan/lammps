# lib/eternia — the Eternia vector backend for LAMMPS

This directory holds the device half of the `ETERNIA` package: an
out-of-core `lj/cut` whose per-atom arrays live in the Clio Context Transfer
Engine and are paged into GPU memory *from inside the force kernel*.

The point is a working-set bound that does not depend on the atom count. A
conventional GPU pair style needs positions, types, forces and the whole
neighbour list resident in VRAM; this one needs only the page caches, so the
simulated system can be larger than the GPU.

## Why this is a separate CMake project

The paging kernel suspends on a page fault using **C++20 device
coroutines** (`co_await vec.HoldPageCoro(...)`). Only `clang++-22` driving
`-x cuda` can compile that — nvcc rejects `co_await` in device code outright,
and the KOKKOS package builds LAMMPS with nvcc. Since one CMake project
cannot host two C++ compilers, `cmake/Modules/Packages/ETERNIA.cmake`
configures this directory as an `ExternalProject` with its own toolchain and
links the resulting static library into LAMMPS.

Everything crossing the boundary is declared in `eternia_lammps.h`, which
mentions neither CUDA nor Clio.

## Building

Build and install iowarp-core with coroutines enabled:

```
cmake -S <core> -B build-coro \
  -DCLIO_CORE_ENABLE_CUDA=ON \
  -DCLIO_GPU_YIELD_CORO=ON \
  -DCMAKE_CUDA_COMPILER=clang++-22 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-coro -j && cmake --install build-coro --prefix <clio-inst>
```

Then LAMMPS:

```
cmake -S cmake -B build -DPKG_ETERNIA=ON \
  -Diowarp-core_DIR=<clio-inst>/lib/cmake/iowarp-core \
  -DETERNIA_CUDA_ARCHITECTURES=89
cmake --build build -j
```

The Clio daemon must be running before LAMMPS starts; the backend connects as
a client.

## Using it

```
newton off                       # required, see below
atom_modify sort 1000 2.0        # strongly recommended, see below

pair_style lj/cut/eternia 2.5 page 256 blocks 64 threads 256 slots 16 stats on
pair_coeff * * 1.0 1.0
```

| keyword   | default       | meaning                                  |
|-----------|---------------|------------------------------------------|
| `page`    | 256 (KB)      | page granularity — how many atoms travel together |
| `blocks`  | 64            | CUDA blocks; each owns its own page cache |
| `threads` | 256           | threads per block                        |
| `slots`   | 16            | resident pages per block, for positions   |
| `tag`     | `lmp_eternia` | CTE tag prefix; must be unique per run    |
| `stats`   | off           | print paging counters each step           |

VRAM used by the caches is roughly
`blocks × slots × page` per array, so the defaults above are about
64 × 16 × 256 KB = 256 MB for positions.

## Two requirements that are not stylistic

**`newton off`.** A half neighbour list requires each pair's force to be
applied to atom *j* as well, and *j* generally lives in another page — usually
another *block's* cache. The page caches are independent per block by design;
a cross-block read-modify-write is not something they support. With a full
list every atom computes its own total force and every write lands in a page
the block already holds. The cost is the usual factor of two in pair
evaluations. The pair style refuses to run with newton on rather than
silently halving the forces.

**Spatial sorting.** A page holds a contiguous *range of atom indices*, so
paging only pays off when atoms near each other in space are near each other
in index — which is exactly what `atom_modify sort` gives. Without it, a
neighbour list touches essentially every page and the cache degenerates to a
fault per pair. Unsorted input is still *correct*, just slow.

## How the kernel is organised

Every Eternia hold is **block-collective**: it can suspend the whole block, so
threads cannot each hold a different page. The obvious atom-major loop ("for
each of my neighbours, hold its page") would therefore serialise the block to
one neighbour at a time.

Instead each block takes a chunk of *i*-atoms — exactly one page of positions
— and runs two passes:

1. **Pass A** walks the chunk's neighbour lists and records, in a shared
   bitmap, which position pages the neighbours actually fall in.
2. **Pass B** holds each recorded page once, block-collectively, and lets
   every thread evaluate the pairs of its own atom that land in that page.

Pass B rescans the neighbour list once per touched page, which is only a good
trade because the touched set is small — and that is exactly the spatial
sorting requirement above.

## Not yet done

- **Virial on the device.** `compute()` falls back to LAMMPS's global
  `virial_fdotr_compute()`, which is correct but needs the force array back on
  the host. Per-atom virial (`compute stress/atom`) is unavailable.
- **Multi-rank.** Each rank would need its own tag prefix and GPU; the pair
  style currently refuses `nprocs > 1`.
- **`single()`**, and therefore anything that calls it (`compute group/group`,
  some fixes), is disabled — it would need a resident position array.
- **Positions are re-uploaded every step.** The integrator still runs on the
  host, so `x` makes a round trip per step. Moving `nve` onto the vector is
  the next step and is what would make the whole loop out-of-core.
