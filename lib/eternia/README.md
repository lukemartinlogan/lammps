# lib/eternia - the Eternia vector backend for LAMMPS

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
`-x cuda` can compile that: nvcc rejects `co_await` in device code outright,
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
cmake -S cmake -B build -DPKG_ETERNIA=ON -DBUILD_MPI=OFF \
  -Diowarp-core_DIR=<clio-inst>/lib/cmake/iowarp-core \
  -DETERNIA_CUDA_ARCHITECTURES=89
cmake --build build -j
```

## Running

The Clio runtime is started **in process** - there is no separate daemon to
launch - but it needs a server configuration, named by `CLIO_SERVER_CONF`:

```
CLIO_SERVER_CONF=clio.yaml ./lmp -in in.melt.eternia
```

`examples/ETERNIA/` has a working `clio.yaml` with a GPU tier above a host
tier. Note that the tier `score` ordering is the opposite of what one might
guess: the vector writes pages at blob score 1.0 and the max-bandwidth
placement engine sorts the preferred group descending, so the **GPU tier must
have the higher score**.

```
newton off                       # required, see below
atom_modify sort 1000 2.0        # strongly recommended, see below

pair_style lj/cut/eternia 2.5 page 256 blocks 64 threads 256 slots 16 stats on
pair_coeff * * 1.0 1.0
```

| keyword   | default       | meaning                                  |
|-----------|---------------|------------------------------------------|
| `page`    | 256 (KB)      | page granularity - how many atoms travel together |
| `blocks`  | 64            | CUDA blocks; each owns its own page cache |
| `threads` | 256           | threads per block (must be a power of two) |
| `slots`   | 16            | resident position pages per block (must be >= 3) |
| `tag`     | `lmp_eternia` | CTE tag prefix; must be unique per run    |
| `stats`   | off           | print paging counters each step, and make a
                                failed page read or writeback fatal |

VRAM used by the caches is roughly `blocks * slots * page` per array, so the
defaults above are about 64 * 16 * 256 KB = 256 MB for positions.

## Two requirements that are not stylistic

**`newton off`**, and it must appear *before* the simulation box is defined.
A half neighbour list requires each pair's force to be applied to atom *j* as
well, and *j* generally lives in another page - usually another *block's*
cache. The page caches are independent per block by design; a cross-block
read-modify-write is not something they support. With a full list every atom
computes its own total force and every write lands in a page the block
already holds. The cost is the usual factor of two in pair evaluations. The
pair style refuses to run with newton on rather than silently halving the
forces.

**Spatial sorting.** A page holds a contiguous *range of atom indices*, so
paging only pays off when atoms near each other in space are near each other
in index - which is what `atom_modify sort` gives. Without it, a neighbour
list touches essentially every page and the cache degenerates to a fault per
pair. Unsorted input is still *correct*, just slow.

## How the kernel is organised

Every Eternia hold is **block-collective**: it can suspend the whole block, so
threads cannot each hold a different page. The obvious atom-major loop ("for
each of my neighbours, hold its page") would therefore serialise the block to
one neighbour at a time.

Instead each block takes a chunk of *i*-atoms - exactly one page of positions
- and walks that chunk's neighbour entries, which the host flattening has
made one contiguous range. For each page of that range:

1. **Pass A** records, in a shared bitmap, which position pages those entries
   refer to.
2. **Pass B** holds each recorded page once, block-collectively, and lets
   every thread evaluate the pairs of its own atom that land in that page.

Pass B rescans the entries once per touched page, which is only a good trade
because the touched set is small - and that is exactly the spatial sorting
requirement above.

Each step begins by dropping the block's caches. The host rewrites positions
into the CTE every step and nothing invalidates the device's resident pages,
so without the drop the kernel would keep serving step 0's coordinates
forever. Moving the integrator onto the vector, so positions are updated *in*
the cache rather than around it, is what would remove both the drop and the
per-step re-upload.

## Verification

Against stock `lj/cut` on FCC melts (`newton off`, same seed), the paged path
reproduces the reference to 7-8 significant figures on energy, total energy
and pressure - the agreement expected from narrowing positions to single
precision.

Correctness is checked ACROSS CONFIGURATIONS, not in one. A single matching
run means very little here: the kernel's behaviour depends on how often a
page fault actually suspends it, so page size, block count and cache size
each change the code path. Every cell below gives E_pair = -6.7733676
against a stock reference of -6.7733681, and every one accounts for all
neighbour entries:

| atoms | page  | blocks | slots | note                    |
|-------|-------|--------|-------|-------------------------|
| 16384 | 64KB  | 4      | 64    | 4 chunks, 4 blocks      |
| 16384 | 64KB  | 1      | 64    | 4 chunks, 1 block       |
| 16384 | 256KB | 1      | 64    | 1 chunk, 32 tiles       |
| 2048  | 4KB   | 1      | 64    | many small pages        |
| 2048  | 4KB   | 8      | 64    | many pages, many blocks |
| 2048  | 4KB   | 8      | 3     | cache far smaller than the working set: heavy eviction |

The kernel also counts the neighbour entries it actually examines and
compares them against the host's `sum(numneigh)`; a mismatch is fatal. That
check exists because the failure mode here is not a crash - it is an energy
that is a fraction of a percent low, which no eye catches.

## Not yet done

- **Multi-rank.** Each rank would need its own tag prefix and GPU; the pair
  style refuses `nprocs > 1`.
- **Per-atom energy and virial** (`compute pe/atom`, `stress/atom`). The
  global energy and virial are computed on the device and are correct; the
  per-atom decompositions are not accumulated.
- **`single()`**, and therefore anything that calls it (`compute
  group/group`, some fixes), is disabled - it would need a resident position
  array.
- **Positions are re-uploaded every step**, because the integrator still runs
  on the host. This is the dominant cost today and the reason the paged path
  is far slower than stock `lj/cut` on a system that fits in memory - the
  comparison it is built for is against out-of-core alternatives, not against
  an in-core run.
