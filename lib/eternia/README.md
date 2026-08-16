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

1. **Pass A** records, in a per-block bitmap, which position pages those
   entries refer to.
2. **Pass B** holds each recorded page once, block-collectively, and lets
   every thread evaluate the pairs of its own atom that land in that page.

Pass B rescans the entries once per touched page, which is only a good trade
because the touched set is small - and that is exactly the spatial sorting
requirement above.

That bitmap, and the staged tile of i-atom positions, live in **global**
per-block scratch rather than in shared memory. This is not a performance
choice: a `co_await` can exit the kernel and have the driver relaunch the
block, so nothing in dynamic shared memory survives a page fault. Only the
final reductions use shared, where no `co_await` runs between the write and
the read.

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

## Larger than VRAM

The point of the exercise, on an 8 GiB (7.99 GiB usable) RTX 4070 Laptop.
An FCC lattice has a size-independent energy per atom, so every row below is
checked against the same reference, and every row accounts for all of its
neighbour entries:

| cells | atoms      | entries       | data     | E_pair     | `run 0` |
|-------|------------|---------------|----------|------------|---------|
| 16    | 16,384     | 1,277,952     | 0.01 GiB | -6.7733676 | 6 s     |
| 32    | 131,072    | 10,223,616    | 0.04 GiB | -6.7733676 | 4 s     |
| 64    | 1,048,576  | 81,788,928    | 0.34 GiB | -6.7733676 | 8 s     |
| 96    | 3,538,944  | 276,037,632   | 1.15 GiB | -6.7733666 | 14 s    |
| 160   | 16,384,000 | 1,277,952,000 | 5.31 GiB | -6.7733677 | 53 s    |
| 184   | 24,918,016 | 1,943,605,248 | **8.08 GiB** | -6.7733681 | 81 s |

Stock `lj/cut` gives -6.7733681. The last row holds more data than the GPU
has memory, and the run pages it: 69,068 position faults and 68,044
evictions in a single force evaluation. The GPU-side cache is
`blocks * slots * page` = 64 * 16 * 256 KB = 256 MB of positions, i.e. about
3% of the dataset.

### A real simulation, not one force evaluation

The table above is `run 0` -- a single force evaluation. That does not
exercise reneighbouring, atom sorting, or the per-step cache drop, so it is
not evidence that a simulation works. A 10-step MD run at the same 24.9M
atoms / 8.08 GiB, reneighbouring every 5 steps, against stock `lj/cut` on
CPU with the identical input:

| step | quantity | `lj/cut/eternia` | `lj/cut`   |
|------|----------|------------------|------------|
| 0    | E_pair   | -6.7733681       | -6.7733681 |
| 0    | Press    | -3.7027173       | -3.7027173 |
| 5    | E_pair   | -6.5490781       | -6.5490781 |
| 5    | Press    | -2.4556949       | -2.4556948 |
| 10   | E_pair   | -5.4355148       | -5.4355150 |
| 10   | TotEng   | -2.2772363       | -2.2772364 |
| 10   | Press    |  2.4443952       |  2.4443949 |

Temperature agrees to all printed digits at every step. 556 s for the 10
steps, and every step accounts for all of its neighbour entries
(1,943,605,248 at step 0; 1,901,186,392 after the list is rebuilt) with zero
failed reads or writebacks.

**The current ceiling is 2^31 neighbour entries**, not memory. At 24.9M atoms
the list already holds 1.94e9 entries - 90% of INT_MAX - because the device
offset table is 32-bit. That is under 1.2x of headroom above the largest
system that exceeds this machine's VRAM, so on a larger GPU the offset table
has to widen to 64 bits before the memory limit is reached. Exceeding it is
now a clean error rather than a silent wrap.

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
