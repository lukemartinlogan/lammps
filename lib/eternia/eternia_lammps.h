/* -*- c++ -*- ----------------------------------------------------------
   Eternia vector backend for LAMMPS -- HOST-SIDE BOUNDARY HEADER.

   This header is the entire contract between LAMMPS and the Eternia (Clio
   CTE gpu_vector) backend, and it deliberately mentions neither CUDA nor
   Clio. That separation is not stylistic: the Eternia paging kernels are
   C++20 DEVICE COROUTINES and can only be compiled by clang++-22 with
   -x cuda -DCLIO_GPU_YIELD_CORO=ON, while LAMMPS itself is built by the
   user's ordinary compiler (and, under the KOKKOS package, by nvcc, which
   cannot compile a device coroutine at all).

   So the split is:

     lib/eternia/eternia_lammps.cc   clang++-22, sees Clio and CUDA
     src/ETERNIA/pair_lj_cut_eternia.cpp   the LAMMPS compiler, sees only this

   Everything crossing the line is a POD or an opaque pointer.
------------------------------------------------------------------------- */

#ifndef LMP_ETERNIA_LAMMPS_H
#define LMP_ETERNIA_LAMMPS_H

#include <cstdint>

namespace eternia_lammps {

/**
 * Geometry of the out-of-core run.
 *
 * The four per-atom arrays each get their OWN Eternia vector, and therefore
 * their own per-block page cache. One vector holding all four would be
 * simpler, but the caches are per block by design and the four arrays have
 * completely different reuse: positions are re-read once per neighbour of
 * every atom, the neighbour list is read strictly once, and forces are
 * write-only. Sharing one cache between them lets the streaming arrays evict
 * the one array that actually has reuse.
 *
 * VRAM cost is the thing to watch: each vector pins
 * nblocks * pages_per_block * page_bytes of device memory, so the four
 * slot counts are separate knobs rather than one.
 */
struct Config {
  /** CTE tag prefix; the four vectors are "<prefix>_x", "_type", "_neigh",
   *  "_f". Must be unique per run, or a previous run's pages are inherited. */
  const char *tag_prefix = "lmp_eternia";
  int gpu_id = 0;

  /** Page granularity, in bytes, shared by all four vectors. A page of x
   *  holds page_bytes/(3*sizeof(float)) atoms, so this is really "how many
   *  atoms travel together" -- see the note on spatial ordering in
   *  UploadAtoms. */
  std::uint64_t page_bytes = 262144;

  /** CUDA launch geometry. Every Eternia hold is BLOCK-COLLECTIVE, so the
   *  block is the unit that owns a page cache and the unit that suspends. */
  std::uint32_t nblocks = 64;
  std::uint32_t nthreads = 256;

  /** Resident pages per block, per vector. */
  std::uint32_t slots_x = 16;
  std::uint32_t slots_type = 4;
  std::uint32_t slots_neigh = 4;
  std::uint32_t slots_f = 8;

  /** Turn on the device-side paging counters (faults/puts/evicts). Costs an
   *  atomic per page event; off by default. */
  bool stats = false;
};

/** Paging activity of one compute(), summed over the four vectors. */
struct Stats {
  std::uint64_t x_faults = 0;
  std::uint64_t x_evicts = 0;
  std::uint64_t neigh_faults = 0;
  std::uint64_t f_puts = 0;
  std::uint64_t f_put_errors = 0;
  std::uint64_t get_errors = 0;

  /** Neighbour entries the kernel actually examined, and the number it should
   *  have. Every entry belongs to exactly one position page, so these must be
   *  equal -- a shortfall means pairs were silently skipped, which shows up
   *  in the energy as a small deficit rather than as an error. */
  std::uint64_t pairs_seen = 0;
  std::uint64_t pairs_expected = 0;
  /** Diagnostics: entries whose type read back as 0 (a page-cache miss shows
   *  up this way, because the padding beyond the atom count is zero), and
   *  entries rejected by the cutoff. */
  std::uint64_t pairs_badtype = 0;
  std::uint64_t pairs_cut = 0;
};

/** Opaque handle; the definition lives in the clang-compiled TU. */
struct Context;

/**
 * Bring up the Clio runtime and the four vectors.
 *
 * @param nall  atoms including ghosts -- the length x/type/f are sized for.
 * @return null on failure (runtime unavailable, CUDA absent, bad geometry).
 */
Context *Create(const Config &cfg, int nall);
void Destroy(Context *ctx);

/**
 * Push positions and types into the CTE, page by page.
 *
 * Written through the CTE's blob path rather than through a seeding kernel,
 * so the host never has to materialise the whole array in VRAM -- which is
 * the entire point of the out-of-core run. Cost is one PutBlob per page.
 *
 * SPATIAL ORDERING MATTERS. A page holds a contiguous RANGE OF ATOM INDICES,
 * so the paging only pays off if atoms near each other in space are near each
 * other in index -- which is exactly what LAMMPS's binned atom sorting gives
 * (atom_modify sort). Without sorting, a neighbour list touches essentially
 * every page and the cache degenerates to a fault per pair.
 *
 * @param x     nall*3 doubles, LAMMPS's native layout; narrowed to float.
 * @param type  nall ints.
 */
void UploadAtoms(Context *ctx, const double *const *x, const int *type, int nall);

/**
 * Push the neighbour list into the CTE in a flattened, page-friendly form.
 *
 * LAMMPS stores neighbours as an array of per-atom pointers into pages of a
 * MyPage allocator, which is not something a device can chase. This flattens
 * it to: for atom slot ii, its neighbours occupy [offset[ii], offset[ii+1]).
 * The offsets array is small and stays in ordinary device memory; only the
 * neighbour indices themselves -- the big array -- are paged.
 */
void UploadNeighbors(Context *ctx, const int *ilist, const int *numneigh,
                     const int *const *firstneigh, int inum);

/** Per-type LJ coefficients, indexed [itype*ntypes_p1 + jtype]. */
void SetLJParams(Context *ctx, const double *lj1, const double *lj2,
                 const double *lj3, const double *lj4, const double *offset,
                 const double *cutsq, int ntypes_p1);

/**
 * Run the paged LJ force kernel and leave the forces in the f vector.
 *
 * @param eflag  accumulate potential energy
 * @param newton_pair LAMMPS newton_pair setting
 * @return false if the kernel failed (a CUDA error, or the yield driver hit
 *         its round cap -- which means a page never landed).
 */
bool ComputeLJCut(Context *ctx, int eflag, int newton_pair);

/**
 * Read forces back out of the CTE and ADD them into LAMMPS's f array.
 *
 * Adds rather than assigns: LAMMPS accumulates contributions from several
 * force styles into one f, so overwriting would silently drop the others.
 */
void DownloadForces(Context *ctx, double *const *f, int nall);

/** Potential energy accumulated by the last ComputeLJCut. */
double GetEnergy(Context *ctx);

/**
 * Global virial from the last ComputeLJCut, in LAMMPS order
 * (xx, yy, zz, xy, xz, yz).
 *
 * Computed per pair on the device rather than left to LAMMPS's
 * virial_fdotr_compute(): fdotr sums x.f over local AND ghost atoms, and a
 * full list with newton off gives ghosts no force at all, so the fallback
 * reported a pressure that was wrong rather than absent.
 */
void GetVirial(Context *ctx, double *out6);

Stats GetStats(Context *ctx);
void ResetStats(Context *ctx);

/** Human-readable reason the last failing call failed; never null. */
const char *LastError();

/** @return true if this build actually has the Eternia backend compiled in.
 *  The stub build (no CUDA, or no coroutine-capable clang) returns false, so
 *  the pair style can fail with a clear message instead of a link error. */
bool Available();

}  // namespace eternia_lammps

#endif
