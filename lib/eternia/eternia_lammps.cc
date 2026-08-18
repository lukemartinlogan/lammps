/* ----------------------------------------------------------------------
   Eternia vector backend for LAMMPS -- DEVICE SIDE.

   Compiled by clang++-22 with -x cuda -DCLIO_GPU_YIELD_CORO=ON. Nothing in
   LAMMPS proper includes this file; the only surface is eternia_lammps.h.

   WHAT IS OUT OF CORE
   -------------------
   Four per-atom arrays live in the CTE and are paged into GPU memory on
   demand from inside the force kernel:

     x      positions, 4 floats/atom (see kPosStride)
     type   atom types, 1 int/atom
     neigh  the flattened neighbour list, 1 int/entry
     f      forces, 4 floats/atom

   The working set is therefore bounded by the page caches, not by the atom
   count -- which is the property that makes an MD run larger than VRAM
   possible at all.

   WHY A FULL NEIGHBOUR LIST IS REQUIRED
   -------------------------------------
   A half list needs each pair's force applied to BOTH atoms, and atom j is
   in some other page -- usually some other BLOCK's cache. Writing it would
   mean a scattered, cross-block, read-modify-write into a paged array, which
   the per-block cache design specifically does not support (caches are
   independent per block; cross-block locking is not on the table). With a
   full list every atom computes its own total force, so all writes land in
   the page the block already holds. The cost is the usual factor of two in
   pair evaluations.

   WHY THE KERNEL IS PAGE-MAJOR, NOT ATOM-MAJOR
   --------------------------------------------
   Every Eternia hold is BLOCK-COLLECTIVE -- it may suspend the whole block --
   so threads cannot each hold a different page. The naive atom-major loop
   ("for each of my neighbours j, hold j's page") would therefore serialise
   the block to one neighbour at a time.

   Instead each block takes a CHUNK of i-atoms (exactly one page of x) and
   runs two passes:

     Pass A  walk the chunk's neighbour lists and record, in a shared
             bitmap, which x pages the neighbours actually fall in.
     Pass B  for each recorded page: hold it once, block-collectively, and
             let every thread evaluate the pairs of ITS atom that land in
             that page.

   Pass B rescans the neighbour list once per touched page. That is only a
   good trade because the touched set is small -- which holds exactly when
   atoms are spatially sorted (see UploadAtoms). Unsorted input is still
   CORRECT here, just slow, which is the right way round.
------------------------------------------------------------------------- */

#include "eternia_lammps.h"

#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(ETERNIA_LAMMPS_ENABLED)

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace eternia_lammps {
namespace {

/**
 * Floats per atom in the position vector.
 *
 * FOUR, not three, and the padding is load-bearing rather than wasteful: a
 * page holds page_bytes/sizeof(float) elements, and with a stride of 3 an
 * atom would straddle the page boundary for every page size that is a power
 * of two. Straddling means a single atom's x, y and z live in two different
 * pages -- two holds for one atom, and a correctness trap the moment one of
 * them is evicted between the two reads. With a stride of 4 the atom index
 * to page mapping is a shift, and no atom is ever split.
 */
constexpr u64 kPosStride = 4;

/** Pages recorded per pass-A window. 2048 bits = 256 bytes of shared. */
constexpr u32 kPageBitmapBits = 2048;
constexpr u32 kPageBitmapWords = kPageBitmapBits / 32;

/**
 * i-atoms staged in shared memory at a time.
 *
 * The i-atoms' positions and types are COPIED OUT of the page cache before
 * the neighbour loop runs, and every later read comes from shared memory.
 * That is a correctness requirement, not an optimisation: the i-page and the
 * j-pages live in the SAME per-block page table, so holding a j-page can
 * evict the i-page, after which the per-thread last_page_ still points at
 * that slot -- now refilled with some other page. Reading atom i through it
 * then yields another page's bytes.
 *
 * The symptom is not subtle once eviction actually happens: two atoms appear
 * to sit at the same coordinates, r6inv explodes, and the run reports
 * E_pair = inf with Press = nan. It stayed hidden for as long as the cache
 * was big enough to hold everything at once (32 slots, 2 pages), which is
 * exactly the regime that is NOT the point of an out-of-core vector.
 *
 * 512 atoms * (3 floats + 1 int) = 8 KB of the 48 KB shared budget. A chunk
 * larger than this is processed in several tiles.
 */
constexpr u32 kTileAtoms = 512;

/**
 * Per-block GLOBAL scratch: page bitmap, min/max range, staged i-atom tile.
 *
 * Global rather than shared because every one of these must survive a
 * co_await -- see the note at the top of PairLJCutCoro.
 */
CTP_INLINE_CROSS_FUN clio::run::u64 ScratchBytesPerBlock() {
  return static_cast<clio::run::u64>(kPageBitmapWords) * sizeof(clio::run::u32) +
         2 * sizeof(clio::run::u64) +
         static_cast<clio::run::u64>(kTileAtoms) * 3 * sizeof(float) +
         static_cast<clio::run::u64>(kTileAtoms) * sizeof(int);
}

#if !CTP_IS_DEVICE_PASS
std::string g_last_error;

void SetError(const char *what) { g_last_error = what; }
#endif

#if defined(CLIO_YIELD_CORO)
constexpr u32 kYieldLaneBytes = 4096;
#else
constexpr u32 kYieldLaneBytes = 256;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define ETERNIA_LMP_CORO 1
#endif

}  // namespace

/**
 * Per-type LJ coefficients, small enough to live in ordinary device memory.
 *
 * Deliberately NOT paged: it is O(ntypes^2), which is kilobytes even for a
 * pathological type count, and it is read by every single pair. Paging it
 * would add a hold to the innermost loop to save nothing.
 */
struct LJParams {
  const float *lj1 = nullptr;
  const float *lj2 = nullptr;
  const float *lj3 = nullptr;
  const float *lj4 = nullptr;
  const float *offset = nullptr;
  const float *cutsq = nullptr;
  int ntypes_p1 = 0;
};

/**
 * HOST ONLY. clang compiles this file TWICE -- once for the host and once for
 * the device -- and gv::Vector, the CTE Client and the yield driver are all
 * declared behind `#if !CTP_IS_DEVICE_PASS`, because none of them can exist on
 * a device. Without this guard the device pass reports every one of them as an
 * unknown name, in a couple of hundred errors that read like a broken
 * toolchain rather than like a missing #if.
 */
#if !CTP_IS_DEVICE_PASS
struct Context {
  /** Set by the upload functions, consumed and cleared by Compute. Kept here
   *  rather than computed at launch so an upload cannot happen without the
   *  matching cache drop -- the two would otherwise be free to disagree, and
   *  the failure that produces is a plausible energy from stale coordinates.
   *  Start true so the first launch drops everything. */
  bool x_dirty = true;
  bool type_dirty = true;
  bool neigh_dirty = true;

  Config cfg;
  int nall = 0;
  int inum = 0;
  /** offset[inum] -- how many neighbour entries the kernel MUST examine. */
  std::uint64_t total_entries = 0;

  gv::Vector<float> *x = nullptr;
  gv::Vector<int> *type = nullptr;
  gv::Vector<int> *neigh = nullptr;
  gv::Vector<float> *f = nullptr;

  /** Neighbour-list index: not paged. `offset` is inum+1 entries and `ilist`
   *  is inum -- both O(atoms) but ONE int each, against the neighbour array's
   *  ~100 ints per atom. Paging the small arrays would cost a hold per atom
   *  to save well under 1% of the bytes. */
  int *d_offset = nullptr;
  int *d_ilist = nullptr;

  LJParams lj;
  float *d_lj_pool = nullptr;   // one allocation backing all six lj arrays
  double *d_energy = nullptr;
  double *d_virial = nullptr;
  unsigned long long *d_pairs = nullptr;
  char *d_scratch = nullptr;

  double energy = 0.0;
  double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Stats stats;

  /** False when the last UploadNeighbors rejected the list. Without it a
   *  rejected upload would leave inum at 0 and the kernel would sail through
   *  computing nothing -- the same silent-zero failure the probe-only scan
   *  produced. */
  bool state_valid = false;
};
#endif  // !CTP_IS_DEVICE_PASS

#if defined(ETERNIA_LMP_CORO)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/** Page of x that atom `a` falls in. */
__device__ inline u64 AtomPage(const gv::DeviceVector<float> &x, u64 a) {
  return x.PageOf(a * kPosStride);
}

/** Wait out this block's outstanding writebacks by PARKING, not spinning. */
__device__ gy::YCoroTask FlushWaitCoro(gv::DeviceVector<float> &v) {
  CLIO_CO_YIELD_WHEN((v.ReapFlushed(), v.ReapFetched()),
                     v.AnyTransferInFlight(), v.FlushWaitTag());
}

/**
 * The LJ force kernel, page-major (see the file header).
 *
 * One block per chunk of i-atoms, chunks handed out round-robin. Everything
 * the block needs about its own atoms -- positions, types, running force --
 * is kept in shared memory for the whole chunk, so the only repeated paging
 * is over the NEIGHBOUR pages, which is the traffic the cache exists to
 * manage.
 */
__device__ gy::YCoroMain PairLJCutCoro(clio::run::u32 drop_mask,
                                       gv::DeviceVector<float> x,
                                       gv::DeviceVector<int> type,
                                       gv::DeviceVector<int> neigh,
                                       gv::DeviceVector<float> f,
                                       const int *offset, const int *ilist,
                                       int inum, LJParams lj, u64 atoms_per_page,
                                       u32 nblocks, u32 block, int eflag,
                                       double *energy_out, double *virial_out,
                                       unsigned long long *pairs_out,
                                       char *scratch) {
  extern __shared__ char smem_raw[];
  // Shared layout, packed by hand because the sizes are runtime values:
  //   [0]                     page bitmap        kPageBitmapWords u32
  //   [after]                 chunk min/max page 2 u64
  //   [after]                 per-thread partial energy
  // The i-atom positions are NOT cached in shared: they are already in the
  // held x page, and re-reading them from there costs one indexed load while
  // a shared copy would cost atoms_per_page*4 floats the block does not have.
  // PAST the yield machinery's own shared memory, not at offset 0. The
  // coroutine driver keeps its per-block state in the first
  // CLIO_YIELD_SMEM_BYTES of the dynamic shared block; starting the bitmap
  // there overwrites it, and the block then behaves as though it had already
  // finished -- no faults, no work, no error.
  // ONLY the final reductions live in shared memory. Everything that has to
  // survive a page fault lives in GLOBAL per-block scratch, because a
  // co_await can EXIT THE KERNEL and have the driver relaunch this block --
  // at which point the dynamic shared block is whatever the new launch got.
  //
  // Keeping the staged tile and the page bitmap in shared was a real bug, and
  // a quiet one: it did not crash, it produced answers that were nearly right
  // and DIFFERENT ON EVERY RUN, with an error that tracked page size and
  // block count because those decide how often a fault actually suspends.
  // The same binary gave E_pair = -6.7733676, -6.605114 and -6.3337926 on
  // three runs of one input.
  //
  // The reductions below are safe in shared: no co_await runs inside them.
  double *epart = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);

  char *sc = scratch + static_cast<u64>(block) * ScratchBytesPerBlock();
  u32 *touched = reinterpret_cast<u32 *>(sc);
  u64 *range = reinterpret_cast<u64 *>(touched + kPageBitmapWords);
  float *tile_x = reinterpret_cast<float *>(range + 2);
  int *tile_t = reinterpret_cast<int *>(tile_x + 3 * kTileAtoms);

  u64 run = 0;
  double e_local = 0.0;
  // Per-pair virial, accumulated here rather than left to LAMMPS's
  // virial_fdotr_compute(). fdotr sums x.f over local AND ghost atoms, so it
  // is only correct when ghosts carry their share of the force -- which is
  // exactly what a full list with newton off does NOT produce: every atom's
  // force is complete, but only for the atoms this kernel owns, and ghosts
  // get nothing. Falling back to fdotr therefore reported a pressure that was
  // wrong rather than missing (kinetic term only, virial silently zero).
  double v_local[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  unsigned long long n_pairs = 0;
  unsigned long long n_badtype = 0;   // itype or jtype read back as 0
  unsigned long long n_cut = 0;       // rejected by the cutoff test

  // DROP THE BLOCK'S CACHES BEFORE READING ANYTHING.
  //
  // The host rewrites positions, types and the neighbour list into the CTE on
  // every step, but nothing tells the device page cache that the bytes behind
  // its resident pages have changed -- the cache is private to the block and
  // there is no invalidation protocol. Without this the kernel keeps serving
  // step 0's pages for the whole run: the paging counters showed "x faults 2"
  // on the first step and 0 on every step after, and the trajectory froze with
  // a correct-looking energy computed from stale coordinates.
  //
  // This makes every step re-fault its working set, which is the honest cost
  // of coordinates that change each step. It is also why the eventual fix is
  // to move the integrator onto the vector so positions are updated IN the
  // cache rather than around it.
  //
  // DROP ONLY WHAT THE HOST REWROTE. Dropping all four every launch made the
  // page cache useless: with a cache large enough to hold the entire dataset
  // the counters still read 18 position faults, 77 list faults and 123 force
  // writebacks on EVERY step, identical run to run. The list changes only when
  // it is rebuilt (every 20 steps in the shipped example) and types never
  // change at all, so most of that traffic was re-fetching bytes the device
  // already had.
  //
  // The mask is set by the UPLOAD functions themselves rather than computed
  // here, so a vector cannot be re-uploaded without being dropped -- the
  // failure mode that would produce is a correct-looking energy computed from
  // stale coordinates, which is exactly what the unconditional drop was
  // guarding against.
  __syncthreads();
  if (threadIdx.x == 0) {
    if (drop_mask & 1u) x.DropAll();
    if (drop_mask & 2u) type.DropAll();
    if (drop_mask & 4u) neigh.DropAll();
    if (drop_mask & 8u) f.DropAll();
  }
  __syncthreads();

  const u64 npages_atoms =
      (static_cast<u64>(inum) + atoms_per_page - 1) / atoms_per_page;

  for (u64 chunk = block; chunk < npages_atoms; chunk += nblocks) {
    const u64 a0 = chunk * atoms_per_page;
    const u64 a1 = (a0 + atoms_per_page < static_cast<u64>(inum))
                       ? (a0 + atoms_per_page)
                       : static_cast<u64>(inum);
    if (a0 >= a1) continue;

  // TILE the chunk. Everything below works on [t0, t1), a run of at most
  // kTileAtoms i-atoms whose positions and types are STAGED IN SHARED MEMORY
  // -- see kTileAtoms for why reading them from the page cache instead is a
  // correctness bug rather than a slow path.
  for (u64 t0 = a0; t0 < a1; t0 += kTileAtoms) {
    const u64 t1 = (t0 + kTileAtoms < a1) ? (t0 + kTileAtoms) : a1;

    // Stage the tile: hold the i-atoms' pages, copy out, and never read them
    // through the cache again.
    co_await x.HoldPageCoro(t0 * kPosStride, (t1 - t0) * kPosStride, &run);
    for (u64 a = t0 + threadIdx.x; a < t1; a += blockDim.x) {
      const u64 o = (a - t0) * 3;
      tile_x[o + 0] = x.at(a * kPosStride + 0);
      tile_x[o + 1] = x.at(a * kPosStride + 1);
      tile_x[o + 2] = x.at(a * kPosStride + 2);
    }
    __syncthreads();
    co_await type.HoldPageCoro(t0, t1 - t0, &run);
    for (u64 a = t0 + threadIdx.x; a < t1; a += blockDim.x) {
      tile_t[a - t0] = type.at(a);
    }
    __syncthreads();

    // f is a DIFFERENT vector with its own cache, so holding it here cannot
    // disturb the position pages the neighbour loop is about to churn
    // through. Zeroing is correct rather than accumulating because a full
    // list computes each atom's force in its entirety.
    co_await f.HoldPageCoro(t0 * kPosStride, (t1 - t0) * kPosStride, &run);
    for (u64 a = t0 + threadIdx.x; a < t1; a += blockDim.x) {
      f[a * kPosStride + 0] = 0.0f;
      f[a * kPosStride + 1] = 0.0f;
      f[a * kPosStride + 2] = 0.0f;
      f[a * kPosStride + 3] = 0.0f;
    }
    __syncthreads();

    // The chunk's neighbour entries occupy ONE contiguous range of the
    // neighbour vector, [k0, k1) -- that is what the flattening in
    // UploadNeighbors buys. So the block can walk that range page by page,
    // faulting each page with a proper collective hold.
    //
    // This has to be a real hold, not a probe. An earlier version scanned the
    // list with TryHoldRawConst, which is PROBE-ONLY: it returns null on a
    // miss and never faults. Nothing else touched the neighbour vector, so
    // every probe missed, the touched-page bitmap stayed empty, and the kernel
    // computed nothing at all -- while still reporting success. The run
    // finished with E_pair exactly 0 and zero faults, zero writebacks.
    const u64 k0 = static_cast<u64>(offset[t0]);
    const u64 k1 = static_cast<u64>(offset[t1]);
    const u64 npe = neigh.h_->elems_per_page_;

    for (u64 np = (k1 > k0 ? k0 / npe : 0); k1 > k0 && np <= (k1 - 1) / npe;
         ++np) {
      const u64 e0 = (k0 > np * npe) ? k0 : np * npe;
      const u64 e1 = (k1 < (np + 1) * npe) ? k1 : (np + 1) * npe;
      if (e0 >= e1) continue;

      // Collective: every thread ends up with this page in its last_page_,
      // so the scans below can read it with at().
      co_await neigh.HoldPageCoro(e0, e1 - e0, &run);

      // ---- PASS A: which position pages do THESE entries touch? --------
      if (threadIdx.x == 0) {
        range[0] = ~0ull;   // min
        range[1] = 0ull;    // max
      }
      __syncthreads();
      {
        u64 lo = ~0ull, hi = 0ull;
        for (u64 k = e0 + threadIdx.x; k < e1; k += blockDim.x) {
          const u64 pg = AtomPage(x, static_cast<u64>(neigh.at(k)));
          if (pg < lo) lo = pg;
          if (pg > hi) hi = pg;
        }
        if (lo != ~0ull) {
          atomicMin(reinterpret_cast<unsigned long long *>(&range[0]),
                    static_cast<unsigned long long>(lo));
          atomicMax(reinterpret_cast<unsigned long long *>(&range[1]),
                    static_cast<unsigned long long>(hi));
        }
      }
      __syncthreads();

      const u64 pg_lo = range[0], pg_hi = range[1];
      if (pg_lo == ~0ull) continue;   // no entries fell to this thread set

      // Windowed, because the bitmap is a fixed 2048 pages: entries spanning
      // more than that (unsorted atoms) are processed in several windows
      // rather than being silently truncated.
      for (u64 win = pg_lo; win <= pg_hi; win += kPageBitmapBits) {
        const u64 win_hi = (win + kPageBitmapBits - 1 < pg_hi)
                               ? (win + kPageBitmapBits - 1)
                               : pg_hi;
        for (u32 w = threadIdx.x; w < kPageBitmapWords; w += blockDim.x) {
          touched[w] = 0u;
        }
        __syncthreads();

        for (u64 k = e0 + threadIdx.x; k < e1; k += blockDim.x) {
          const u64 pg = AtomPage(x, static_cast<u64>(neigh.at(k)));
          if (pg >= win && pg <= win_hi) {
            const u32 b = static_cast<u32>(pg - win);
            atomicOr(&touched[b >> 5], 1u << (b & 31u));
          }
        }
        __syncthreads();

        // ---- PASS B: one hold per touched page, all threads compute ----
        for (u64 pg = win; pg <= win_hi; ++pg) {
          const u32 b = static_cast<u32>(pg - win);
          if ((touched[b >> 5] & (1u << (b & 31u))) == 0u) continue;

          // SECOND views of x and type, so holding the neighbour atoms' page
          // does not dislodge the per-thread last_page_ pointing at the
          // i-atoms. They share the block's page table -- this is a
          // register-level alias, not a second cache -- which is also why
          // slots_x must be >= 3: the i-page and the j-page have to be
          // resident together, and a third slot keeps a claim from evicting
          // one of them.
          gv::DeviceVector<float> xn = x;
          gv::DeviceVector<int> tn = type;
          const u64 xpe = x.h_->elems_per_page_;
          co_await xn.HoldPageCoro(pg * xpe, xpe, &run);
          co_await tn.HoldPageCoro(pg * xpe / kPosStride, xpe / kPosStride,
                                   &run);

          for (u64 a = t0 + threadIdx.x; a < t1; a += blockDim.x) {
            const int ii = static_cast<int>(a);
            const int i = ilist[ii];
            // From SHARED, not from the page cache: the j-page hold above
            // may well have evicted the page atom i lives on.
            const u64 so = (a - t0) * 3;
            const float xtmp = tile_x[so + 0];
            const float ytmp = tile_x[so + 1];
            const float ztmp = tile_x[so + 2];
            const int itype = tile_t[a - t0];

            // This atom's entries, clipped to the neighbour page currently
            // held. Both ends matter: reading outside [e0, e1) would index a
            // page this block does not have.
            u64 s = static_cast<u64>(offset[ii]);
            u64 e = static_cast<u64>(offset[ii + 1]);
            if (s < e0) s = e0;
            if (e > e1) e = e1;

            float fx = 0.0f, fy = 0.0f, fz = 0.0f;
            for (u64 k = s; k < e; ++k) {
              const u64 j = static_cast<u64>(neigh.at(k));
              if (AtomPage(x, j) != pg) continue;   // another hold's business
              // Every neighbour entry belongs to EXACTLY ONE position page,
              // so across all pg iterations this must count each entry once.
              // Comparing the total against the host's sum(numneigh) turns
              // "the energy is a bit low" into "N entries were never
              // examined", which is the difference between guessing and
              // knowing.
              ++n_pairs;
              const float delx = xtmp - xn.at(j * kPosStride + 0);
              const float dely = ytmp - xn.at(j * kPosStride + 1);
              const float delz = ztmp - xn.at(j * kPosStride + 2);
              const int jtype = tn.at(j);
              const float rsq = delx * delx + dely * dely + delz * delz;
              const int c = itype * lj.ntypes_p1 + jtype;
              if (itype == 0 || jtype == 0) ++n_badtype;
              if (rsq >= lj.cutsq[c]) { ++n_cut; continue; }

              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
              const float forcelj = r6inv * (lj.lj1[c] * r6inv - lj.lj2[c]);
              const float fpair = forcelj * r2inv;
              fx += delx * fpair;
              fy += dely * fpair;
              fz += delz * fpair;
              if (eflag) {
                // Half, because a full list visits every pair twice.
                e_local += 0.5 * static_cast<double>(
                    r6inv * (lj.lj3[c] * r6inv - lj.lj4[c]) - lj.offset[c]);
              }
              // Same halving, and for the same reason.
              v_local[0] += 0.5 * static_cast<double>(delx * delx * fpair);
              v_local[1] += 0.5 * static_cast<double>(dely * dely * fpair);
              v_local[2] += 0.5 * static_cast<double>(delz * delz * fpair);
              v_local[3] += 0.5 * static_cast<double>(delx * dely * fpair);
              v_local[4] += 0.5 * static_cast<double>(delx * delz * fpair);
              v_local[5] += 0.5 * static_cast<double>(dely * delz * fpair);
            }
            // Accumulate: this hold contributes only the pairs that fell in
            // its page, and the other holds contribute the rest.
            f[static_cast<u64>(i) * kPosStride + 0] += fx;
            f[static_cast<u64>(i) * kPosStride + 1] += fy;
            f[static_cast<u64>(i) * kPosStride + 2] += fz;
          }
          __syncthreads();
        }
      }
    }

    // The chunk's forces are complete; start their writeback and move on.
    // BeginFlush only ISSUES the puts -- FlushWaitCoro below is what makes
    // them durable before the host reads them back.
    __syncthreads();
    if (threadIdx.x == 0) {
      f.BeginFlush(t0 * kPosStride, (t1 - t0) * kPosStride);
    }
    __syncthreads();
    }
  }

  // Per-block energy reduction, then one atomic into the global accumulator.
  if (eflag) {
    epart[threadIdx.x] = e_local;
    __syncthreads();
    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) epart[threadIdx.x] += epart[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(energy_out, epart[0]);
    }
  }

  for (int vc = 0; vc < 6; ++vc) {
    epart[threadIdx.x] = v_local[vc];
    __syncthreads();
    for (u32 s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
      if (threadIdx.x < s2) epart[threadIdx.x] += epart[threadIdx.x + s2];
      __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&virial_out[vc], epart[0]);
    __syncthreads();
  }

  atomicAdd(&pairs_out[0], n_pairs);
  atomicAdd(&pairs_out[1], n_badtype);
  atomicAdd(&pairs_out[2], n_cut);

  // Every force page must be durable before the host reads it: the puts were
  // only issued above.
  co_await FlushWaitCoro(f);
}

__global__ void PairLJCutKernel(clio::run::IpcManagerGpuInfo info,
                                clio::run::u32 drop_mask,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<int> type,
                                gv::DeviceVector<int> neigh,
                                gv::DeviceVector<float> f, const int *offset,
                                const int *ilist, int inum, LJParams lj,
                                u64 atoms_per_page, u32 nblocks, int eflag,
                                double *energy_out, double *virial_out,
                                unsigned long long *pairs_out, char *scratch,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  // Each of the four vectors carries its own page table, and each must be
  // told which logical block this physical block is standing in for -- the
  // yield driver relaunches blocks, so blockIdx.x is NOT the identity.
  x.block_override_ = yv.Block();
  type.block_override_ = yv.Block();
  neigh.block_override_ = yv.Block();
  f.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PairLJCutCoro(drop_mask, x, type, neigh, f, offset, ilist, inum, lj,
                               atoms_per_page, nblocks, yv.Block(), eflag,
                               energy_out, virial_out, pairs_out, scratch));
}

#if !CTP_IS_DEVICE_PASS
/** Drives the yieldable kernel to completion, relaunching after each park. */
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};
#endif  // !CTP_IS_DEVICE_PASS

#endif  // ETERNIA_LMP_CORO

// ---------------------------------------------------------------------------
// Host API. Entirely inside the host pass: it calls into the CTE client and
// the yield driver, neither of which exists on the device side.
// ---------------------------------------------------------------------------
#if !CTP_IS_DEVICE_PASS

bool Available() {
#if defined(ETERNIA_LMP_CORO)
  return true;
#else
  return false;
#endif
}

const char *LastError() { return g_last_error.c_str(); }

namespace {

/** Write one page of a vector straight from host memory.
 *
 *  Goes through the CTE blob path rather than a seeding kernel on purpose: a
 *  kernel would need the whole array staged in VRAM first, which is exactly
 *  the constraint the out-of-core run exists to escape. A vector page IS a
 *  blob named "p<page_num>" under the vector's tag, so the host can fill the
 *  backing store one page at a time and never hold more than one page. */
bool PutPage(clio::cte::core::Client &core, const clio::cte::core::TagId &tag,
             u64 page_num, const char *bytes, u64 nbytes) {
  char name[32];
  gv::PageBlobName(page_num, name);
  auto fut = core.AsyncPutBlob(tag, std::string(name), 0, nbytes, bytes, 1.0f);
  fut.Wait();
  const bool ok = fut.get() != nullptr && fut->GetReturnCode() == 0;
  return ok;
}

bool GetPage(clio::cte::core::Client &core, const clio::cte::core::TagId &tag,
             u64 page_num, char *bytes, u64 nbytes) {
  char name[32];
  gv::PageBlobName(page_num, name);
  auto fut = core.AsyncGetBlob(tag, std::string(name), 0, nbytes, 0u, bytes);
  fut.Wait();
  return fut.get() != nullptr && fut->GetReturnCode() == 0;
}

}  // namespace

Context *Create(const Config &cfg, int nall) {
#if !defined(ETERNIA_LMP_CORO)
  (void)cfg; (void)nall;
  SetError("this LAMMPS was built without the Eternia backend "
           "(needs CUDA and clang++-22 with -DCLIO_GPU_YIELD_CORO=ON)");
  return nullptr;
#else
  if (nall <= 0) {
    SetError("Create: nall must be positive");
    return nullptr;
  }
  // Both of these are documented requirements, and neither was checked. An
  // unenforced constraint here does not fail loudly: the block reductions
  // for energy and virial halve blockDim.x each step, so a thread count that
  // is not a power of two silently DROPS the contributions of the threads
  // above the largest power of two below it, and reports a smaller energy
  // with no error anywhere. That is the same silent-wrong-answer shape as
  // every other defect this backend has had.
  if (cfg.nthreads == 0 || cfg.nthreads > 1024 ||
      (cfg.nthreads & (cfg.nthreads - 1)) != 0) {
    SetError("threads per block must be a power of two between 1 and 1024: "
             "the energy and virial reductions are tree reductions and would "
             "silently drop contributions otherwise");
    return nullptr;
  }
  // The i-atom page, the j-atom page and one spare have to be resident
  // together; with fewer slots a claim can evict the page the block is
  // reading from.
  if (cfg.slots_x < 3) {
    SetError("slots must be at least 3: the i-atom page and the j-atom page "
             "must be resident together, with one spare so a claim cannot "
             "evict either");
    return nullptr;
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    SetError("Clio runtime init failed (is the daemon running?)");
    return nullptr;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    SetError("Clio CTE client init failed");
    return nullptr;
  }

  auto *ctx = new Context();
  ctx->cfg = cfg;
  ctx->nall = nall;

  const std::string p(cfg.tag_prefix);
  const std::vector<int> gpus{cfg.gpu_id};
  try {
    ctx->x = new gv::Vector<float>(p + "_x", gpus, cfg.page_bytes, cfg.nblocks,
                                   cfg.slots_x,
                                   static_cast<u64>(nall) * kPosStride);
    ctx->type = new gv::Vector<int>(p + "_type", gpus, cfg.page_bytes,
                                    cfg.nblocks, cfg.slots_type,
                                    static_cast<u64>(nall));
    ctx->f = new gv::Vector<float>(p + "_f", gpus, cfg.page_bytes, cfg.nblocks,
                                   cfg.slots_f,
                                   static_cast<u64>(nall) * kPosStride);
    // The neighbour vector is sized in UploadNeighbors, once the list length
    // is known; it is created there.
  } catch (const std::exception &e) {
    SetError(e.what());
    Destroy(ctx);
    return nullptr;
  }
  if (cfg.stats) {
    ctx->x->EnableStats();
    ctx->f->EnableStats();
  }

  // ZERO-FILL THE FORCE VECTOR'S BACKING STORE.
  //
  // Forces are write-only from the kernel's point of view, but the page cache
  // does not know that: a hold FAULTS the page in before the kernel writes
  // it, and a page whose blob was never created comes back as a failed get.
  // The bytes themselves do not matter -- the kernel zeroes the chunk it
  // holds -- but the failed read does: it leaves the slot holding whatever it
  // held before, and it is indistinguishable from a real read failure.
  //
  // One pass at context creation, which happens at most once per box change.
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    const u64 page_elems = cfg.page_bytes / sizeof(float);
    const u64 npages =
        (static_cast<u64>(nall) * kPosStride + page_elems - 1) / page_elems;
    std::vector<float> zeros(page_elems, 0.0f);
    for (u64 pg = 0; pg < npages; ++pg) {
      if (!PutPage(core, ctx->f->TagId(), pg,
                   reinterpret_cast<const char *>(zeros.data()),
                   zeros.size() * sizeof(float))) {
        SetError("could not initialise the force vector's backing store");
        Destroy(ctx);
        return nullptr;
      }
    }
  }
  if (cudaMalloc(&ctx->d_energy, sizeof(double)) != cudaSuccess ||
      cudaMalloc(&ctx->d_virial, 6 * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&ctx->d_pairs, 3 * sizeof(unsigned long long)) != cudaSuccess ||
      cudaMalloc(&ctx->d_scratch,
                 static_cast<size_t>(cfg.nblocks) * ScratchBytesPerBlock()) !=
          cudaSuccess) {
    SetError("cudaMalloc(energy/virial) failed");
    Destroy(ctx);
    return nullptr;
  }
  return ctx;
#endif
}

void Destroy(Context *ctx) {
  if (ctx == nullptr) return;
#if defined(ETERNIA_LMP_CORO)
  delete ctx->x;
  delete ctx->type;
  delete ctx->neigh;
  delete ctx->f;
  if (ctx->d_offset) cudaFree(ctx->d_offset);
  if (ctx->d_ilist) cudaFree(ctx->d_ilist);
  if (ctx->d_lj_pool) cudaFree(ctx->d_lj_pool);
  if (ctx->d_energy) cudaFree(ctx->d_energy);
  if (ctx->d_virial) cudaFree(ctx->d_virial);
  if (ctx->d_pairs) cudaFree(ctx->d_pairs);
  if (ctx->d_scratch) cudaFree(ctx->d_scratch);
#endif
  delete ctx;
}

void UploadAtoms(Context *ctx, const double *const *x, const int *type,
                 int nall) {
#if defined(ETERNIA_LMP_CORO)
  if (ctx == nullptr) return;
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);

  const u64 page_elems_x = ctx->cfg.page_bytes / sizeof(float);
  const u64 atoms_per_page = page_elems_x / kPosStride;
  const u64 npages =
      (static_cast<u64>(nall) + atoms_per_page - 1) / atoms_per_page;

  std::vector<float> buf(page_elems_x);
  for (u64 pg = 0; pg < npages; ++pg) {
    const u64 a0 = pg * atoms_per_page;
    const u64 a1 = (a0 + atoms_per_page < static_cast<u64>(nall))
                       ? (a0 + atoms_per_page)
                       : static_cast<u64>(nall);
    std::memset(buf.data(), 0, buf.size() * sizeof(float));
    for (u64 a = a0; a < a1; ++a) {
      const u64 o = (a - a0) * kPosStride;
      buf[o + 0] = static_cast<float>(x[a][0]);
      buf[o + 1] = static_cast<float>(x[a][1]);
      buf[o + 2] = static_cast<float>(x[a][2]);
      buf[o + 3] = 0.0f;
    }
    PutPage(core, ctx->x->TagId(), pg,
            reinterpret_cast<const char *>(buf.data()),
            buf.size() * sizeof(float));
  }

  const u64 page_elems_t = ctx->cfg.page_bytes / sizeof(int);
  const u64 tpages =
      (static_cast<u64>(nall) + page_elems_t - 1) / page_elems_t;
  std::vector<int> tbuf(page_elems_t);
  for (u64 pg = 0; pg < tpages; ++pg) {
    const u64 a0 = pg * page_elems_t;
    const u64 a1 = (a0 + page_elems_t < static_cast<u64>(nall))
                       ? (a0 + page_elems_t)
                       : static_cast<u64>(nall);
    std::memset(tbuf.data(), 0, tbuf.size() * sizeof(int));
    for (u64 a = a0; a < a1; ++a) tbuf[a - a0] = type[a];
    PutPage(core, ctx->type->TagId(), pg,
            reinterpret_cast<const char *>(tbuf.data()),
            tbuf.size() * sizeof(int));
  }
#else
  (void)ctx; (void)x; (void)type; (void)nall;
#endif
  // Mark the cache stale for exactly the vectors this rewrote.
  ctx->x_dirty = true;
  ctx->type_dirty = true;
}

void UploadNeighbors(Context *ctx, const int *ilist, const int *numneigh,
                     const int *const *firstneigh, int inum) {
#if defined(ETERNIA_LMP_CORO)
  if (ctx == nullptr) return;
  ctx->inum = inum;
  ctx->state_valid = false;

  // Flatten LAMMPS's MyPage-backed list into one contiguous array plus an
  // offset table. The pointers themselves are not something a device can
  // chase, and the pages they point into are not contiguous.
  // The kernel chunks i-atoms by their PAGE, and a page is a range of atom
  // INDICES -- but the loop variable is a position in the ilist. Those two
  // only coincide when ilist[ii] == ii, and if they diverge the block holds
  // the page for index range [a0,a1) and then reads atoms from somewhere
  // else entirely: wrong forces, no error.
  //
  // In practice they do coincide (a full list over the local atoms is built
  // in index order, and atom_modify sort physically reorders atoms), so this
  // is a precondition to CHECK rather than a case to handle.
  for (int ii = 0; ii < inum; ++ii) {
    if (ilist[ii] != ii) {
      SetError("neighbour list is not in atom-index order (ilist[ii] != ii); "
               "the paged kernel chunks atoms by page and cannot honour a "
               "permuted list");
      ctx->state_valid = false;
      return;
    }
  }

  // 64-bit while summing, because this is the number that overflows first.
  // A 25M-atom run already reaches 1.94e9 entries -- 90% of INT_MAX -- so the
  // headroom above the largest system that fits this machine's VRAM is under
  // a factor of 1.2. Overflowing would not fail: it would wrap the offsets
  // and index the neighbour vector somewhere else entirely.
  std::vector<u64> off64(static_cast<size_t>(inum) + 1);
  off64[0] = 0;
  for (int ii = 0; ii < inum; ++ii) {
    off64[ii + 1] = off64[ii] + static_cast<u64>(numneigh[ilist[ii]]);
  }
  const u64 total = off64[inum];
  if (total > static_cast<u64>(0x7FFFFFFF)) {
    SetError("neighbour list has more than 2^31 entries; the device offset "
             "table is 32-bit and would wrap. Reduce the atom count or the "
             "cutoff, or widen the offset table to 64 bits");
    return;
  }
  std::vector<int> off(static_cast<size_t>(inum) + 1);
  for (int ii = 0; ii <= inum; ++ii) {
    off[ii] = static_cast<int>(off64[ii]);
  }
  ctx->total_entries = total;

  const u64 page_elems = ctx->cfg.page_bytes / sizeof(int);
  delete ctx->neigh;
  ctx->neigh = nullptr;
  try {
    ctx->neigh = new gv::Vector<int>(std::string(ctx->cfg.tag_prefix) + "_neigh",
                                     std::vector<int>{ctx->cfg.gpu_id},
                                     ctx->cfg.page_bytes, ctx->cfg.nblocks,
                                     ctx->cfg.slots_neigh, total);
  } catch (const std::exception &e) {
    SetError(e.what());
    return;
  }
  if (ctx->cfg.stats) ctx->neigh->EnableStats();

  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  const u64 npages = (total + page_elems - 1) / page_elems;
  std::vector<int> buf(page_elems);
  u64 written = 0;
  for (u64 pg = 0; pg < npages; ++pg) {
    std::memset(buf.data(), 0, buf.size() * sizeof(int));
    const u64 e0 = pg * page_elems;
    const u64 e1 = (e0 + page_elems < total) ? (e0 + page_elems) : total;
    // Walk the flattened index back to (atom, k). Linear in the page, not in
    // the whole list, because `written` carries the cursor between pages.
    for (u64 e = e0; e < e1; ++e) {
      // Binary search the offset table for the owning atom.
      int lo = 0, hi = inum;
      while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (static_cast<u64>(off[mid]) <= e) lo = mid; else hi = mid;
      }
      const int ii = lo;
      const int *fn = firstneigh[ilist[ii]];
      // NEIGHMASK is applied here rather than on the device: the special-bond
      // bits LAMMPS packs into the high bits of j would otherwise be read as
      // part of the atom index and index a page that does not exist.
      buf[e - e0] = fn[e - static_cast<u64>(off[ii])] & 0x3FFFFFFF;
    }
    PutPage(core, ctx->neigh->TagId(), pg,
            reinterpret_cast<const char *>(buf.data()),
            buf.size() * sizeof(int));
    written = e1;
  }
  (void)written;

  // The offset and ilist tables stay in ordinary device memory.
  if (ctx->d_offset) cudaFree(ctx->d_offset);
  if (ctx->d_ilist) cudaFree(ctx->d_ilist);
  cudaMalloc(&ctx->d_offset, off.size() * sizeof(int));
  cudaMalloc(&ctx->d_ilist, static_cast<size_t>(inum) * sizeof(int));
  cudaMemcpy(ctx->d_offset, off.data(), off.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_ilist, ilist, static_cast<size_t>(inum) * sizeof(int),
             cudaMemcpyHostToDevice);
  ctx->state_valid = true;
#else
  (void)ctx; (void)ilist; (void)numneigh; (void)firstneigh; (void)inum;
#endif
  // Mark the cache stale for exactly the vectors this rewrote.
  ctx->neigh_dirty = true;
}

void SetLJParams(Context *ctx, const double *lj1, const double *lj2,
                 const double *lj3, const double *lj4, const double *offset,
                 const double *cutsq, int ntypes_p1) {
#if defined(ETERNIA_LMP_CORO)
  if (ctx == nullptr) return;
  const size_t n = static_cast<size_t>(ntypes_p1) * ntypes_p1;
  std::vector<float> pool(6 * n);
  for (size_t i = 0; i < n; ++i) {
    pool[0 * n + i] = static_cast<float>(lj1[i]);
    pool[1 * n + i] = static_cast<float>(lj2[i]);
    pool[2 * n + i] = static_cast<float>(lj3[i]);
    pool[3 * n + i] = static_cast<float>(lj4[i]);
    pool[4 * n + i] = static_cast<float>(offset[i]);
    pool[5 * n + i] = static_cast<float>(cutsq[i]);
  }
  if (ctx->d_lj_pool) cudaFree(ctx->d_lj_pool);
  cudaMalloc(&ctx->d_lj_pool, pool.size() * sizeof(float));
  cudaMemcpy(ctx->d_lj_pool, pool.data(), pool.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  ctx->lj.lj1 = ctx->d_lj_pool + 0 * n;
  ctx->lj.lj2 = ctx->d_lj_pool + 1 * n;
  ctx->lj.lj3 = ctx->d_lj_pool + 2 * n;
  ctx->lj.lj4 = ctx->d_lj_pool + 3 * n;
  ctx->lj.offset = ctx->d_lj_pool + 4 * n;
  ctx->lj.cutsq = ctx->d_lj_pool + 5 * n;
  ctx->lj.ntypes_p1 = ntypes_p1;
#else
  (void)ctx; (void)lj1; (void)lj2; (void)lj3; (void)lj4; (void)offset;
  (void)cutsq; (void)ntypes_p1;
#endif
}

bool ComputeLJCut(Context *ctx, int eflag, int newton_pair) {
#if !defined(ETERNIA_LMP_CORO)
  (void)ctx; (void)eflag; (void)newton_pair;
  SetError("Eternia backend not compiled in");
  return false;
#else
  if (ctx == nullptr || ctx->neigh == nullptr || !ctx->state_valid) {
    if (ctx != nullptr && ctx->neigh != nullptr && !ctx->state_valid) {
      return false;   // UploadNeighbors already set the reason
    }
    SetError("ComputeLJCut: no neighbour list uploaded");
    return false;
  }
  // Newton on would need each pair's force applied to atom j as well, which
  // is a scattered cross-block write into a paged array -- see the file
  // header. The pair style requests a full list and turns newton off, so
  // this is a guard against a configuration that would silently halve the
  // forces, not a limitation discovered at run time.
  if (newton_pair) {
    SetError("pair_style lj/cut/eternia requires newton off for pairs");
    return false;
  }

  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(ctx->cfg.gpu_id);
  cudaMemset(ctx->d_energy, 0, sizeof(double));
  cudaMemset(ctx->d_virial, 0, 6 * sizeof(double));
  cudaMemset(ctx->d_pairs, 0, 3 * sizeof(unsigned long long));

  const u64 atoms_per_page =
      (ctx->cfg.page_bytes / sizeof(float)) / kPosStride;
  // Shared holds ONLY the reduction scratch now; everything that must
  // survive a fault moved to d_scratch in global memory.
  const size_t smem =
      CLIO_YIELD_SMEM_BYTES + ctx->cfg.nthreads * sizeof(double);

  auto xd = ctx->x->GetDevice(ctx->cfg.gpu_id);
  auto td = ctx->type->GetDevice(ctx->cfg.gpu_id);
  auto nd = ctx->neigh->GetDevice(ctx->cfg.gpu_id);
  auto fd = ctx->f->GetDevice(ctx->cfg.gpu_id);

  // f is ALWAYS dropped: it is the output, rewritten every step, and a stale
  // resident force page would be added to rather than replaced.
  u32 drop_mask = (ctx->x_dirty ? 1u : 0u) | (ctx->type_dirty ? 2u : 0u) |
                  (ctx->neigh_dirty ? 4u : 0u) | 8u;
  // DIAGNOSTIC ONLY. ETERNIA_DROP_MASK forces the mask so the question "does
  // the page cache survive a kernel launch at all" can be asked directly.
  // Forcing 0 produces wrong forces by design -- the host has rewritten the
  // coordinates and the kernel would use the previous step's -- so it is for
  // measuring fault counts, never for a result.
  if (const char *e = std::getenv("ETERNIA_DROP_MASK")) {
    drop_mask = static_cast<u32>(std::atoi(e));
  }
  YieldRunner runner(ctx->cfg.nblocks, ctx->cfg.nthreads);
  const auto k_t0 = std::chrono::steady_clock::now();
  const u32 rounds = runner.Run(
      [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
        PairLJCutKernel<<<g, b, smem>>>(gpu, drop_mask, xd, td, nd, fd, ctx->d_offset,
                                        ctx->d_ilist, ctx->inum, ctx->lj,
                                        atoms_per_page, ctx->cfg.nblocks,
                                        eflag, ctx->d_energy, ctx->d_virial,
                                        ctx->d_pairs, ctx->d_scratch, vw, sv);
      });
  // Check the LAUNCH, not just the sync. A bad launch configuration (too much
  // shared memory, too many threads) is reported by cudaGetLastError and can
  // leave cudaDeviceSynchronize returning success -- so the kernel never runs,
  // no page is ever faulted, and every force comes back zero with no error.
  const cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    SetError(cudaGetErrorString(launch_err));
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    SetError(cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  // Cleared only after a SUCCESSFUL launch, so a failed step re-drops next
  // time rather than trusting a cache the kernel may not have refreshed.
  ctx->x_dirty = false;
  ctx->type_dirty = false;
  ctx->neigh_dirty = false;
  ctx->stats.drop_mask = drop_mask;
  ctx->stats.rounds = rounds;
  ctx->stats.kernel_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - k_t0).count();
  if (rounds == 0) {
    // RunToCompletion returning zero rounds means the driver never saw the
    // kernel finish -- a page that never landed, which would otherwise show
    // up as quietly zero forces.
    SetError("yield driver made no progress (a page never landed)");
    return false;
  }
  cudaMemcpy(&ctx->energy, ctx->d_energy, sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(ctx->virial, ctx->d_virial, 6 * sizeof(double),
             cudaMemcpyDeviceToHost);
  unsigned long long pc[3] = {0, 0, 0};
  cudaMemcpy(pc, ctx->d_pairs, 3 * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  ctx->stats.pairs_seen = pc[0];
  ctx->stats.pairs_badtype = pc[1];
  ctx->stats.pairs_cut = pc[2];
  ctx->stats.pairs_expected = ctx->total_entries;

  if (ctx->cfg.stats) {
    auto sx = ctx->x->ReadStats(ctx->cfg.gpu_id);
    auto sn = ctx->neigh->ReadStats(ctx->cfg.gpu_id);
    auto sf = ctx->f->ReadStats(ctx->cfg.gpu_id);
    ctx->stats.x_faults = sx.faults;
    ctx->stats.x_evicts = sx.evicts;
    ctx->stats.neigh_faults = sn.faults;
    ctx->stats.f_puts = sf.puts;
    ctx->stats.f_put_errors = sf.put_errors;
    ctx->stats.get_errors = sx.get_errors + sn.get_errors + sf.get_errors;
  }
  return true;
#endif
}

void DownloadForces(Context *ctx, double *const *f, int nall) {
#if defined(ETERNIA_LMP_CORO)
  if (ctx == nullptr) return;
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  const u64 page_elems = ctx->cfg.page_bytes / sizeof(float);
  const u64 atoms_per_page = page_elems / kPosStride;
  const u64 npages =
      (static_cast<u64>(nall) + atoms_per_page - 1) / atoms_per_page;
  std::vector<float> buf(page_elems);
  for (u64 pg = 0; pg < npages; ++pg) {
    if (!GetPage(core, ctx->f->TagId(), pg,
                 reinterpret_cast<char *>(buf.data()),
                 buf.size() * sizeof(float))) {
      continue;   // a page the kernel never wrote contributes nothing
    }
    const u64 a0 = pg * atoms_per_page;
    const u64 a1 = (a0 + atoms_per_page < static_cast<u64>(nall))
                       ? (a0 + atoms_per_page)
                       : static_cast<u64>(nall);
    for (u64 a = a0; a < a1; ++a) {
      const u64 o = (a - a0) * kPosStride;
      // ADD: LAMMPS sums several force styles into one f array.
      f[a][0] += buf[o + 0];
      f[a][1] += buf[o + 1];
      f[a][2] += buf[o + 2];
    }
  }
#else
  (void)ctx; (void)f; (void)nall;
#endif
}

double GetEnergy(Context *ctx) { return ctx ? ctx->energy : 0.0; }

void GetVirial(Context *ctx, double *out6) {
  for (int i = 0; i < 6; ++i) out6[i] = ctx ? ctx->virial[i] : 0.0;
}

Stats GetStats(Context *ctx) { return ctx ? ctx->stats : Stats(); }

void ResetStats(Context *ctx) {
#if defined(ETERNIA_LMP_CORO)
  if (ctx == nullptr) return;
  if (ctx->x) ctx->x->ResetStats();
  if (ctx->neigh) ctx->neigh->ResetStats();
  if (ctx->f) ctx->f->ResetStats();
  ctx->stats = Stats();
#else
  (void)ctx;
#endif
}

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace eternia_lammps

#else   // !ETERNIA_LAMMPS_ENABLED

// ---------------------------------------------------------------------------
// STUB BUILD. Compiled by the ordinary compiler when CUDA or a
// coroutine-capable clang is missing, so `pair_style lj/cut/eternia` fails
// with a sentence the user can act on instead of a link error.
// ---------------------------------------------------------------------------
namespace eternia_lammps {

struct Context {};

bool Available() { return false; }
const char *LastError() {
  return "this LAMMPS was built without the Eternia backend (needs CUDA and "
         "clang++-22 with -DCLIO_GPU_YIELD_CORO=ON)";
}
Context *Create(const Config &, int) { return nullptr; }
void Destroy(Context *) {}
void UploadAtoms(Context *, const double *const *, const int *, int) {}
void UploadNeighbors(Context *, const int *, const int *, const int *const *,
                     int) {}
void SetLJParams(Context *, const double *, const double *, const double *,
                 const double *, const double *, const double *, int) {}
bool ComputeLJCut(Context *, int, int) { return false; }
void DownloadForces(Context *, double *const *, int) {}
double GetEnergy(Context *) { return 0.0; }
void GetVirial(Context *, double *out6) {
  for (int i = 0; i < 6; ++i) out6[i] = 0.0;
}
Stats GetStats(Context *) { return Stats(); }
void ResetStats(Context *) {}

}  // namespace eternia_lammps

#endif  // ETERNIA_LAMMPS_ENABLED
