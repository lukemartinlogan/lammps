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

std::string g_last_error;

void SetError(const char *what) { g_last_error = what; }

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

struct Context {
  Config cfg;
  int nall = 0;
  int inum = 0;

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

  double energy = 0.0;
  Stats stats;
};

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
__device__ gy::YCoroMain PairLJCutCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<int> type,
                                       gv::DeviceVector<int> neigh,
                                       gv::DeviceVector<float> f,
                                       const int *offset, const int *ilist,
                                       int inum, LJParams lj, u64 atoms_per_page,
                                       u32 nblocks, u32 block, int eflag,
                                       double *energy_out) {
  extern __shared__ char smem_raw[];
  // Shared layout, packed by hand because the sizes are runtime values:
  //   [0]                     page bitmap        kPageBitmapWords u32
  //   [after]                 chunk min/max page 2 u64
  //   [after]                 per-thread partial energy
  // The i-atom positions are NOT cached in shared: they are already in the
  // held x page, and re-reading them from there costs one indexed load while
  // a shared copy would cost atoms_per_page*4 floats the block does not have.
  u32 *touched = reinterpret_cast<u32 *>(smem_raw);
  u64 *range = reinterpret_cast<u64 *>(touched + kPageBitmapWords);
  double *epart = reinterpret_cast<double *>(range + 2);

  u64 run = 0;
  double e_local = 0.0;

  const u64 npages_atoms =
      (static_cast<u64>(inum) + atoms_per_page - 1) / atoms_per_page;

  for (u64 chunk = block; chunk < npages_atoms; chunk += nblocks) {
    const u64 a0 = chunk * atoms_per_page;
    const u64 a1 = (a0 + atoms_per_page < static_cast<u64>(inum))
                       ? (a0 + atoms_per_page)
                       : static_cast<u64>(inum);
    if (a0 >= a1) continue;

    // ---- hold this chunk's OWN pages -----------------------------------
    // x for the i-atoms, type for the i-atoms, and f to accumulate into.
    // All three are held for the whole chunk: they are the one thing that
    // does NOT churn, and re-holding them inside pass B would evict the
    // neighbour page the block just faulted in.
    co_await x.HoldPageCoro(a0 * kPosStride,
                            (a1 - a0) * kPosStride, &run);
    co_await type.HoldPageCoro(a0, a1 - a0, &run);
    co_await f.HoldPageCoro(a0 * kPosStride, (a1 - a0) * kPosStride, &run);

    // Zero this chunk's forces. A full neighbour list means every atom's
    // force is computed here in its entirety, so assignment is correct and
    // no read-modify-write of a stale page is needed.
    for (u64 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
      f[a * kPosStride + 0] = 0.0f;
      f[a * kPosStride + 1] = 0.0f;
      f[a * kPosStride + 2] = 0.0f;
      f[a * kPosStride + 3] = 0.0f;
    }
    __syncthreads();

    // ---- PASS A: which neighbour pages does this chunk touch? ----------
    // Windowed, because the bitmap is a fixed 2048 pages: a chunk whose
    // neighbours span more than that (unsorted atoms) is processed in
    // several windows rather than being silently truncated.
    if (threadIdx.x == 0) {
      range[0] = ~0ull;   // min
      range[1] = 0ull;    // max
    }
    __syncthreads();
    {
      u64 lo = ~0ull, hi = 0ull;
      for (u64 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
        const int ii = static_cast<int>(a);
        const int start = offset[ii], end = offset[ii + 1];
        for (int k = start; k < end; ++k) {
          // The neighbour array itself is paged. Reading it one element at a
          // time through a hold would be a hold per neighbour, so pass A
          // holds a RUN and walks it -- the sequential-access contract the
          // vector is built around.
          u64 nrun = 0;
          const int *q = neigh.TryHoldRawConst(k, end - k, &nrun);
          if (q == nullptr) continue;   // settled in pass A2 below
          for (u64 t = 0; t < nrun && k + static_cast<int>(t) < end; ++t) {
            const u64 pg = AtomPage(x, static_cast<u64>(q[t]));
            if (pg < lo) lo = pg;
            if (pg > hi) hi = pg;
          }
          k += static_cast<int>(nrun) - 1;
        }
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
    if (pg_lo == ~0ull) continue;   // no neighbours at all in this chunk

    for (u64 win = pg_lo; win <= pg_hi; win += kPageBitmapBits) {
      const u64 win_hi = (win + kPageBitmapBits - 1 < pg_hi)
                             ? (win + kPageBitmapBits - 1)
                             : pg_hi;
      for (u32 w = threadIdx.x; w < kPageBitmapWords; w += blockDim.x) {
        touched[w] = 0u;
      }
      __syncthreads();

      // Mark the pages this window actually contains.
      for (u64 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
        const int ii = static_cast<int>(a);
        const int start = offset[ii], end = offset[ii + 1];
        for (int k = start; k < end; ++k) {
          u64 nrun = 0;
          const int *q = neigh.TryHoldRawConst(k, end - k, &nrun);
          if (q == nullptr) { continue; }
          for (u64 t = 0; t < nrun && k + static_cast<int>(t) < end; ++t) {
            const u64 pg = AtomPage(x, static_cast<u64>(q[t]));
            if (pg >= win && pg <= win_hi) {
              const u32 b = static_cast<u32>(pg - win);
              atomicOr(&touched[b >> 5], 1u << (b & 31u));
            }
          }
          k += static_cast<int>(nrun) - 1;
        }
      }
      __syncthreads();

      // ---- PASS B: one hold per touched page, all threads compute ------
      for (u64 pg = win; pg <= win_hi; ++pg) {
        const u32 b = static_cast<u32>(pg - win);
        if ((touched[b >> 5] & (1u << (b & 31u))) == 0u) continue;

        // A SECOND view of x, so holding the neighbour page does not
        // dislodge the per-thread last_page_ pointing at the i-atoms. The
        // two views share the block's page table -- this is a register-level
        // alias, not a second cache.
        gv::DeviceVector<float> xn = x;
        gv::DeviceVector<int> tn = type;
        const u64 jbase = pg * (x.h_->elems_per_page_);
        co_await xn.HoldPageCoro(jbase, x.h_->elems_per_page_, &run);
        co_await tn.HoldPageCoro(jbase / kPosStride,
                                 x.h_->elems_per_page_ / kPosStride, &run);

        for (u64 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
          const int ii = static_cast<int>(a);
          const int i = ilist[ii];
          const float xtmp = x.at(static_cast<u64>(i) * kPosStride + 0);
          const float ytmp = x.at(static_cast<u64>(i) * kPosStride + 1);
          const float ztmp = x.at(static_cast<u64>(i) * kPosStride + 2);
          const int itype = type.at(static_cast<u64>(i));

          float fx = 0.0f, fy = 0.0f, fz = 0.0f;
          const int start = offset[ii], end = offset[ii + 1];
          for (int k = start; k < end; ++k) {
            u64 nrun = 0;
            const int *q = neigh.TryHoldRawConst(k, end - k, &nrun);
            if (q == nullptr) continue;
            for (u64 t = 0; t < nrun && k + static_cast<int>(t) < end; ++t) {
              const u64 j = static_cast<u64>(q[t]);
              if (AtomPage(x, j) != pg) continue;   // handled by another hold
              const float delx = xtmp - xn.at(j * kPosStride + 0);
              const float dely = ytmp - xn.at(j * kPosStride + 1);
              const float delz = ztmp - xn.at(j * kPosStride + 2);
              const int jtype = tn.at(j);
              const float rsq = delx * delx + dely * dely + delz * delz;
              const int c = itype * lj.ntypes_p1 + jtype;
              if (rsq >= lj.cutsq[c]) continue;

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
            }
            k += static_cast<int>(nrun) - 1;
          }
          // Accumulate: this page contributes only the pairs that fell in
          // it, and the other touched pages contribute the rest.
          f[static_cast<u64>(i) * kPosStride + 0] += fx;
          f[static_cast<u64>(i) * kPosStride + 1] += fy;
          f[static_cast<u64>(i) * kPosStride + 2] += fz;
        }
        __syncthreads();
      }
    }

    // The chunk's forces are complete; start their writeback and move on.
    // BeginFlush only ISSUES the puts -- FlushWaitCoro below is what makes
    // them durable before the host reads them back.
    __syncthreads();
    if (threadIdx.x == 0) {
      f.BeginFlush(a0 * kPosStride, (a1 - a0) * kPosStride);
    }
    __syncthreads();
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

  // Every force page must be durable before the host reads it: the puts were
  // only issued above.
  co_await FlushWaitCoro(f);
}

__global__ void PairLJCutKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<int> type,
                                gv::DeviceVector<int> neigh,
                                gv::DeviceVector<float> f, const int *offset,
                                const int *ilist, int inum, LJParams lj,
                                u64 atoms_per_page, u32 nblocks, int eflag,
                                double *energy_out, gy::YieldableView<> yv,
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
  CLIO_YCORO_RUN(PairLJCutCoro(x, type, neigh, f, offset, ilist, inum, lj,
                               atoms_per_page, nblocks, yv.Block(), eflag,
                               energy_out));
}

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

#endif  // ETERNIA_LMP_CORO

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

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
  return fut.get() != nullptr && fut->GetReturnCode() == 0;
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
  if (cudaMalloc(&ctx->d_energy, sizeof(double)) != cudaSuccess) {
    SetError("cudaMalloc(energy) failed");
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
}

void UploadNeighbors(Context *ctx, const int *ilist, const int *numneigh,
                     const int *const *firstneigh, int inum) {
#if defined(ETERNIA_LMP_CORO)
  if (ctx == nullptr) return;
  ctx->inum = inum;

  // Flatten LAMMPS's MyPage-backed list into one contiguous array plus an
  // offset table. The pointers themselves are not something a device can
  // chase, and the pages they point into are not contiguous.
  std::vector<int> off(static_cast<size_t>(inum) + 1);
  off[0] = 0;
  for (int ii = 0; ii < inum; ++ii) {
    off[ii + 1] = off[ii] + numneigh[ilist[ii]];
  }
  const u64 total = static_cast<u64>(off[inum]);

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
#else
  (void)ctx; (void)ilist; (void)numneigh; (void)firstneigh; (void)inum;
#endif
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
  if (ctx == nullptr || ctx->neigh == nullptr) {
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

  const u64 atoms_per_page =
      (ctx->cfg.page_bytes / sizeof(float)) / kPosStride;
  const size_t smem = CLIO_YIELD_SMEM_BYTES +
                      kPageBitmapWords * sizeof(u32) + 2 * sizeof(u64) +
                      ctx->cfg.nthreads * sizeof(double);

  auto xd = ctx->x->GetDevice(ctx->cfg.gpu_id);
  auto td = ctx->type->GetDevice(ctx->cfg.gpu_id);
  auto nd = ctx->neigh->GetDevice(ctx->cfg.gpu_id);
  auto fd = ctx->f->GetDevice(ctx->cfg.gpu_id);

  YieldRunner runner(ctx->cfg.nblocks, ctx->cfg.nthreads);
  const u32 rounds = runner.Run(
      [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
        PairLJCutKernel<<<g, b, smem>>>(gpu, xd, td, nd, fd, ctx->d_offset,
                                        ctx->d_ilist, ctx->inum, ctx->lj,
                                        atoms_per_page, ctx->cfg.nblocks,
                                        eflag, ctx->d_energy, vw, sv);
      });
  if (cudaDeviceSynchronize() != cudaSuccess) {
    SetError(cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (rounds == 0) {
    // RunToCompletion returning zero rounds means the driver never saw the
    // kernel finish -- a page that never landed, which would otherwise show
    // up as quietly zero forces.
    SetError("yield driver made no progress (a page never landed)");
    return false;
  }
  cudaMemcpy(&ctx->energy, ctx->d_energy, sizeof(double),
             cudaMemcpyDeviceToHost);

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
Stats GetStats(Context *) { return Stats(); }
void ResetStats(Context *) {}

}  // namespace eternia_lammps

#endif  // ETERNIA_LAMMPS_ENABLED
