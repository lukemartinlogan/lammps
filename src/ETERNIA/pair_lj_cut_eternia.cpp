/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   ETERNIA package -- out-of-core lj/cut.
------------------------------------------------------------------------- */

#include "pair_lj_cut_eternia.h"

#include <cstdlib>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJCutEternia::PairLJCutEternia(LAMMPS *lmp) :
    PairLJCut(lmp), ctx(nullptr), ctx_nall(0), params_dirty(1)
{
  // A full list is not a preference, it is what makes the paged write path
  // possible at all: with a half list each pair must also be applied to atom
  // j, which lives in another page and usually another block's cache, and
  // the per-block caches deliberately do not support cross-block writes.
  one_coeff = 0;
  single_enable = 0;    // single() would need a resident position array
  respa_enable = 0;
  manybody_flag = 0;
}

/* ---------------------------------------------------------------------- */

PairLJCutEternia::~PairLJCutEternia()
{
  if (ctx) eternia_lammps::Destroy(ctx);
}

/* ----------------------------------------------------------------------
   pair_style lj/cut/eternia <cutoff> [keyword value ...]
------------------------------------------------------------------------- */

void PairLJCutEternia::settings(int narg, char **arg)
{
  if (narg < 1) utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia", error);

  // Cutoff handling is entirely PairLJCut's; only the first argument belongs
  // to it, so the keywords are stripped off before delegating.
  char *cutarg[1] = {arg[0]};
  PairLJCut::settings(1, cutarg);

  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "page") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia page", error);
      cfg.page_bytes = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp) * 1024;
      iarg += 2;
    } else if (strcmp(arg[iarg], "blocks") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia blocks", error);
      cfg.nblocks = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "threads") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia threads", error);
      cfg.nthreads = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "slots") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia slots", error);
      // `slots` sets ALL FOUR vectors, not just positions. It previously set
      // slots_x alone and left neigh at 4 and f at 8, so the neighbour list --
      // by far the largest array, 45 MB of a 53 MB dataset at 62,500 atoms --
      // thrashed a four-page cache no matter how large the total was made.
      // Every one of those misses suspends the block, and a suspension costs a
      // whole grid teardown and relaunch, which is what made this kernel slow.
      //
      // Positions need >= 3 (i-page, j-page, spare). The others are given the
      // same budget rather than a fixed fraction: which vector dominates is a
      // property of the system, not of the style.
      const int sl = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      cfg.slots_x = sl;
      cfg.slots_neigh = sl;
      cfg.slots_f = sl;
      cfg.slots_type = (sl < 4) ? sl : 4;   // types are one int per atom
      iarg += 2;
    } else if (strcmp(arg[iarg], "tag") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia tag", error);
      cfg.tag_prefix = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "stats") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style lj/cut/eternia stats", error);
      cfg.stats = (strcmp(arg[iarg + 1], "on") == 0);
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown pair_style lj/cut/eternia keyword: {}", arg[iarg]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::init_style()
{
  // Under ETERNIA_BASELINE this style must behave as PairLJCut in EVERY
  // respect, not just in compute(). The paged kernel asks for a FULL neighbour
  // list because it writes only its own i-atoms; the stock kernel expects a
  // HALF list. Delegating compute() while still requesting a full list makes
  // the stock kernel visit every pair twice -- measured E_pair = -10.570786
  // against the correct -4.7855792, a 2.2x error that looks like a physics
  // difference rather than a list-type mismatch.
  //
  // The `newton off` requirement is also lifted here: it exists for the paged
  // kernel's sake, and stock lj/cut is happy either way.
  if (baseline_mode()) {
    PairLJCut::init_style();
    return;
  }

  // init_style runs before every run command, and the coefficients may have
  // been changed by a pair_coeff since the last one. The context outlives a
  // run, so unless they are re-pushed the device keeps the previous run's
  // values -- wrong forces, no error.
  params_dirty = 1;

  if (!eternia_lammps::Available())
    error->all(FLERR, "pair_style lj/cut/eternia: {}", eternia_lammps::LastError());

  if (force->newton_pair)
    error->all(FLERR,
               "pair_style lj/cut/eternia requires 'newton off' for pairs: a "
               "half list would need each pair's force written to atom j, "
               "which lives in another block's page cache");

  if (comm->nprocs > 1)
    error->all(FLERR,
               "pair_style lj/cut/eternia is single-rank for now (each rank "
               "would need its own CTE tag prefix and GPU)");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::ensure_context()
{
  const int nall = atom->nlocal + atom->nghost;

  // The vectors' logical length is fixed at construction, so a grown atom
  // count means a new context. Rebuilding is cheap relative to a
  // reneighbouring step and happens at most once per box change.
  if (ctx && nall > ctx_nall) {
    eternia_lammps::Destroy(ctx);
    ctx = nullptr;
  }
  if (!ctx) {
    // Headroom, so ordinary ghost-count jitter does not rebuild every step.
    ctx_nall = nall + nall / 8 + 64;
    ctx = eternia_lammps::Create(cfg, ctx_nall);
    if (!ctx)
      error->all(FLERR, "pair_style lj/cut/eternia: {}", eternia_lammps::LastError());

  }

  // Push the coefficients whenever they may have changed, NOT only when the
  // context is created -- see params_dirty.
  if (params_dirty) {
    const int np1 = atom->ntypes + 1;
    // LAMMPS stores these as double** from memory->create; the rows are
    // contiguous, so lj1[0] is the flat [np1*np1] block the backend wants.
    eternia_lammps::SetLJParams(ctx, lj1[0], lj2[0], lj3[0], lj4[0], offset[0],
                                cutsq[0], np1);
    params_dirty = 0;
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::push_state()
{
  const int nall = atom->nlocal + atom->nghost;
  eternia_lammps::UploadAtoms(ctx, atom->x, atom->type, nall);

  // The neighbour list only changes when it is REBUILT. Re-uploading it on
  // every step re-sent 45 MB per step at 62,500 atoms for bytes the device
  // already had, and -- worse -- forced the kernel to drop its list cache
  // every launch, because the host had rewritten the backing store underneath
  // it. The counters showed the result plainly: 77 list faults per step, the
  // same on every step, with a page cache large enough to hold the whole
  // dataset several times over.
  //
  // `neighbor->ago` is 0 on the step the list was rebuilt. Atom sorting also
  // invalidates the flattening -- the list indexes atoms by their current
  // ordering -- but LAMMPS sorts during reneighbouring, so the same test
  // covers it. A changed atom count means a new context anyway.
  //
  // Erring towards uploading is the safe direction: a needless upload costs
  // time, a missed one computes forces from a stale list.
  if (neighbor->ago == 0 || !neigh_pushed_once) {
    eternia_lammps::UploadNeighbors(ctx, list->ilist, list->numneigh,
                                    list->firstneigh, list->inum);
    neigh_pushed_once = 1;
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::compute(int eflag, int vflag)
{
  // ETERNIA_BASELINE runs the STOCK lj/cut kernel from the parent class
  // instead of the paged one. Same binary, same input script, one environment
  // variable -- so a performance comparison does not need a second build or a
  // second input, and nothing but the kernel differs between the two runs.
  //
  // Announced once, loudly. A run that says lj/cut/eternia in its input and
  // silently computed something else would be the worst kind of benchmark
  // result: correct, fast, and measuring the wrong thing.
  //
  // NOTE the baseline here is LAMMPS's CPU pair style. This build has no GPU
  // package, so this comparison is paged-GPU against stock-CPU and is NOT a
  // like-for-like GPU measurement; see the ETERNIA README.
  if (baseline_mode()) {
    if (comm->me == 0 && !baseline_announced) {
      baseline_announced = 1;
      utils::logmesg(lmp, "lj/cut/eternia: ETERNIA_BASELINE set, running the "
                          "stock lj/cut CPU kernel instead of the paged one\n");
    }
    PairLJCut::compute(eflag, vflag);
    return;
  }

  ev_init(eflag, vflag);

  ensure_context();
  if (cfg.stats) eternia_lammps::ResetStats(ctx);

  // Positions AND the neighbour list go up every step. The list is only
  // rebuilt on reneighbouring, but the flattened form indexes atoms by their
  // current ordering, so anything that reorders atoms (a sort, an exchange)
  // invalidates the flattening even when the list itself is unchanged.
  push_state();

  if (!eternia_lammps::ComputeLJCut(ctx, eflag, force->newton_pair))
    error->all(FLERR, "pair_style lj/cut/eternia: {}", eternia_lammps::LastError());

  eternia_lammps::DownloadForces(ctx, atom->f, atom->nlocal + atom->nghost);

  if (eflag_global) eng_vdwl += eternia_lammps::GetEnergy(ctx);

  // The virial comes back from the kernel, computed per pair. It must NOT be
  // left to virial_fdotr_compute(): that sums x.f over local and ghost atoms,
  // and a full list with newton off gives ghosts no force, so fdotr here
  // silently reported the kinetic term alone.
  if (vflag_global) {
    double v[6];
    eternia_lammps::GetVirial(ctx, v);
    for (int k = 0; k < 6; k++) virial[k] += v[k];
  }

  if (cfg.stats) report_stats();
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::report_stats()
{
  const eternia_lammps::Stats s = eternia_lammps::GetStats(ctx);
  if (comm->me == 0) {
    utils::logmesg(lmp,
                   "eternia step {}: mask {} ago {} rounds {} kernel_ms {:.2f} [launch {:.1f} copy {:.1f} upl {:.1f}] scanned {} holds {} | x faults {} evicts {} | neigh faults {} | "
                   "f puts {} (errors {}) | get errors {} | pairs {}/{} passb_cyc {} hold_cyc {}\n",
                   update->ntimestep, s.drop_mask, neighbor->ago, s.rounds, s.kernel_ms,
                   s.t_launch_ms, s.t_copy_ms, s.t_upload_ms, s.entries_scanned, s.block_launches,
                   s.x_faults, s.x_evicts, s.neigh_faults,
                   s.f_puts, s.f_put_errors, s.get_errors, s.pairs_seen,
                   s.pairs_expected, s.pairs_badtype, s.pairs_cut);
  }
  // A non-zero error count means pages were silently dropped: forces would be
  // wrong, not merely slow, so this is fatal rather than a warning.
  // Every neighbour entry belongs to exactly one position page, so anything
  // other than equality means the paged walk skipped pairs -- which the
  // energy only reports as a small deficit.
  if (s.pairs_seen != s.pairs_expected)
    error->all(FLERR,
               "pair_style lj/cut/eternia: the kernel examined {} neighbour "
               "entries but the list holds {} -- pairs were skipped",
               s.pairs_seen, s.pairs_expected);

  if (s.f_put_errors || s.get_errors)
    error->all(FLERR,
               "pair_style lj/cut/eternia: {} failed page writebacks and {} "
               "failed reads -- forces are incomplete",
               s.f_put_errors, s.get_errors);
}
