/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   ETERNIA package -- out-of-core lj/cut.
------------------------------------------------------------------------- */

#include "pair_lj_cut_eternia.h"

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
    PairLJCut(lmp), ctx(nullptr), ctx_nall(0)
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
  if (narg < 1) error->all(FLERR, "Illegal pair_style lj/cut/eternia command");

  // Cutoff handling is entirely PairLJCut's; only the first argument belongs
  // to it, so the keywords are stripped off before delegating.
  char *cutarg[1] = {arg[0]};
  PairLJCut::settings(1, cutarg);

  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "page") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style command: page");
      cfg.page_bytes = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp) * 1024;
      iarg += 2;
    } else if (strcmp(arg[iarg], "blocks") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style command: blocks");
      cfg.nblocks = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "threads") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style command: threads");
      cfg.nthreads = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "slots") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style command: slots");
      cfg.slots_x = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "tag") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style command: tag");
      cfg.tag_prefix = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "stats") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal pair_style command: stats");
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

    const int np1 = atom->ntypes + 1;
    // LAMMPS stores these as double** from memory->create; the rows are
    // contiguous, so lj1[0] is the flat [np1*np1] block the backend wants.
    eternia_lammps::SetLJParams(ctx, lj1[0], lj2[0], lj3[0], lj4[0], offset[0],
                                cutsq[0], np1);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::push_state()
{
  const int nall = atom->nlocal + atom->nghost;
  eternia_lammps::UploadAtoms(ctx, atom->x, atom->type, nall);
  eternia_lammps::UploadNeighbors(ctx, list->ilist, list->numneigh,
                                  list->firstneigh, list->inum);
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::compute(int eflag, int vflag)
{
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

  // The virial is not computed on the device yet, so ask LAMMPS for the
  // global fdotr form rather than reporting zero pressure -- which would be
  // wrong quietly, in a way an NPT run would not survive.
  if (vflag_fdotr) virial_fdotr_compute();

  if (cfg.stats) report_stats();
}

/* ---------------------------------------------------------------------- */

void PairLJCutEternia::report_stats()
{
  const eternia_lammps::Stats s = eternia_lammps::GetStats(ctx);
  if (comm->me == 0) {
    utils::logmesg(lmp,
                   "eternia step {}: x faults {} evicts {} | neigh faults {} | "
                   "f puts {} (errors {}) | get errors {}\n",
                   update->ntimestep, s.x_faults, s.x_evicts, s.neigh_faults,
                   s.f_puts, s.f_put_errors, s.get_errors);
  }
  // A non-zero error count means pages were silently dropped: forces would be
  // wrong, not merely slow, so this is fatal rather than a warning.
  if (s.f_put_errors || s.get_errors)
    error->all(FLERR,
               "pair_style lj/cut/eternia: {} failed page writebacks and {} "
               "failed reads -- forces are incomplete",
               s.f_put_errors, s.get_errors);
}
