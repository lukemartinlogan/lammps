/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   This file is part of the ETERNIA package: an out-of-core pair style whose
   per-atom arrays live in the Clio CTE and are paged into GPU memory from
   inside the force kernel.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/eternia,PairLJCutEternia);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_ETERNIA_H
#define LMP_PAIR_LJ_CUT_ETERNIA_H

#include "pair_lj_cut.h"

// The boundary header: no CUDA, no Clio. See lib/eternia/eternia_lammps.h
// for why the split exists.
#include "eternia_lammps.h"

namespace LAMMPS_NS {

/**
 * lj/cut, with x / type / neighbours / f held out of core.
 *
 * Derives from PairLJCut purely to inherit the coefficient bookkeeping
 * (settings, coeff, init_one, restart, the lj1..lj4 tables). compute() is
 * replaced wholesale -- it does no pair arithmetic itself, it hands the
 * arrays to the Eternia backend and reads the forces back.
 *
 * Syntax:
 *   pair_style lj/cut/eternia <cutoff> [keyword value ...]
 *
 *   page      <KB>     page granularity, default 256
 *   blocks    <n>      CUDA blocks, default 64
 *   threads   <n>      threads per block, default 256
 *   slots     <n>      resident pages per block for x, default 16
 *   tag       <name>   CTE tag prefix, default "lmp_eternia"
 *   stats     <on|off> report paging counters each reneighbouring
 */
class PairLJCutEternia : public PairLJCut {
 public:
  PairLJCutEternia(class LAMMPS *);
  ~PairLJCutEternia() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void init_style() override;

 protected:
  eternia_lammps::Config cfg;
  eternia_lammps::Context *ctx;

  /** nall the context was sized for; a change forces a rebuild, because the
   *  vectors' logical length is fixed at construction. */
  int ctx_nall;

  /** Reupload positions and the neighbour list. Positions change every step,
   *  the list only on reneighbouring -- but the flattened list indexes atoms
   *  by their CURRENT ordering, so a sort invalidates both together. */
  void push_state();
  void ensure_context();
  void report_stats();
};

}    // namespace LAMMPS_NS

#endif
#endif
