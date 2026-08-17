.. index:: pair_style lj/cut/eternia

pair_style lj/cut/eternia command
=================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style lj/cut/eternia cutoff keyword value ...

* cutoff = global cutoff for Lennard-Jones interactions (distance units)
* zero or more keyword/value pairs may be appended
* keyword = *page* or *blocks* or *threads* or *slots* or *tag* or *stats*

  .. parsed-literal::

       *page* value = size of one page of the out-of-core arrays (kilobytes)
       *blocks* value = number of GPU thread blocks
       *threads* value = threads per block (must be a power of two)
       *slots* value = resident pages per block for positions (must be >= 3)
       *tag* value = name prefix for this run's data in the transfer engine
       *stats* value = *on* or *off* = report paging counters each step

Examples
""""""""

.. code-block:: LAMMPS

   newton off
   atom_modify sort 1000 2.0
   pair_style lj/cut/eternia 2.5
   pair_coeff * * 1.0 1.0

   pair_style lj/cut/eternia 2.5 page 256 blocks 64 threads 128 slots 16 stats on
   pair_coeff 1 1 1.0 1.0 2.5

Description
"""""""""""

.. versionadded:: TBD

Style *lj/cut/eternia* computes the same standard 12/6 Lennard-Jones
potential as :doc:`pair_style lj/cut <pair_lj_cut>`, but holds the per-atom
positions, types, forces and the neighbor list outside of GPU memory, in the
IOWarp Context Transfer Engine, and reads them into the GPU on demand from
inside the force kernel.

The purpose is to allow a simulation whose data is larger than the memory of
the GPU. A conventional GPU pair style must fit positions, types, forces and
the entire neighbor list into GPU memory at once. This style needs only its
page caches, whose size is set by the *blocks*, *slots* and *page* keywords,
so the size of the system is limited by the storage behind the transfer
engine rather than by the GPU.

The trade is speed for capacity. On a system that fits in GPU memory
comfortably this style is much slower than :doc:`pair_style lj/cut
<pair_lj_cut>` or its GPU variants, because every page of data makes a round
trip. It is intended for systems that cannot be run at all otherwise.

The coefficients are identical to :doc:`pair_style lj/cut <pair_lj_cut>`:

* :math:`\\epsilon` (energy units)
* :math:`\\sigma` (distance units)
* cutoff (distance units, optional, overrides the global cutoff)

Two settings are required rather than advisory:

**newton off** must be set, and it must appear before the simulation box is
defined, because LAMMPS does not allow the setting to change afterwards. This
style uses a full neighbor list so that every atom computes its own total
force. With a half list each pair's force would also have to be applied to
atom *j*, which generally belongs to a different page and a different thread
block's cache, and those caches are independent of one another. The style
stops with an error if newton is on for pairs, rather than computing half of
each force.

**atom_modify sort** is strongly recommended. A page holds a contiguous range
of atom indices, so paging is only efficient when atoms that are close in
space are also close in index, which is what sorting provides. Without it a
neighbor list touches nearly every page and the cache brings no benefit. An
unsorted system still gives correct results, only slowly.

The *stats* keyword prints the number of page faults, evictions and
writebacks each step, and makes a failed page read or writeback a fatal
error. It also checks that the kernel examined exactly as many neighbor list
entries as the list contains, which is stopped as an error if it does not.

Restrictions
""""""""""""

This style is part of the ETERNIA package. It is only enabled if LAMMPS was
built with that package, which additionally requires a CUDA device, a
coroutine-capable clang, and an installation of IOWarp Core. See the
:doc:`Build package <Build_package>` page for more info, and
``lib/eternia/README.md`` for the build recipe.

This style runs on a single MPI rank. Each rank would need its own name
prefix and its own GPU, which is not implemented.

The per-atom energy and virial (:doc:`compute pe/atom <compute_pe_atom>` and
:doc:`compute stress/atom <compute_stress_atom>`) are not available; the
global energy and virial are computed and are correct. The *single* method is
not implemented, so styles and computes that call it, such as :doc:`compute
group/group <compute_group_group>`, cannot be used with this pair style.

Related commands
""""""""""""""""

:doc:`pair_style lj/cut <pair_lj_cut>`, :doc:`pair_coeff <pair_coeff>`,
:doc:`newton <newton>`, :doc:`atom_modify <atom_modify>`

Default
"""""""

page = 256, blocks = 64, threads = 256, slots = 16, tag = lmp_eternia,
stats = off
