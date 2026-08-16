# ---------------------------------------------------------------------------
# lib/eternia -- the device half of the ETERNIA package.
#
# A STANDALONE project on purpose. The Eternia paging kernels are C++20
# device coroutines, which only clang++-22 driving -x cuda can compile, while
# LAMMPS itself is built with the user's compiler (and, with KOKKOS, with
# nvcc, which rejects co_await in device code outright). One CMake project
# cannot use two C++ compilers, so cmake/Modules/Packages/ETERNIA.cmake
# configures this one via ExternalProject with its own toolchain, and links
# the resulting static library into LAMMPS.
#
# Configure directly (outside LAMMPS) like this:
#
#   cmake -S lib/eternia -B build-eternia \
#     -DCMAKE_CUDA_COMPILER=clang++-22 \
#     -DCMAKE_CUDA_ARCHITECTURES=89 \
#     -Diowarp-core_DIR=<clio-install>/lib/cmake/iowarp-core
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.30)
project(lammps_eternia LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ETERNIA_ENABLE "Build the real Eternia backend (needs CUDA + clang)" ON)

if(ETERNIA_ENABLE)
  enable_language(CUDA)

  if(NOT CMAKE_CUDA_COMPILER MATCHES "clang")
    message(FATAL_ERROR
      "lib/eternia requires clang as the CUDA compiler "
      "(-DCMAKE_CUDA_COMPILER=clang++-22): nvcc cannot compile the device "
      "coroutines the paging kernel is built from.")
  endif()

  # These go BEFORE iowarp-core, and none of them is used here directly.
  # iowarp-core's exported targets name them in their link interfaces
  # (`nvcomp::nvcomp` on ctp::cuda, `yaml-cpp::yaml-cpp` on ctp::configure,
  # ...) but its package config does not find_package them itself, so
  # consuming iowarp-core out of tree fails at GENERATE time with "target not
  # found" unless the imported targets already exist. All QUIET: a Clio built
  # without a given dependency exports no reference to it.
  #
  # This is a workaround for an iowarp-core export bug, not a real dependency
  # of this library; it can be deleted once the config finds its own
  # transitive dependencies.
  find_package(nvcomp QUIET CONFIG)
  find_package(yaml-cpp QUIET CONFIG)
  find_package(Threads QUIET)

  find_package(iowarp-core REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  # Mirrors iowarp-core's own clang-CUDA settings. -fno-strict-aliasing is not
  # optional: device TUs in this tree type-pun through differently-typed
  # fills, and clang is the first compiler here to exploit the UB.
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -fno-strict-aliasing")
  if(EXISTS "${CUDAToolkit_TARGET_DIR}/include/cccl")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -isystem ${CUDAToolkit_TARGET_DIR}/include/cccl")
  endif()

  # .cc, but compiled as CUDA: the file holds __global__ kernels and
  # co_await, and naming it .cu would invite nvcc from any build system that
  # guesses by extension.
  set_source_files_properties(eternia_lammps.cc PROPERTIES LANGUAGE CUDA)

  add_library(lammps_eternia STATIC eternia_lammps.cc)
  target_compile_definitions(lammps_eternia PRIVATE
      ETERNIA_LAMMPS_ENABLED CLIO_YIELD_CORO)
  target_include_directories(lammps_eternia PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${IOWARP_CORE_INCLUDE_DIR})
  # ctp::cudacxx and clio::run::clio_run_cxx_gpu are NOT optional extras: they
  # are what carry CTP_ENABLE_CUDA=1 into the compile. The host-only targets
  # deliberately do not propagate it (so a consumer can link both halves), and
  # without it every Clio header compiles its HOST branch -- which fails on
  # the first __threadfence/__shfl_sync/atomicCAS in a device function, in
  # about two hundred errors that all look like a broken toolchain.
  #
  # The names are the INSTALLED ones, which differ from the in-tree names the
  # benchmarks use (ctp::cuda_cxx -> ctp::cudacxx,
  # clio_run_cxx_gpu -> clio::run::clio_run_cxx_gpu).
  target_link_libraries(lammps_eternia PUBLIC
      clio::run::cxx
      clio::run::clio_run_cxx_gpu
      clio::run::admin_client
      clio::run::bdev_client
      clio::cte::core_client
      ctp::cudacxx
      CUDA::cudart)
  # SEPARABLE COMPILATION OFF, deliberately, and iowarp-core makes the same
  # choice for the same reason (see CLIO_CUDA_SEPARABLE in
  # cmake/ClioCoreCommon.cmake).
  #
  # Separable compilation leaves RELOCATABLE device code in the object file,
  # which needs a device-link step before the kernel is registered with the
  # CUDA runtime -- and LAMMPS links this library with an ordinary C++
  # compiler, which does no such step. The result is a binary that builds and
  # runs but whose every launch fails with "invalid resource handle": no
  # kernel executes, no page is faulted, and every force comes back zero.
  # Letting CMake do the device link instead is not an option either, because
  # under clang-CUDA it drives fatbinary with pre-CUDA-13 options
  # ("fatbinary fatal: Unknown option 'im'").
  #
  # Off is correct rather than merely expedient: every device callee here
  # lives in a header (gpu_vector and the yield machinery are header-only), so
  # this translation unit's fatbin is already self-contained.
  set_target_properties(lammps_eternia PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF)
else()
  # Stub: the same API, every entry point reporting why it is unavailable, so
  # `pair_style lj/cut/eternia` fails with a sentence instead of a link error.
  add_library(lammps_eternia STATIC eternia_lammps.cc)
  target_include_directories(lammps_eternia PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

install(TARGETS lammps_eternia
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(FILES eternia_lammps.h DESTINATION include)
