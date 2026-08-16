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
  target_link_libraries(lammps_eternia PUBLIC
      clio::run::cxx
      clio::run::admin_client
      clio::run::bdev_client
      clio::cte::core_client
      CUDA::cudart)
  set_target_properties(lammps_eternia PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  # Stub: the same API, every entry point reporting why it is unavailable, so
  # `pair_style lj/cut/eternia` fails with a sentence instead of a link error.
  add_library(lammps_eternia STATIC eternia_lammps.cc)
  target_include_directories(lammps_eternia PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

install(TARGETS lammps_eternia ARCHIVE DESTINATION lib)
install(FILES eternia_lammps.h DESTINATION include)
