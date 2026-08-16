# ---------------------------------------------------------------------------
# ETERNIA package: out-of-core pair styles backed by the Clio CTE gpu_vector.
#
# The device half (lib/eternia) is built as an EXTERNAL PROJECT rather than as
# an ordinary target, because it needs clang++-22 driving -x cuda for its C++20
# device coroutines while LAMMPS is compiled with the user's own compiler. A
# single CMake project cannot host two C++ compilers, so the two are separated
# by a static library and a plain-C++ header.
#
# Required:
#   -Diowarp-core_DIR=<clio-install>/lib/cmake/iowarp-core
# Optional:
#   -DETERNIA_CUDA_COMPILER=clang++-22   (default)
#   -DETERNIA_CUDA_ARCHITECTURES=89      (default: CMAKE_CUDA_ARCHITECTURES)
# ---------------------------------------------------------------------------

set(ETERNIA_SOURCE_DIR ${LAMMPS_LIB_SOURCE_DIR}/eternia)

set(ETERNIA_CUDA_COMPILER "clang++-22" CACHE STRING
    "clang used to compile the Eternia device coroutines")
set(ETERNIA_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architectures for the Eternia kernels")
if(NOT ETERNIA_CUDA_ARCHITECTURES)
  set(ETERNIA_CUDA_ARCHITECTURES 89)
endif()

if(NOT iowarp-core_DIR)
  message(FATAL_ERROR
    "PKG_ETERNIA needs the Clio install: pass "
    "-Diowarp-core_DIR=<clio-install>/lib/cmake/iowarp-core")
endif()

include(ExternalProject)

set(ETERNIA_BINARY_DIR ${CMAKE_BINARY_DIR}/eternia-build)
set(ETERNIA_INSTALL_DIR ${CMAKE_BINARY_DIR}/eternia-install)

ExternalProject_Add(eternia_build
  SOURCE_DIR ${ETERNIA_SOURCE_DIR}
  BINARY_DIR ${ETERNIA_BINARY_DIR}
  INSTALL_DIR ${ETERNIA_INSTALL_DIR}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${ETERNIA_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CUDA_COMPILER=${ETERNIA_CUDA_COMPILER}
    -DCMAKE_CUDA_ARCHITECTURES=${ETERNIA_CUDA_ARCHITECTURES}
    -Diowarp-core_DIR=${iowarp-core_DIR}
    -DETERNIA_ENABLE=ON
  BUILD_BYPRODUCTS ${ETERNIA_INSTALL_DIR}/lib/liblammps_eternia.a
)

add_library(LAMMPS::eternia STATIC IMPORTED)
set_target_properties(LAMMPS::eternia PROPERTIES
  IMPORTED_LOCATION ${ETERNIA_INSTALL_DIR}/lib/liblammps_eternia.a
  INTERFACE_INCLUDE_DIRECTORIES ${ETERNIA_SOURCE_DIR})
add_dependencies(LAMMPS::eternia eternia_build)

# The backend pulls in the Clio client libraries and the CUDA runtime, and
# those are resolved in the LAMMPS link line, not inside the static archive.
find_package(iowarp-core REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(lammps PRIVATE
  LAMMPS::eternia
  clio::run::cxx
  clio::run::admin_client
  clio::run::bdev_client
  clio::cte::core_client
  CUDA::cudart)

# src/ETERNIA is picked up by the standard package machinery; this only adds
# the boundary header's directory so pair_lj_cut_eternia.h can find it.
target_include_directories(lammps PRIVATE ${ETERNIA_SOURCE_DIR})
