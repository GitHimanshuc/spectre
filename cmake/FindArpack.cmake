# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT ARPACK_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(ARPACK_ROOT "")
  set(ARPACK_ROOT $ENV{ARPACK_ROOT})
endif()

find_path(ARPACK_INCLUDE_DIR arpack/arpackdef.h
  PATH_SUFFIXES include
  HINTS ${ARPACK_ROOT}/include/)

find_library(ARPACK_LIBRARIES
  NAMES arpack
  PATH_SUFFIXES lib64 lib
  HINTS ${ARPACK_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Arpack
  FOUND_VAR ARPACK_FOUND
  REQUIRED_VARS ARPACK_LIBRARIES ARPACK_INCLUDE_DIR
  )
