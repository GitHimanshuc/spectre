# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Arpack REQUIRED)

add_library(Arpack INTERFACE IMPORTED)

message(STATUS "Arpack libs: " ${ARPACK_LIBRARIES})
message(STATUS "Arpack incl: " ${ARPACK_INCLUDE_DIR})

set_property(TARGET Arpack
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${ARPACK_LIBRARIES})
set_property(TARGET Arpack
  APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARPACK_INCLUDE_DIR})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Arpack
  )
