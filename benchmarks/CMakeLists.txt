#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)

# \file     CMakeLists.txt
# \brief    Builds the benchmarks provided
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
link_libraries( upcxx_depspawn depspawn )

if(UNIX)
  if(APPLE)
    set( CMAKE_MACOSX_RPATH ON )
  else()
    link_libraries( rt )
  endif()
endif()

option( STACK_TILE "Allocate benchmark tiles in the stack" OFF )

if(STACK_TILE)
  add_definitions(-DSTACK_TILE)
endif(STACK_TILE)

set( TILESIZE "200" CACHE STRING "Tile size for benchmarks" )

if(TILESIZE)
  add_definitions(-DTILESIZE=${TILESIZE})
endif(TILESIZE)

#SharedArray.h and DistrBlockMatrix.h
include_directories( ${PROJECT_SOURCE_DIR}/tests )

add_executable( T0 T0.cpp )
target_compile_definitions( T0 PUBLIC UPCXX_DEPSPAWN_NO_CACHE )

add_executable( gauss-seidel gauss-seidel.cpp )
target_compile_definitions( gauss-seidel PUBLIC UPCXX_DEPSPAWN_NO_CACHE )

add_custom_target(benchmarks)
add_dependencies(benchmarks T0 gauss-seidel)

#set(BLA_STATIC ON)
#set(BLA_VENDOR OpenBLAS)
find_package( BLAS )
#message( "BLAS_LINKER_FLAGS  : " ${BLAS_LINKER_FLAGS})
#message( "BLAS_LIBRARIES  : " ${BLAS_LIBRARIES})

if(BLAS_FOUND)
  if(APPLE)
	  include_directories("/Applications/Xcode.app//Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers")
  endif()
  add_subdirectory( cholesky )
  add_subdirectory( lu_factorization )
endif(BLAS_FOUND)
