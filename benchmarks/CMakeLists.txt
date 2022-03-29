#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)

# \file     CMakeLists.txt
# \brief    Builds the benchmarks provided
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

cmake_minimum_required( VERSION 2.8.7 )

link_libraries( upcxx_depspawn depspawn )

if(${UNIX})
  if(${APPLE})
    set( CMAKE_MACOSX_RPATH ON )
	#Needed in my High Sierra
	link_directories( /usr/local/lib )
  else(${APPLE})
    link_libraries( rt )
  endif(${APPLE})
endif(${UNIX})

if( EXISTS /opt/OpenBLAS )
	include_directories( /opt/OpenBLAS/include )
	link_directories( /opt/OpenBLAS/lib )
endif()

link_libraries( openblas )

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

add_subdirectory( cholesky )
add_subdirectory( lu_factorization )

