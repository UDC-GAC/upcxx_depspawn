#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)

# \file     CMakeLists.txt
# \brief    Main CMakeLists.txt of the project
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

cmake_minimum_required( VERSION 2.8.7 )

project( UPCxx_DepSpawn )

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

if (NOT UPCXX_NETWORK)

  execute_process(COMMAND upcxx-meta GASNET_CONDUIT 
                  OUTPUT_VARIABLE tmp 
			      OUTPUT_STRIP_TRAILING_WHITESPACE )

  set(UPCXX_NETWORK ${tmp} CACHE STRING
      "Options are: ibv aries smp udp mpi"
      FORCE)

  unset( tmp )

endif()

find_package( depspawn REQUIRED )

option( UPCXX_DEPSPAWN_AUTOMATIC_CACHE     "Use caching automatically"          OFF )

if(UPCXX_DEPSPAWN_AUTOMATIC_CACHE)
  add_definitions(-DUPCXX_DEPSPAWN_AUTOMATIC_CACHE)
endif()
 
if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
	add_compile_options( -codemode=debug )
	SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -codemode=debug")
else()
    add_compile_options( -codemode=opt )
	SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -codemode=opt")
	#RelWithDebInfo adds -O2 -g -DNDEBUG
	#Release adds -O3 -DNDEBUG
endif()

message( WARNING "*** Compiling for network ${UPCXX_NETWORK} ***" )
add_compile_options( -network=${UPCXX_NETWORK} )
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -network=${UPCXX_NETWORK}" )

set( CMAKE_CXX_COMPILER upcxx ) # at $ENV{HOME}/new_upcxx/bin/
add_compile_options( -std=c++11 -threadmode=par  )
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -threadmode=par")

include_directories( include )

add_subdirectory( src )
add_subdirectory( tests )
add_subdirectory( benchmarks )

