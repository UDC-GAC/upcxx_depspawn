#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)

# \file     CMakeLists.txt
# \brief    Builds the tests
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

cmake_minimum_required( VERSION 2.8.7 )

add_definitions( -DBZ_THREADSAFE )

link_libraries( upcxx_depspawn depspawn )

if(${UNIX})
  if(${APPLE})
    set( CMAKE_MACOSX_RPATH ON )
  else(${APPLE})
    link_libraries( rt )
  endif(${APPLE})
endif(${UNIX})

set( TESTBINARIES test1 test2 test3 testDistrBlockMatrix testCacheBasic testCache test_args test_api test_central bench_upcxx_depspawn )

#set( TESTBINARIES test1 test2 test2wait test3wait testDistrBlockMatrix testHistogram testSharedVar testDistrMatrix testDistrTriangMatrix test_central )

#set( TEST_CACHE_TESTBINARIES test_api test_args testCacheBasic testCache bench_upcxx_depspawn )

foreach(b ${TESTBINARIES} )
  add_executable( ${b} ${b}.cpp )
  #Cmake 3.13 is required for add_link_options and target_link_options
endforeach(b)

#if(UPCXX_DEPSPAWN_AUTOMATIC_CACHE) #build no autom
#  add_executable( test_args_noauto_persistentCache test_args.cpp )
#  target_compile_definitions( test_args_noauto_persistentCache PUBLIC -UUPCXX_DEPSPAWN_AUTOMATIC_CACHE )
#else(UPCXX_DEPSPAWN_AUTOMATIC_CACHE) #build autom
#  add_executable( test_args_auto_persistentCache test_args.cpp )
#  target_compile_definitions( test_args_auto_persistentCache PUBLIC UPCXX_DEPSPAWN_AUTOMATIC_CACHE )
#endif(UPCXX_DEPSPAWN_AUTOMATIC_CACHE)

# Tests that are built but do not belong to the standard check tests
add_executable( bench_sched_perf bench_sched_perf.cpp )
add_executable( bench_ping_pong  bench_ping_pong.cpp )

# run check tests
add_custom_target(check
                  DEPENDS ${TESTBINARIES}
                  COMMENT "Run tests")


foreach(b ${TESTBINARIES} )

  add_custom_command(TARGET check PRE_BUILD COMMAND ${CMAKE_COMMAND} -E echo test ${b} :)
  add_custom_command(TARGET check PRE_BUILD COMMAND upcxx-run -n 4 ${b} || exit 1)
  add_custom_command(TARGET check PRE_BUILD COMMAND ${CMAKE_COMMAND} -E echo ------------------------)

endforeach(b)
add_custom_command(TARGET check PRE_BUILD COMMAND echo "*" TESTS DONE. SUCCESSFUL)

