#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)

# \file     CMakeLists.txt
# \brief    Builds the library
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

option( UPCXX_DEPSPAWN_ATOMIC_EXACT_MATCH  "Use atomics for exact match" OFF )

add_library( upcxx_depspawn STATIC upcxx_depspawn.cpp persistent_cache.cpp )

if(UPCXX_DEPSPAWN_ATOMIC_EXACT_MATCH)
	target_compile_definitions( upcxx_depspawn PRIVATE ATOMIC_EXACT_MATCH )
endif()

#install

install( TARGETS upcxx_depspawn EXPORT upcxx_depspawn-targets
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib
         INCLUDES DESTINATION include )

install( DIRECTORY ../include/upcxx_depspawn
         DESTINATION include
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )
