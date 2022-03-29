/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     cached_global_ptr_basics.h
/// \brief    Common utils for caches for remote UPC++ objects
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#ifndef __CACHED_GLOBAL_PTR_BASICS_H
#define __CACHED_GLOBAL_PTR_BASICS_H

/*
// Ensure g++ uses new ABI https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
// Otherwise size(), empty(), etc. are slow and unsafe, as they cross the list to count the elements!
#define _GLIBCXX_USE_CXX11_ABI 1
*/

#include <upcxx/upcxx.hpp>
// Only depends on DepSpawn for profiling
#include "depspawn/depspawn_utils.h"

namespace depspawn {
  
  namespace internal {
    
    /// Keeps profiling information
    struct ProfiledCache {
      
      /// Type of container for storing
      using ProfiledCache_Storage_t = std::vector<ProfiledCache*>;
      
      /// Should store a pointer to every profiled cache
      static ProfiledCache_Storage_t *Profiled_caches;
      
      /// Allows to discover the kind of cache used at runtime
      const char * const cacheTypeName_;
      
      std::atomic<uint32_t> Hits;
      std::atomic<uint32_t> Accs;
      DEPSPAWN_PROFILEDEFINITION(std::atomic<uint32_t> Prefetches);
      DEPSPAWN_PROFILEDEFINITION(std::atomic<uint32_t> UsefulPrefetches);
      DEPSPAWN_PROFILEDEFINITION(volatile size_t Size);
      DEPSPAWN_PROFILEDEFINITION(volatile size_t Cleans);
      DEPSPAWN_PROFILEDEFINITION(volatile size_t RemoteInvalidations);
      DEPSPAWN_PROFILEDEFINITION(volatile size_t Reorders);

      ProfiledCache(const char * const cacheTypeName);
      
      void clear_profiling() noexcept;
      
      int dump_profiling(char *buff, upcxx::intrank_t rank_number = 0) const;
      
    };

  }; // namespace internal
  
}; // namespace depspawn

#endif
