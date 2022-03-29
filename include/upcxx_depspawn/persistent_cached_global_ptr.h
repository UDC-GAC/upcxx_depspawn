/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     persistent_cached_global_ptr.h
/// \brief    A thread-safe persistent cache for remote UPC++ objects
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#ifndef __UPCXX_PERSISTENT_CACHED_GLOBAL_PTR_H
#define __UPCXX_PERSISTENT_CACHED_GLOBAL_PTR_H

#include <type_traits>
#include <unordered_map>
#include <list>
#include "cached_global_ptr_basics.h"
#include "access_control.h"

namespace upcxx {

  /// Keeps copies of non-local references in a LRU cache.
  /// The current implementation has only been tested for read-only data
  class PersistentCache : public depspawn::internal::ProfiledCache {

    struct Node_t;

  public:

    using underlying_global_ptr_t = char;
    using Key_t       = upcxx::global_ptr<underlying_global_ptr_t>;
    using storage_t   = std::list<Node_t>;
    using map_t       = std::unordered_map<Key_t, typename storage_t::iterator>;
    using handle_t    = typename storage_t::iterator;

  private:
    
    /// Stores a local copy in the cache
    struct Node_t {
      static constexpr uint32_t EmptyFlag = 0x10000;
      static intrank_t here_;                     ///< Rank of this process

      underlying_global_ptr_t * const data_;
      std::atomic<uint32_t> uses_; ///< Number of tasks currently using this entry
      const size_t sz_;
      volatile bool written_;
      typename map_t::const_iterator map_it_; ///< Points to the map entry that points to this node in the storage

      Node_t(const size_t sz);
      
      Node_t(const Node_t& other) = delete;
      Node_t(Node_t&& other) = delete;
      Node_t& operator=(const Node_t&) = delete;
      Node_t& operator=(Node_t&&) = delete;
      
      bool notInUse() const noexcept { return !(uint32_t)uses_.load(); }

      bool isEmpty() const noexcept { return ((uint32_t)uses_.load()) & EmptyFlag; }
      
      void setReady() noexcept { uses_.fetch_and(~EmptyFlag); }
      
      void waitReady() const;
      
      /// Returns whether it should be invalidated for safeness
      bool release();

      ~Node_t();
      
      static inline intrank_t here()
      {
        if (here_ == -1) {
          here_ = upcxx::rank_me();
        }

        return here_;
      }

    };
    
    size_t max_size_;                 ///< Maximum number of elements allowed in the cache
    size_t slack_;                    ///< Excess elements beyond max_size_ allowed before a cache reduction is performed
    depspawn::internal::ReadWriteAccessControl_t access_control_;  ///< Controls parallel accesses to the cache
    map_t map_;                       ///< Allows to quickly find the cache nodes
    storage_t list_;                  ///< Stores the cache nodes
    size_t size_;                     ///< == list_.size(). Implemented due to problems with _GLIBCXX_USE_CXX11_ABI
    bool self_tune_;
    uint32_t last_hits_;
    size_t deepest_hit_;
    static constexpr uint32_t TunePeriod = 128; ///< Tune period must be a power of 2

    void reorderLRU(const handle_t& st_it);

    void periodicClean();

    void tune() noexcept;

  public:
    
    /// \internal here_ cannot be set to upcxx:rank_me() in case the object is static
    PersistentCache(size_t max_size = TunePeriod, size_t slack = 8);

    void set_cache_limits(size_t max_size, size_t slack = 8) noexcept;
    
    size_t max_size() const noexcept { return max_size_; }

    size_t slack() const noexcept { return slack_; }

    void set_self_tune(bool value) noexcept;

    /*
    /// Quick check of whether this is in the cache
    bool peek(const intrank_t rank, underlying_global_ptr_t * const cached_ptr)
    {
      const Key_t key(upcxx::detail::internal_only(), rank, cached_ptr);

      if (key.is_local()) {
        return true;
      }

      access_control_.reader_enter();
      const bool ret = (map_.find(key) != map_.end());
      access_control_.reader_exit();
      
      return ret;
    }
    */

    /// Makes a prefetching if the last parameter provides a future.
    /// The prefetchs must not try to use the returned handle_t or cached_ptr!
    handle_t get(intrank_t& rank, underlying_global_ptr_t *& cached_ptr, const size_t sz, upcxx::future<> * const fut = nullptr);
    
    static inline intrank_t here() { return Node_t::here(); }

    bool is_valid(const handle_t handle) const noexcept { return (handle != list_.end()); }
    
    size_t size() const noexcept { return size_; }

    bool empty() const noexcept { return !size_; }
    
    handle_t copy(const handle_t handle) const
    {
      if (is_valid(handle)) {
        handle->uses_.fetch_add(1);
      }
      return handle;
    }
    
    /// Decreases the use counter local copy of the remote key provided.
    void release(const handle_t handle)
    {
      if (is_valid(handle)) {
        const Key_t key_copy { handle->map_it_->first };
        if(handle->release()) {
          invalidate(key_copy);
        }
      }
    }

    void annotateWritten(const handle_t handle) const
    {
      if(is_valid(handle)) { //local handles do not need this mark
        //fprintf(stderr, "%d GH P=%d V=%d\n", upcxx::myrank(), handle->map_it_->first.where(), *(int *)handle->data_);
        handle->written_ = true;
      }
    }

    void invalidate(const Key_t& key);
    
    void clear() noexcept;

  };
  
  /// Global cache used by upcxx_depspawn
  extern PersistentCache GlobalRefCache;


  /// global_ptr that keeps copies of non-local data in a cache
  template <typename T, upcxx::memory_kind KindSet = upcxx::memory_kind::host>
  class cached_global_ptr : public global_ptr<T, KindSet> {

    using handle_t = typename PersistentCache::handle_t;
    const handle_t handle_;

    handle_t get_handle()
    {
      //Hack private_detail_do_not_use_ names for UPC++ 2021.3
      return GlobalRefCache.get(this->private_detail_do_not_use_rank_, reinterpret_cast<PersistentCache::underlying_global_ptr_t *&>(this->private_detail_do_not_use_raw_ptr_), sizeof(T));
    }

  public:

    /// Builds from non-const global_ptr
    cached_global_ptr(const global_ptr<typename std::remove_const<T>::type, KindSet>& gptr) :
    global_ptr<T, KindSet>(gptr),
    handle_(get_handle())
    {
      static_assert(std::is_const<T>::value, "only constant references should be cached");
    }

    /// Builds from const global_ptr
    cached_global_ptr(const global_ptr<T, KindSet>& gptr) :
    global_ptr<T, KindSet>(gptr),
    handle_(get_handle())
    {
      static_assert(std::is_const<T>::value, "only constant references should be cached");
    }

    // Copy constructor
    cached_global_ptr(const cached_global_ptr& other) :
    global_ptr<T, KindSet>(other),
    handle_(GlobalRefCache.copy(other.handle_))
    { }

    operator global_ptr<typename std::remove_const<T>::type, KindSet>() const
    {
      GlobalRefCache.annotateWritten(handle_);
      // Notice that the global_ptr is totally unrelated to the cache
      // The cache will be only notified that the writes have finished when this cached_global_ptr is destroyed
      // Thus in all the scenarios this cached_global_ptr must outlive the global_ptr returned
      return upcxx::const_pointer_cast<typename std::remove_const<T>::type>(*this);
    }

    static const char *cacheTypeName() noexcept { return GlobalRefCache.cacheTypeName_; }

    ~cached_global_ptr()
    {
      GlobalRefCache.release(handle_);
    }

  };
  
} // namespace upcxx
#endif
