/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
 Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     persistent_cache.cpp
/// \brief    implements the persistent cache
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>
///

#include "upcxx_depspawn/persistent_cached_global_ptr.h"

namespace depspawn {
  namespace internal {
  
  extern bool UPCXX_DEPSPAWN_YIELD;
  extern bool UPCXX_DEPSPAWN_PREFETCH;
  extern void idle_progress(const bool);
  
  ProfiledCache::ProfiledCache_Storage_t *ProfiledCache::Profiled_caches = nullptr;

  ProfiledCache::ProfiledCache(const char * const cacheTypeName) :
  cacheTypeName_(cacheTypeName)
  {
    clear_profiling();
    
    if(Profiled_caches == nullptr) {
      Profiled_caches = new ProfiledCache_Storage_t();
    }

    Profiled_caches->push_back(this);
  }
  
  void ProfiledCache::clear_profiling() noexcept {
    Hits = Accs = 0;
    DEPSPAWN_PROFILEACTION(Prefetches = UsefulPrefetches = 0);
    DEPSPAWN_PROFILEACTION(Size = Cleans = RemoteInvalidations = Reorders = 0);
  }
  
  int ProfiledCache::dump_profiling(char *buff, upcxx::intrank_t rank_number) const {
    int tmp = 0;
    DEPSPAWN_PROFILEACTION(tmp = sprintf(buff, "Cache P%d %zu/%zu %.1f%% Pf=%zu/%zu  %.1f%% Sz=%zu CL=%zu RO=%zu RI=%zu\n", rank_number, (size_t)Hits, (size_t)Accs, ((float)Hits/(float)Accs) * 100.f, (size_t)UsefulPrefetches, (size_t)Prefetches,
        ((float)UsefulPrefetches/(float)Prefetches) * 100.f, Size, Cleans, Reorders, RemoteInvalidations));
    return tmp;
  }
  
  } // namespace internal
}  // namespace depspawn

namespace upcxx {

PersistentCache::Node_t::Node_t(const size_t sz) :
//data_(static_cast<underlying_global_ptr_t *>(malloc(sz))),
data_(upcxx::new_array<underlying_global_ptr_t>(sz).raw_internal(upcxx::detail::internal_only())), //so it works with rget outside cache
uses_(EmptyFlag|1),
sz_(sz),
written_(false)
{
  assert(data_ != nullptr);
}

void PersistentCache::Node_t::waitReady() const
{
  while (isEmpty()) {
    if(depspawn::internal::UPCXX_DEPSPAWN_YIELD || depspawn::internal::UPCXX_DEPSPAWN_PREFETCH) {
      depspawn::internal::idle_progress(true);
    }
  }
}

PersistentCache::Node_t::~Node_t()
{
  assert(notInUse());
  assert(!written_);
  //free(data_);
  assert(here_ != -1); // It should have been required before the construction
  upcxx::global_ptr<underlying_global_ptr_t> p(upcxx::detail::internal_only(), here_, data_);
  upcxx::delete_array<underlying_global_ptr_t>(p);
}

bool PersistentCache::Node_t::release()
{
  const bool dump = !(uses_.fetch_sub(1) - 1) && written_;
  if(dump) {
    uses_.fetch_or(EmptyFlag); //Keep alive the node during the write-back
    written_ =  false;
    upcxx::rput(data_, map_it_->first, sz_).wait();
    setReady();
  }
  return dump;
}

void PersistentCache::reorderLRU(const handle_t& st_it)
{
  DEPSPAWN_PROFILEACTION(Reorders++);

  const auto begin_it = list_.begin();
  if (st_it != begin_it) {
    if (deepest_hit_) {
      const auto depth = std::distance(begin_it, st_it);
      deepest_hit_ = std::max<size_t>(deepest_hit_, depth);
    }
    list_.splice(begin_it, list_, st_it);
  }
}

void PersistentCache::periodicClean()
{ size_t removable_num = 0;

  typename storage_t::const_iterator * const removable = static_cast<typename storage_t::const_iterator *>(alloca((slack_ + 1) * sizeof(typename storage_t::const_iterator)));
  
  DEPSPAWN_PROFILEACTION(Cleans++);
  
  assert(size_ > (max_size_ + slack_));
  
  const auto list_end = list_.crend();
  
  for(auto rit = list_.crbegin(); rit != list_end; ) {
    const bool must_erase = rit->notInUse();
    ++rit;
    if(must_erase) {
      removable[removable_num++] = rit.base();
      if ( removable_num > slack_ ) {
        break;
      }
    }
  }
  
  for(size_t i = 0; i < removable_num; i++) {
    assert(removable[i]->notInUse());
    map_.erase(removable[i]->map_it_);
    list_.erase(removable[i]);
  }
  size_ -= removable_num;
  
}

void PersistentCache::tune() noexcept
{ static uint32_t last_period_hits = 0;
  static size_t prev_deepest_hit; // only meaningful when deepest_hit_ != 0
  static int periods_counter = 0;
  static int  periods_limit; // only meaningful when deepest_hit_ != 0

  const auto hits = Hits.load(std::memory_order_relaxed);
  if (hits < last_hits_) { // in case of clear_profiling()
    last_hits_ = 0;
    last_period_hits = 0;
    periods_counter = 0;
    deepest_hit_ = 0;
  }
  const auto new_period_hits = hits - last_hits_;
  const float hit_rate = new_period_hits / (float)TunePeriod;

  if (deepest_hit_) {
    if(++periods_counter == periods_limit) {
      if (deepest_hit_ > prev_deepest_hit) {
        prev_deepest_hit = deepest_hit_; // reevaluate
        periods_limit = std::max<int>(1, (size_ - deepest_hit_) / (3 * TunePeriod));
      } else {
        max_size_ = (deepest_hit_ + size_) / 2; // std::min((deepest_hit_ + max_size_) / 2, size_);
        deepest_hit_ = 0;
      }
      periods_counter = 0;
    }
  } else {

    if (hit_rate > 0.98f) {
      max_size_ -= max_size_ / 33; // -3% size
      periods_counter = 0;
      periods_limit = std::max<int>(1, size_ / (3 * TunePeriod));
      deepest_hit_ = prev_deepest_hit = 1; // collect depth data
    } else {
      
      if (hit_rate < 0.96f) {
        size_t proposed_size;

        const size_t min_elems = (500UL << 20) / list_.cbegin()->sz_; // 500MB
        const size_t max_elems = (4UL << 30) / list_.cbegin()->sz_; // 4GB

        if(max_size_ < min_elems) {
          proposed_size = max_size_ + TunePeriod;
        } else {

          const float rate = (TunePeriod - new_period_hits) / (float)TunePeriod; // + (100-hit_rate)% size
          proposed_size = max_size_ + max_size_ * rate;

          if(new_period_hits < (last_period_hits + 5)) {
            if (++periods_counter == 4) {
              periods_counter = 0;
              periods_limit = std::max<int>(1, size_ / (3 * TunePeriod));
              deepest_hit_ = prev_deepest_hit = 1;
            }
          } else {
            periods_counter = 0;
          }
        }
        max_size_ = std::min<size_t>(proposed_size, max_elems);
      }
    }
  }

  last_period_hits = new_period_hits;
  last_hits_ = hits;
}

PersistentCache::PersistentCache(size_t max_size, size_t slack) :
depspawn::internal::ProfiledCache("PersistentCache"),
max_size_(max_size), slack_(slack), size_(0),
self_tune_(false), last_hits_(0), deepest_hit_(0)
{}

void PersistentCache::set_cache_limits(size_t max_size, size_t slack) noexcept
{
  max_size_ = max_size;
  slack_ = slack;

  if (self_tune_ && (max_size_ < TunePeriod)) {
    max_size_ = TunePeriod;
  }
}

void PersistentCache::set_self_tune(bool value) noexcept {
  self_tune_ = value;
  if (self_tune_ && (max_size_ < TunePeriod)) {
    max_size_ = TunePeriod;
  }
}

PersistentCache::handle_t PersistentCache::get(intrank_t& rank, underlying_global_ptr_t *& cached_ptr, const size_t sz, upcxx::future<> * const fut)
{ handle_t st_it;
  Node_t *node;

  const Key_t key(upcxx::detail::internal_only(), rank, cached_ptr);

  if (key.is_local()) { //Use (rank == here()) for debugging
    // cached_ptr is already valid
    return list_.end();
  }

  rank = here();

  const bool is_prefetch = (fut != nullptr);

  if (is_prefetch) {
    DEPSPAWN_PROFILEACTION(Prefetches.fetch_add(1, std::memory_order_relaxed));
  } else {
    const auto tmp = Accs.fetch_add(1, std::memory_order_relaxed);
    if (self_tune_ && !((tmp + 1) & (TunePeriod - 1))) {
      tune();
    }
  }

  access_control_.reader_enter();

  auto map_it = map_.find(key);

  if (map_it != map_.end()) {
    st_it = map_it->second;
    node = &(*st_it);
    node->uses_.fetch_add(1);
    access_control_.reader_exit();
    if (!is_prefetch) {
      Hits.fetch_add(1, std::memory_order_relaxed);
      node->waitReady();
    }
    // Notice that this test is out of control by access_control_
    if (st_it != list_.begin()) {    //Keep LRU, also for prefetched data
      access_control_.writer_enter();
      reorderLRU(st_it);
      access_control_.writer_exit();
    }
    if (is_prefetch) {
      node->uses_.fetch_sub(1); //Undo previous fetch_add(1)
      return st_it; // avoid possible problems in accesses to *st_it and node below
    }
  } else {
    access_control_.reader_exit();
    access_control_.writer_enter();
    const auto insert_pair = map_.emplace(std::piecewise_construct, std::make_tuple(key), std::make_tuple());
    if (!insert_pair.second) { // element already present in map
      st_it = insert_pair.first->second;
      st_it->uses_.fetch_add(1); // Ensures the element does not dissapear right when we leave the critical section
      if (!is_prefetch) {
        reorderLRU(st_it);
      }
    } else {
      //list_.emplace_front(std::piecewise_construct, std::make_tuple(sz));
      list_.emplace_front(sz);
      size_++;
      st_it = list_.begin();
      st_it->map_it_ = insert_pair.first;
      insert_pair.first->second = st_it;
      if (size_ > (max_size_ + slack_)) {
        periodicClean();
      }
    }
    access_control_.writer_exit();
    
    node = &(*st_it);
    if (insert_pair.second) {
      upcxx::future<> f = upcxx::rget(key, (underlying_global_ptr_t *)node->data_, sz);
      if (is_prefetch) {
        DEPSPAWN_PROFILEACTION(UsefulPrefetches.fetch_add(1, std::memory_order_relaxed));
        *fut = f.then([this, st_it, node] {
          if (st_it != list_.begin()) {    //Bring to top of LRU on arrival
            access_control_.writer_enter();
            reorderLRU(st_it);
            access_control_.writer_exit();
          }
          node->setReady();
          node->uses_.fetch_sub(1);
        });
      } else {
        f.wait();
        node->setReady();
      }
    } else {
      if (is_prefetch) {
        node->uses_.fetch_sub(1); //Undo previous fetch_add(1)
        return st_it; // avoid possible problems in accesses to node below
      } else {
        Hits.fetch_add(1, std::memory_order_relaxed);
        node->waitReady();
      }
    }
  }

  assert( !node->isEmpty() || is_prefetch );

  DEPSPAWN_PROFILEACTION(Size = size_);

  cached_ptr = node->data_;

  return st_it;
}

void PersistentCache::invalidate(const Key_t& key)
{
  access_control_.reader_enter();
  auto map_it = map_.find(key);
  const bool is_erasable = (map_it != map_.end()) && map_it->second->notInUse();
  access_control_.reader_exit();
  
  if (is_erasable) {
    access_control_.writer_enter();
    map_it = map_.find(key);  // Could have been erased in the meantime
    if ( (map_it != map_.end()) && map_it->second->notInUse() ) {
      DEPSPAWN_PROFILEACTION(RemoteInvalidations++);
      list_.erase(map_it->second);
      map_.erase(map_it);
      size_--;
    }
    access_control_.writer_exit();
  }
}

void PersistentCache::clear() noexcept
{
  access_control_.writer_enter();

  if (depspawn::internal::UPCXX_DEPSPAWN_PREFETCH) { //for pending prefetchs
    for (Node_t& node: list_) {
      while (node.isEmpty()) {
        depspawn::internal::idle_progress(false);
      }
    }
  }

  list_.clear();
  size_ = 0;
  map_.clear();
  clear_profiling();
  access_control_.writer_exit();
}

} // namespace upcxx
