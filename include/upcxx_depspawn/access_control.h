/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     access_control.h
/// \brief    Controls for concurrent accesses
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#ifndef __ACCESS_CONTROL_H
#define __ACCESS_CONTROL_H

#include <atomic>
#include <cassert>

namespace depspawn {
  
namespace internal {
  
  /// Control based on lock/unlock on a std::atomic_flag
  struct LockFlagAccessControl_t {

    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;  ///< Controls access to the stored object
    
    LockFlagAccessControl_t() noexcept
    { }
    
    inline bool try_lock() noexcept {
      return !lock_.test_and_set(std::memory_order_acquire);
    }

    inline void lock() noexcept {
      while (!try_lock()) { }
    }

    inline void unlock() noexcept {
      lock_.clear(std::memory_order_release);
    }
    
  };

  /// Control based on a std::atomic<bool>
  class AtomicFlagAccessControl_t {
    
    std::atomic<bool> flag_;
    
  public:
    
    constexpr AtomicFlagAccessControl_t() noexcept :
    flag_{false}
    {}
    
    /// Tries to lock the object, but it does not wait if it fails.
    /// Returns true if locked, false otherwise
    inline bool try_lock() noexcept {
      bool tmp = flag_.load(std::memory_order_relaxed); 
      const bool ret = !tmp && !flag_.exchange(true);
      assert(!ret || is_locked());
      return ret;
    }
    
    inline void lock() noexcept {
      while (!try_lock()) { }
      assert(is_locked());
    }
    
    /// Unlocks the object
    inline void unlock() noexcept {
      assert(is_locked());
      flag_.store(false);
    }

    inline bool is_locked() const noexcept { return flag_.load(std::memory_order_relaxed); /**/ }
  
  };

  /// Object with same API as AtomicFlagAccessControl_t but without functionallity
  struct DummyAtomicFlagAccessControl_t {
    constexpr DummyAtomicFlagAccessControl_t() noexcept = default;
    inline void unlock() noexcept { }
    inline bool try_lock() noexcept { return true; }
    inline void lock() noexcept { }
    inline constexpr bool is_locked() const noexcept { return false; }
  };
  
  /// Control based on atomics that supports multiple reades/single writer
  struct ReadWriteAccessControl_t {
    
    std::atomic<int> readers_;
    std::atomic<bool> writer_;
    
    ReadWriteAccessControl_t() noexcept :
    readers_(0), writer_(false)
    { }
    
    ~ReadWriteAccessControl_t()
    {
      assert(readers_ == 0);
      assert(!writer_);
    }
    
    void reader_enter() noexcept {
      do {
        readers_++;
        
        if (writer_) {
          readers_--;
          while (writer_);
          // { upcxx::advance(); }   // or do nothing...
        } else {
          break;
        }
      } while (1);
    }
    
    void reader_exit() noexcept {
      readers_--;
    }
    
    void writer_enter() noexcept {
      while (writer_.exchange(true));
      // { upcxx::advance(); }      // or do nothing...
      while (readers_);
      // { upcxx::advance(); }      // or do nothing...
    }
    
    void writer_exit() noexcept {
      writer_ = false;
    }
    
  };

  /// Counts the number of types some value is repeated so an action can be taken
  template<typename VAL_TYPE>
  class ConsecutiveRepsCtrl {

    VAL_TYPE cur_val_;
    unsigned int n_consecutive_reps_;

  public:

    constexpr ConsecutiveRepsCtrl(VAL_TYPE init_val = 0) noexcept :
    cur_val_{init_val}, n_consecutive_reps_{0}
    {}
  
    //Notice that the first invocation will spuriously increase the counter if new_val == init_val provided
    unsigned int report(const VAL_TYPE new_val) noexcept
    {
      if(cur_val_ == new_val) {
        n_consecutive_reps_++;
      } else {
        n_consecutive_reps_ = 0;
      }
      cur_val_ = new_val;
      return n_consecutive_reps_;
    }
    
    constexpr unsigned int n_consecutive_reps() const noexcept { return n_consecutive_reps_; }
    
    bool trigger(const unsigned int limit, const bool do_reset) noexcept
    {
      const bool do_trigger = (n_consecutive_reps_ >= limit);
      if(do_reset && do_trigger) {
        reset();
      }
      return do_trigger;
    }

    void reset() noexcept
    {
      cur_val_ = 0;
      n_consecutive_reps_ = 0;
    }
    
  };

}; // namespace internal
  
}; // namespace depspawn

#endif
