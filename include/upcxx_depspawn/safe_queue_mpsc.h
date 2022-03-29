/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     safe_queue_mpsc.h
/// \brief    Provides a thread-safe queue for multiple producers and a single consumer
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#ifndef __SAFE_QUEUE_MPSC_H
#define __SAFE_QUEUE_MPSC_H

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <memory>

namespace depspawn {
  
namespace internal {
  
  /// Thread-safe queue of constant size that supports multiple producers and a single consumer
  /** It can store up to (LENGTH-1) elements, therefore LENGTH must be greater than 1.
      Notice that the implementation is designed for a class that supports copy constructors
      instead of operator= to copy values of it, which is the case of the upcxx::global_ptr.
    */
  template<typename T, uint32_t LENGTH>
  class SafeQueue_MPSC {
    
    typename std::aligned_storage<sizeof(T), alignof(T)>::type data[LENGTH];     ///< queue storage. volatile not allowed by T::operator= sometimes
    std::atomic<uint32_t> push_point; ///< points to current push location
    std::atomic<uint32_t> cur_size;   ///< number of elemens in the queue
    volatile uint32_t pop_point;      ///< points to current pop location
    
    static_assert(LENGTH > 1, "The implementation requires storage for at least two items");
    
    /// Get next circular element. General case
    template<class INDEX_TYPE>
    typename std::enable_if<(LENGTH & (LENGTH-1)), INDEX_TYPE>::type next_point(INDEX_TYPE i) const noexcept
    {
      return (i == (LENGTH - 1)) ? 0 : (i + 1);
    }
    
    /// Get next circular element. Optimization for powers of two
    template<class INDEX_TYPE>
    typename std::enable_if<!(LENGTH & (LENGTH-1)), INDEX_TYPE>::type next_point(INDEX_TYPE i) const noexcept
    {
      return (i + 1) & (LENGTH-1);
    }
    
    /// Items in queue between the pop point and the provided one
    /** @internal pop_point is read in a non-safe manner but
     this should be ok in the loopin which this function is used.
     If LENGT power of 2, it could ((LENGTH|i) - pop_point) & (LENGTH - 1) */
    constexpr uint32_t data_before_point(uint32_t i) const noexcept
    {
      return (i >= pop_point) ? (i - pop_point) : (LENGTH - pop_point + i);
    }
    
    /// Safely acquire an insertion point
    uint32_t get_push_point() noexcept
    { uint32_t tmp, next;
      
      do {
        do {
          tmp = push_point;
          next = next_point(tmp);
        } while ( next == pop_point );
      } while (!push_point.compare_exchange_weak(tmp, next));
      
      return tmp;
    }
    
  public:
    
    static constexpr uint32_t ReservedSize = LENGTH;

    constexpr SafeQueue_MPSC() noexcept :
    push_point{0}, cur_size{0}, pop_point{0}
    {}
    
    uint32_t size() const noexcept { return cur_size.load(std::memory_order_relaxed); }
    
    bool empty() const noexcept { return (cur_size.load(std::memory_order_relaxed) != 0); }
    
    void clear()
    {
      push_point = 0;
      cur_size = 0;
      pop_point = 0;
    }
    
    /// Copy push
    void push(const T& val)
    {
      const auto my_push_point = get_push_point();
      //*reinterpret_cast<T *>(data + my_push_point) = val;
      new (data + my_push_point) T(val);

      // wait for previous pushes to complete their notification
      // this ensures that our notification does not lead try_pop to read a no-yet-written element
      while (data_before_point(my_push_point) != size()) { }
      
      cur_size.fetch_add(1); // this marks the moment when the element is actually inserted
    }
    
    /// Move push
    void push(T&& val)
    {
      const auto my_push_point = get_push_point();
      //*reinterpret_cast<T *>(data + my_push_point) = std::move(val);
      new (data + my_push_point) T(std::move(val));

      // wait for previous pushes to complete their notification
      // this ensures that our notification does not lead try_pop to read a no-yet-written element
      while (data_before_point(my_push_point) != size()) { }
      
      cur_size.fetch_add(1); // this marks the moment when the element is actually inserted
    }
    
    bool try_pop(T& val)
    {
      const bool ret = (size() != 0);
      if (ret) {
        //val = *reinterpret_cast<T *>(data + pop_point);
        new (std::addressof(val)) T(*(reinterpret_cast<T *>(data + pop_point)));
        const auto next_pop_point = next_point(pop_point);
  
        // pop_point must be increased after cur_size is reduced.
        //Otherwise its advance may make data_before_point-based loops think they are fulfilled
        cur_size.fetch_sub(1);
        pop_point = next_pop_point;
      }
      return ret;
    }
    
    void pop(T& val)
    {
      while (!try_pop(val)) { }
    }
    
  };

}; // namespace internal
  
}; // namespace depspawn

#endif
