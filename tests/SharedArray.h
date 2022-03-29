/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#ifndef SHARED_ARRAY_H
#define SHARED_ARRAY_H

/// Shared array with cyclic distribution
///
/// Each process consecutively stores in a single memory block all its chunks
template<typename T>
class SharedArray {

  int nelems_;
  int blk_sz_;
  upcxx::global_ptr<T> * data_;     //< data_[i] points to the data in process i
  upcxx::global_ptr<T> local_data_; //< Table data in this process, resident in shared memory

public:
  
  SharedArray() :
  nelems_(0),
  blk_sz_(0),
  data_(nullptr),
  local_data_(nullptr)
  {}

  SharedArray(int nelems, int blk_sz = 1)
  {
    init(nelems, blk_sz);
  }

  void init(const int nelems, int blk_sz = 1)
  {
    nelems_ = nelems;
    blk_sz_ = blk_sz;
    const int np = upcxx::rank_n();
    const int nelems_per_rank = ((nelems_ + blk_sz_ - 1)/blk_sz_ + np - 1) / np * blk_sz;
    data_ = new upcxx::global_ptr<T>[np];
    local_data_ = upcxx::new_array<T>(nelems_per_rank);

    data_[upcxx::rank_me()] = local_data_;
    for (int root = 0; root < np; root++) {
      data_[root] = upcxx::broadcast(data_[root], root).wait();
    }
  }

  upcxx::global_ptr<T> operator[] (int i)
  {
    const int np = upcxx::rank_n();
    const int block_id = i / blk_sz_;
    const int phase = i % blk_sz_;
    const int local_offset = (block_id / np) * blk_sz_ + phase;
    const int owner = block_id % np;
    return data_[owner] + local_offset;
  }
  
  int size() const noexcept { return nelems_; }

  ~SharedArray()
  {
    if (data_ != nullptr) {
      upcxx::delete_array<T>(local_data_);
      delete [] data_;
    }
  }

};

#endif
