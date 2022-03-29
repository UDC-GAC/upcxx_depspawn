/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     bench_ping_pong.cpp
/// \brief    Benchmark ping pong pattern for different problem sizes
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"

using namespace depspawn;

constexpr bool Verbose = false;
constexpr size_t Nrounds = 128;

template<size_t SIZE>
struct Tile {
  unsigned char data_[SIZE];
};

SharedArray<char> Signals;
char Mode;
upcxx::persona_scope *ps_master = nullptr;
std::chrono::high_resolution_clock::time_point t0, t1;

template<size_t SIZE, bool VERBOSE = Verbose>
void wr1_rd2(upcxx::global_ptr<Tile<SIZE>> dest, upcxx::global_ptr<const Tile<SIZE>> input)
{ Tile<SIZE> * const recv =  new Tile<SIZE>{};

  // Without get on input, upcxx_spawn actually never brings the data
  upcxx::rget(input, recv, 1).wait();
  dest.local()->data_[SIZE-1] = recv->data_[SIZE-1] + 1;
  if (VERBOSE) {
    std::cerr << upcxx::rank_me() << ' ' << Mode << ' ' <<  static_cast<unsigned>(dest.local()->data_[SIZE-1]) << std::endl;
  }
  delete recv;
}

template<size_t SIZE>
void test_ping_pong()
{ double upcxx_measurement, upcxx_spawn_measurement;

  const int myrank = upcxx::rank_me();
  const int other_rank = 1 - myrank;

  assert(upcxx::master_persona().active_with_caller());
  
  SharedArray<Tile<SIZE>> tiles(2);

  *(Signals[myrank].local()) = static_cast<char>(!myrank); //Rank 0-> 1; Rank 1 -> 0
  
  // Gasnet measurement
  if (!myrank && Verbose) {
    std::cerr << "upcxx test\n";
  }
  memset(tiles[myrank].local(), 0, SIZE);
  upcxx::barrier();

  {
    upcxx::global_ptr<Tile<SIZE>> ptr = upcxx::new_<Tile<SIZE>>();
    Tile<SIZE> * const recv = ptr.local();
    Mode = 'M'; //Manual

    t0 = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < Nrounds; i++) {
      while(!(*(Signals[myrank].local()))) { upcxx::progress(); }
      upcxx::rget(tiles[other_rank], recv, 1).wait();
      // run f on tiles[myrank] and recv
      wr1_rd2<SIZE>(tiles[myrank], ptr);
      *(Signals[myrank].local()) = 0;
      upcxx::rput(static_cast<char>(1), Signals[1 - myrank]).wait();
    }
    
    upcxx::barrier();
    t1 = std::chrono::high_resolution_clock::now();

    //upcxx::delete_<Tile<SIZE>>(ptr);
    upcxx_measurement = std::chrono::duration <double>(t1 - t0).count();
  }

  // UPC++ DepSpawn measurement
  if (!myrank && Verbose) {
    std::cerr << "upcxx_spawn test\n";
  }
  memset(tiles[myrank].local(), 0, SIZE);
  upcxx::barrier();

  if (ps_master) {
    delete ps_master;
    ps_master = nullptr;
  }

  {
    Mode = 'D'; //DepSpawn
    t0 = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < Nrounds; i++) {
      upcxx_spawn(wr1_rd2<SIZE>, tiles[0], tiles[1]);
      upcxx_spawn(wr1_rd2<SIZE>, tiles[1], tiles[0]);
    }
    
    upcxx_wait_for_all();
    
    t1 = std::chrono::high_resolution_clock::now();
    upcxx_spawn_measurement = std::chrono::duration <double>(t1 - t0).count();
  }
  
  if (!upcxx::master_persona().active_with_caller()) {
    //std::cerr << 'P' << upcxx::rank_me() << "takes\n"; //std
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    std::cerr << 'P' << upcxx::rank_me() << "owns\n"; // never used
  }

  if (!myrank) {
    std::cout << std::right << std::setw(7) << SIZE << ' ' << std::left << std::setw(10) << upcxx_measurement << ' ' << std::setw(10) <<upcxx_spawn_measurement << std::endl;
  }

  upcxx::barrier();
}

int main(int argc, char **argv)
{

  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 4;

  set_threads(nthreads);

  upcxx::init();
  
  const int myrank = upcxx::rank_me();

  if(upcxx::rank_n() != 2) {
    if (!myrank) {
      std::cerr << "\n  This benchmark be run with two processes\n\n";
    }
    exit(EXIT_FAILURE);
  }

  if (!myrank) {
    std::cerr << "Using " << upcxx::rank_n() << " Procs x " << nthreads << " threads\n";
    // This liberates the master persona
    print_upcxx_depspawn_runtime_setup();
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
    std::cerr << "  Size     UPC++  UPC++Depspawn\n";
  }

  upcxx::barrier();
  Signals.init(2);
  upcxx::barrier();
  

  test_ping_pong<1>();
  test_ping_pong<2>();
  test_ping_pong<4>();
  test_ping_pong<8>();
  test_ping_pong<16>();
  test_ping_pong<32>();
  test_ping_pong<64>();
  test_ping_pong<128>();
  test_ping_pong<256>();
  test_ping_pong<512>();
  test_ping_pong<1024>();
  
  test_ping_pong<2048>();
  test_ping_pong<4096>();
  test_ping_pong<8192>();
  test_ping_pong<16384>();
  test_ping_pong<32768>();
  test_ping_pong<65536>();
  test_ping_pong<131072>();
  test_ping_pong<262144>();
  test_ping_pong<524288>();
  test_ping_pong<1048576>();

  test_ping_pong<2097152>();
  test_ping_pong<4194304>();
  test_ping_pong<8388608>();

  upcxx::barrier();

  upcxx::finalize();
  
  return 0;
}

