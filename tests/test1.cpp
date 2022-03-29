/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     test1.cpp
/// \brief    Tests spawn of a single function without arguments
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include "upcxx_depspawn/upcxx_depspawn.h"

using namespace depspawn;

volatile bool test_ok;

void f()
{
  std::cout << "Running f on thread " << std::this_thread::get_id() << std::endl;
  
  printf("Node %d of %d runs f\n", upcxx::rank_me(), upcxx::rank_n());
  test_ok = true;
}

int main(int argc, char **argv)
{
  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 2;
  
  upcxx::init();

  const int myrank = upcxx::rank_me();
  
  set_threads(nthreads);

  test_ok = (myrank != 0);
  
  if (!myrank) {
    std::cerr << "Using " << upcxx::rank_n() << " Procs x " << nthreads << " threads\n";
    std::cerr << "Main thread " << std::this_thread::get_id() << std::endl;
  }
  
  std::cerr << "P " << myrank << "enters\n";

  upcxx::barrier();

  upcxx_spawn(f);

  upcxx_wait_for_all();

  //while (!test_ok) { }

  upcxx::persona_scope scope(upcxx::master_persona());

  upcxx::barrier();

  upcxx::finalize();

  if (!myrank) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  return !test_ok;
}
