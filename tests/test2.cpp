/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     test2.cpp
/// \brief    Tests spawn of independent functions with 0 or 1 arguments within each process
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <thread>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"

using namespace depspawn;

SharedArray<int> Table;

std::atomic<int> test;

void f()
{
  std::cout << "Running f on thread " << std::this_thread::get_id() << " in P " << upcxx::rank_me() << std::endl;

  test++;
}

void g(upcxx::global_ptr<int> value)
{
  std::cout << "Running g on thread " << std::this_thread::get_id() << " in P " << upcxx::rank_me() << std::endl;

  int tmp = upcxx::rget(value).wait();
  printf("Node %d obtained the value %d located in node %d\n", upcxx::rank_me(), tmp, value.where());

  if ( (tmp == value.where()) && (value.where() == upcxx::rank_me()) ) {
    test++;
  }
  
  upcxx::rput(tmp + 1, value).wait();
}

int main(int argc, char **argv)
{
  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 2;
  
  set_threads(nthreads);

  test = 0;

  upcxx::init();
  
  const int myrank = upcxx::rank_me();

  if (!myrank) {
    std::cout << "Using " << upcxx::rank_n() << " Procs x " << nthreads << " threads\n";
    std::cout << "Main thread " << std::this_thread::get_id() << std::endl;
  }

  upcxx::barrier();
  Table.init(upcxx::rank_n());
  upcxx::rput(myrank, Table[myrank]).wait();
  upcxx::barrier();
  
  upcxx_spawn(f);
  
  for (int i = 0; i < upcxx::rank_n(); ++i) {
    upcxx_spawn(g, Table[i]);
  }
  
  const int should_be = myrank ? 1 : 2;
  
  upcxx_wait_for_all();

  std::cout << myrank << " completed upcxx_wait_for_all()\n";

  //while (test != should_be) { };
  
  upcxx::persona_scope scope(upcxx::master_persona());

  upcxx::barrier();
  
  const bool test_ok = (test == should_be) && (*(Table[myrank].local()) == (myrank + 1));

  upcxx::finalize();

  //std::cout << myrank << ' ' << test << ' ' << should_be << std::endl;
  
  if (!myrank || !test_ok) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  return !test_ok;
}
