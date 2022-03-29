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
#include "common_io.cpp"

using namespace depspawn;

SharedArray<int> Table;

std::atomic<int> test;

void f()
{
  LOG("runs f on " << std::this_thread::get_id());
}

void g(upcxx::global_ptr<int> value, upcxx::global_ptr<const int> addend)
{
  auto addend_fut = upcxx::rget(addend);

  LOG("enters g on " << std::this_thread::get_id());

  int tmp = upcxx::rget(value).wait();
  int addend_val = addend_fut.wait();

  LOG(tmp << "from P" << value.where() << " += " << addend_val << " from P" << addend.where());
  
  upcxx::rput(tmp + addend_val, value).wait();
  
  LOG("exits g");
}

void print_table(const char *str = nullptr)
{
  upcxx::barrier();

  if (!upcxx::rank_me()) {
    if (str != nullptr) {
      fprintf(stderr, "%s\n", str);
    }
    for (int i = 0; i < upcxx::rank_n(); ++i) {
      fprintf(stderr, "Table[%d]=%d\n", i, upcxx::rget(Table[i]).wait());
    }
    puts("=========");
  }

  upcxx::barrier();
}

int main(int argc, char **argv)
{ bool test_ok = true;

  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 2;
  
  set_threads(nthreads);

  test = 0;

  upcxx::init();
  
  const int myrank = upcxx::rank_me();

  if (!myrank) {
    std::cout << "Using " << upcxx::rank_n() << " Procs x " << nthreads << " threads\n";
    std::cout << "Main thread " << std::this_thread::get_id() << std::endl;
  }

  const std::thread::id this_id = std::this_thread::get_id();
  
  for (int i = 0; i < upcxx::rank_n(); ++i) {
    if (i == myrank) {
      std::cerr << myrank << '/' << upcxx::rank_n() << " runs main on " << this_id << std::endl;
    }
    upcxx::barrier();
  }

  Table.init(upcxx::rank_n());
  upcxx::rput(myrank, Table[myrank]).wait();
  upcxx::barrier();
  
  print_table();

  upcxx_spawn(f);
  
  for (int i = 0; i < upcxx::rank_n(); ++i) {
    upcxx_spawn(g, Table[i], Table[(i ? i : upcxx::rank_n()) - 1]); //v[i] = v[i] + v[i-1]
  }
  
  upcxx_wait_for_all();

  upcxx::persona_scope scope(upcxx::master_persona());

  LOG(" completed upcxx_wait_for_all()\n");

  print_table();

  upcxx::barrier();
  
  int should_be = upcxx::rank_n() - 1;
  for (int i = 0; i < upcxx::rank_n(); ++i) {
    should_be += i;
    const auto read_value = upcxx::rget(Table[i]).wait();
    test_ok = test_ok && (read_value == should_be);
    if (read_value != should_be) {
      std::cerr << "Table[" << i << "]=" << read_value << " != " << should_be << std::endl;
    }
  }

  upcxx::finalize();

  //std::cout << myrank << ' ' << test << ' ' << should_be << std::endl;
  
  if (!myrank || !test_ok) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  return !test_ok;
}
