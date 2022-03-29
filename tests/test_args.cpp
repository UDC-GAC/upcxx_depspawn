/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     test_args.cpp
/// \brief    Tests usage of different kinds of parameters and arguments in tasks
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdio>
#include <thread>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "common_io.cpp"  // This is only for serializing parallel prints
#include "SharedArray.h"

using namespace depspawn;

SharedArray<int> Table;
int myrank;
int *LocalTable;
upcxx::persona_scope *ps_master = nullptr;

//////////
//template <typename T>
//T get(const T& v) { return v; }

template <typename T>
T get(const upcxx::global_ptr<T>& v) { return upcxx::rget(v).wait(); }

//////////
template <typename T>
int where(const T& v) { return -1; }

template <typename T>
int where(const upcxx::global_ptr<T>& v) { return v.where(); }

template <typename T>
int where(const upcxx::cached_global_ptr<T>& v) { return v.where(); }

//////////

void disable_ps_master()
{
  if (ps_master) {
    delete ps_master;
    ps_master = nullptr;
  }
}

void enable_ps_master()
{
  if (!upcxx::master_persona().active_with_caller()) {
    //std::cerr << 'P' << upcxx::rank_me() << "takes\n"; //std
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    std::cerr << 'P' << upcxx::rank_me() << "owns\n"; // never used
  }
}

template <typename IN_PARAM>
void g(upcxx::global_ptr<int> value, IN_PARAM addend, const int i)
{
  // LOG("Node " << upcxx::myrank() << " enters g on " << std::this_thread::get_id());
  /**/
  const int tmp = upcxx::rget(value).wait();
  const int addend_value = get(addend);
  fprintf(stderr, "Ctr=%d P%d: %d from P%d += %d from P%d\n", i, upcxx::rank_me(), tmp, value.where(), addend_value, where(addend));
  upcxx::rput(tmp + addend_value, value).wait();
  /**/
  // fprintf(stderr, "Node %d exits its g\n", upcxx::myrank());
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

template <typename IN_PARAM>
bool basic_test()
{ bool test_ok = true;

  disable_ps_master();

  for (int i = 0; i < upcxx::rank_n(); ++i) {
    const int other_index = (i ? i : upcxx::rank_n()) - 1;
    LocalTable[i] += LocalTable[other_index];
    upcxx_spawn(g<IN_PARAM>, Table[i], Table[other_index], std::move(i)); //v[i] = v[i] + v[i-1]
  }
  
  upcxx_wait_for_all();
  enable_ps_master();

  fprintf(stderr, "P%d -------------\n", myrank);
  
  // print_table();
  
  if (!myrank) {
    for (int i = 0; i < upcxx::rank_n(); ++i) {
      const int tmp = upcxx::rget(Table[i]).wait();
      test_ok = test_ok && (tmp == LocalTable[i]);
      if (tmp != LocalTable[i]) {
        std::cerr << "Table[" << i << "]=" << tmp << " != " << LocalTable[i] << std::endl;
      }
    }
  }
  
  upcxx::barrier();
  
  return test_ok;
}

int main(int argc, char **argv)
{ bool test_ok = true;
  
  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 2;
  
  set_threads(nthreads);

  upcxx::init();
  
  myrank = upcxx::rank_me();

  if (!myrank) {
    std::cout << "Using " << upcxx::rank_n() << " Procs x " << nthreads << " threads\n";
    std::cout << "UPCXX_DEPSPAWN_AUTOMATIC_CACHE: ";
#ifdef UPCXX_DEPSPAWN_AUTOMATIC_CACHE
    std::cout << UPCXX_DEPSPAWN_AUTOMATIC_CACHE << '\n';
#else
    std::cout <<"undefined\n";
#endif
  }
  
  Table.init(upcxx::rank_n());
  LocalTable = new int [upcxx::rank_n()];

  upcxx::barrier();
  upcxx::rput(myrank, Table[myrank]).wait();
  
  for (int i = 0; i < upcxx::rank_n(); i++) {
    LocalTable[i] = i;
  }

  print_table();

  // UPCXX_DEPSPAWN_AUTOMATIC_CACHE transforms upcxx::global_ptr<const int> -> upcxx::cached_global_ptr<const int>
  test_ok = basic_test<upcxx::global_ptr<const int>>()
  && basic_test<upcxx::global_ptr<int>>()
  && basic_test<upcxx::cached_global_ptr<const int>>();
  //&& basic_test<int>() && basic_test<const int>() ;

  upcxx::finalize();

  delete [] LocalTable;

  if (!myrank) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  return !test_ok;
}
