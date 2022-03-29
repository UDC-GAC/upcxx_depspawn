/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     testCacheBasic.cpp
/// \brief    Tests basic functions of cache depending on constness of type
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdio>
#include <ctime>
#include <numeric>
#include <sstream>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"

using namespace depspawn;

constexpr float wait_secs = 0.1f;

#if __cplusplus < 201400
namespace std {
  // note: this implementation does not disable this overload for array types
  template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args)
  {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }
};
#endif

SharedArray<int> Table;
int Myrank, Nranks;
upcxx::persona_scope *ps_master = nullptr;

void mywait(float seconds)
{
  const size_t microseconds = static_cast<size_t>(seconds * 1000000.f);
  std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
}

void print_table(int ntest, int subtest, const char *str = nullptr)
{
  if (!Myrank) {
    printf("TEST %d: subtest %d %s\n", ntest, subtest, (str != nullptr) ? str : "");
    for (int i = 0; i < Nranks; ++i) {
      printf("Table[%d]=%d\n", i, upcxx::rget(Table[i]).wait());
    }
    puts("=========");
  }
}

void gci(upcxx::global_ptr<int> out, upcxx::global_ptr<const int> in)
{
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  mywait(wait_secs);
  f.wait();
}

void gi(upcxx::global_ptr<int> out, upcxx::global_ptr<int> in)
{
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  mywait(wait_secs);
  f.wait();
}

void g_cci(upcxx::global_ptr<int> out, upcxx::cached_global_ptr<const int> in)
{
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  mywait(wait_secs);
  f.wait();
}

/* Prohibited by static_assert(std::is_const<T>::value)
   Also cached_global_ptr complains about multiple overloads of the constructor
   for non cost T because T == std::remove_const<T>::type
void g_ci(upcxx::global_ptr<int> out, upcxx::cached_global_ptr<int> in)
{
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  mywait(wait_secs);
  f.wait();
}
*/

/* Prohibited by static_assert(std::is_const<T>::value)
   Also cached_global_ptr complains about multiple overloads of the constructor
   for non cost T because T == std::remove_const<T>::type
// Test cached_global_ptr on non-const data that are actually written.
// Works fine because the data used is local
void gci_cci(upcxx::cached_global_ptr<int> out, upcxx::cached_global_ptr<int> in)
{
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  mywait(wait_secs);
  f.wait();
}
*/

/**************************************************************/

void gi2(upcxx::global_ptr<int> out, upcxx::global_ptr<int> in)
{
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  auto f2 = upcxx::rput(inval + 1, in);
  mywait(wait_secs);
  f.wait();
  f2.wait();
}

/* Prohibited by static_assert(std::is_const<T>::value)
   Also cached_global_ptr complains about multiple overloads of the constructor
   for non cost T because T == std::remove_const<T>::type
void g_ci2(upcxx::global_ptr<int> out, upcxx::cached_global_ptr<int> in)
{
  std::cerr << "R=" << Myrank << std::endl;
  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  auto f = upcxx::rput(inval + outval, out);
  auto f2 = upcxx::rput(inval + 1, in);
  mywait(wait_secs);
  f.wait();
  f2.wait();
}
*/

/*
   The role of out_xxx and in_xxx is to let DepSpawn know where to run the task

   p1 is the source from which "out" was built, and p2 is the source from which "in" was built.
   Their purpose is to ensure that the cached_global_ptr sources outlive the global_ptr built
   from them so that they write back the last version of the data.
 */
void gi_sp(upcxx::global_ptr<int> out_xxx, upcxx::global_ptr<int> in_xxx,
           std::shared_ptr<upcxx::cached_global_ptr<const int>> p1,
           std::shared_ptr<upcxx::cached_global_ptr<const int>> p2, int debug)
{
  upcxx::global_ptr<int> out = *p1;
  upcxx::global_ptr<int> in = *p2;

  assert(out.where() == Myrank); // Certifies that it points to some local (cached) data
  assert(in.where() == Myrank);  // Certifies that it points to some local (cached) data

  const int inval = upcxx::rget(in).wait();
  const int outval = upcxx::rget(out).wait();
  fprintf(stderr, "%d Rt=%d %d+=%d\n", Myrank, debug, outval, inval);
  auto f = upcxx::rput(inval + outval, out);
  auto f2 = upcxx::rput(inval + 1, in);
  mywait(wait_secs);
  f.wait();
  f2.wait();
}

/**************************************************************/

bool test1(int ntest)
{ int i, local_table[Nranks];

  std::iota(local_table, local_table + Nranks, 0);

  for (i = 1; i < Nranks; i++) {
    local_table[i] += local_table[i-1];
  }
  for (i = Nranks - 1; i > 0; i--) {
    local_table[i] += local_table[i-1];
  }
  
  upcxx::barrier();
  upcxx::rput(Myrank, Table[Myrank]).wait();
  upcxx::barrier();
  
  if (ps_master) {
    delete ps_master;
    ps_master = nullptr;
  }

  for (i = 1; i < Nranks; i++) {
    switch (ntest) {
      case 0:
        upcxx_spawn(gci, Table[i], Table[i-1]);
        break;
      case 1:
        upcxx_spawn(gi, Table[i], Table[i-1]);
        break;
      case 2:
        upcxx_spawn(g_cci, Table[i], Table[i-1]);
        break;
      case 3:
        //upcxx_spawn(g_ci, Table[i], Table[i-1]);
        break;
      case 4:
        //upcxx_spawn(gci_cci, Table[i], Table[i-1]);
        break;
      default:
        break;
    }
  }
  
  for (i = Nranks - 1; i > 0; i--) {
    switch (ntest) {
      case 0:
        upcxx_spawn(gci, Table[i], Table[i-1]);
        break;
      case 1:
        upcxx_spawn(gi, Table[i], Table[i-1]);
        break;
      case 2:
        upcxx_spawn(g_cci, Table[i], Table[i-1]);
        break;
      case 3:
        //upcxx_spawn(g_ci, Table[i], Table[i-1]);
        break;
      case 4:
        //upcxx_spawn(gci_cci, Table[i], Table[i-1]);
        break;
      default:
        break;
    }
  }
  
  upcxx_wait_for_all();
  
  if (!upcxx::master_persona().active_with_caller()) {
    //std::cerr << 'P' << upcxx::rank_me() << "takes\n"; //std
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    std::cerr << 'P' << upcxx::rank_me() << "owns\n"; // never used
  }

  print_table(1, ntest);
  
  for (i = 0; i < Nranks; i++) {
    if (upcxx::rget(Table[i]).wait() != local_table[i]) {
      std::cerr << "Error at rank " << Myrank << " position " << i << std::endl;
      break;
    }
  }
  
  return (i == Nranks);
}

/* Subtest 2 cannot succeed because with the current implementation the
 written items remain in the cache until evicted
 and during the generation of the tasks, we generate in each process many cached_global_ptrs associated
 to each one of the shared elements, many of which will be written. Thus,
 1) the second, third, etc. actual uses of those cached_global_ptrs will find probably outdated data in the cache, and
 2) the last one will write the (wrong) value in the owner.
 
 For example, running with 3 processes , we can get
 
 0 GH P=1 V=1       // Rank 0 gets handle for data v=1 from proc=1
 0 Rt=1 1+=0        // Rank 0 runs task 1, which starts with 1+=0
 ...
 1 Rel 0 (0) R u=0
 0 Rel 2 (2) R u=1
 0 Rel 1 (1) W u=2 //Rank 0 releases cache data v=1 from proc=1 which has been Written, with pending uses_=2
 0 Rel 2 (2) R u=0
 1 Rel 0 (0) R u=0
 0 Rel 1 (1) W u=2
 0 Rel 1 (1) W u=1
 1 GH P=2 V=2
 1 Rt=2 2+=1      //This should turn table[1] into 2
 1 GH P=2 V=3
 1 Rel 3 (2) W u=1
 1 Rt=-2 3+=2     //This should turn table[1] into 3
 1 W 5 -> 2
 0 GH P=1 V=1  // Rank 0 WRONGFULLY gets handle for data v=1 from proc=1 because it is in its stale cache
 0 Rt=-1 1+=1
 0 W 2 -> 1    //Upon final release, Rank 0 overwrites table[1]=2 (in proc 1)
 
 The solution could involve writing back and invalidating in each release, but then we have the problem of dealing with all the
 existing cached_global_ptrs that have been created pointing to that cache entry. This might be done in annotateWritten
 for those that are transformed into global_ptrs for writing, but read-only cached_global_ptrs that point to the same element
 woudl need to check/overload in each read access. This could be a bit inefficient and further complicate the implementation,
 when it does not seem a real use case.
 */
bool test2(int ntest)
{ int i, local_table[Nranks];
  std::shared_ptr<upcxx::cached_global_ptr<const int>> p1, p2;

  std::iota(local_table, local_table + Nranks, 0);
  
  for (i = 1; i < Nranks; i++) {
    local_table[i] += local_table[i-1];
    local_table[i-1]++;
  }
  for (i = Nranks - 1; i > 0; i--) {
    local_table[i] += local_table[i-1];
    local_table[i-1]++;
  }
  
  upcxx::barrier();
  upcxx::rput(Myrank, Table[Myrank]).wait();
  upcxx::barrier();

  if (ps_master) {
    delete ps_master;
    ps_master = nullptr;
  }

  for (i = 1; i < Nranks; i++) {
    switch (ntest) {
      case 0:
        upcxx_spawn(gi2, Table[i], Table[i-1]);
        break;
      case 1:
        upcxx_spawn([](upcxx::global_ptr<int> out, upcxx::global_ptr<int> in, int debug) {
          upcxx::cached_global_ptr<const int> out_c(out);
          upcxx::cached_global_ptr<const int> in_c(in);
          //fprintf(stderr, "%d Rt=%d %d+=%d\n", Myrank, debug, out_c.get(), in_c.get());
          gi2(out_c, in);
        }, Table[i], Table[i-1], std::move(i));
        break;
      case 2:
        p1.reset(new upcxx::cached_global_ptr<const int>(Table[i]));
        p2.reset(new upcxx::cached_global_ptr<const int>(Table[i-1]));
        // You could put *p1 and *p2 instead of Table[i] and Table[i-1], but that
        // would generate local global_ptrs, and all the processes could think that they have to run the function!
        //May be overload upcxx_fill_args for upcxx::cached_global_ptr? But they should NOT be used as user-level arguments
        upcxx_spawn(gi_sp, Table[i], Table[i-1], std::move(p1), std::move(p2), std::move(i));
        break;
      case 3:
        //if (!Myrank) { std::cerr << Table[i].where() << '&' << Table[i-1].where() << std::endl; }
        //upcxx_spawn(g_ci2, Table[i], Table[i-1]);
        break;
      default:
        break;
    }
  }
  
  for (i = Nranks - 1; i > 0; i--) {
    switch (ntest) {
      case 0:
        upcxx_spawn(gi2, Table[i], Table[i-1]);
        break;
      case 1:
        upcxx_spawn([](upcxx::global_ptr<int> out, upcxx::global_ptr<int> in, int debug) {
          upcxx::cached_global_ptr<const int> out_c(out);
          upcxx::cached_global_ptr<const int> in_c(in);
          //fprintf(stderr, "%d Rt=%d %d+=%d\n", Myrank, debug, out_c.get(), in_c.get());
          gi2(out_c, in);
        }, Table[i], Table[i-1], std::move(-i));
        break;
      case 2:
        p1.reset(new upcxx::cached_global_ptr<const int>(Table[i]));
        p2.reset(new upcxx::cached_global_ptr<const int>(Table[i-1]));
        upcxx_spawn(gi_sp, Table[i], Table[i-1], std::move(p1), std::move(p2), std::move(-i));
        break;
      case 3:
        //if (!Myrank) { std::cerr << Table[i].where() << '&' << Table[i-1].where() << std::endl; }
        //upcxx_spawn(g_ci2, Table[i], Table[i-1]);
        break;
      default:
        break;
    }
  }
  
  p1.reset();
  p2.reset();

  upcxx_wait_for_all();
  
  if (!upcxx::master_persona().active_with_caller()) {
    //std::cerr << 'P' << upcxx::rank_me() << "takes\n"; //std
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    std::cerr << 'P' << upcxx::rank_me() << "owns\n"; // never used
  }

  print_table(2, ntest);
  
  for (i = 0; i < Nranks; i++) {
    int Table_i = upcxx::rget(Table[i]).wait();
    if ( Table_i != local_table[i]) {
      std::stringstream ss;
      ss << "Error at rank " << Myrank << " position " << i << " : " << Table_i << "!=" << local_table[i] << ' ' << Table[i].raw_internal(upcxx::detail::internal_only());
      std::cerr << ss.str() << std::endl;
      break;
    }
  }
  
  return (i == Nranks);
}

int main(int argc, char **argv)
{ bool test_ok = true;
  
  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 3;
  
  set_threads(nthreads);
  
  upcxx::init();
  
  Myrank = upcxx::rank_me();
  Nranks = upcxx::rank_n();
  
  if (!Myrank) {
    std::cout << "Using " << Nranks << " Procs x " << nthreads << " threads Cache: " << upcxx::cached_global_ptr<const int>::cacheTypeName() << std::endl;
  }

  Table.init(Nranks);
  
  for (int i = 0; (i < 3) && test_ok; i++) {
    test_ok = test_ok && test1(i);
  }

  for (int i = 0; (i < 2) && test_ok; i++) {
    test_ok = test_ok && test2(i);
  }
  
  upcxx::barrier();

  if (!Myrank || !test_ok) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  return !test_ok;
}
