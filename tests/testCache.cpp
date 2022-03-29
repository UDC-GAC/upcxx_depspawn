/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/
/// \file     testCache.cpp
/// \brief    Tests correctness and performance of many alternative implementations, including some based on cached_global_ptr parameters
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"
#include "common_io.cpp"  // This is only for serializing parallel prints

std::atomic<int> executed;    //For debugging


using namespace depspawn;

#ifdef NDEBUG
static const int N = 400;
#else
#warning Problem size reduced to 399 (was 8) in Debug mode
static const int N = 399;
#endif

static const int DATA_RANGE = 10;

enum RunsToMake { All, AllExceptCachedDepSpawn, DepSpawn, CachedDepSpawn };

int myrank, nranks;
SharedArray<int> golden_v, seq_test_v, v;
SharedArray<int> sh_flag_v;
upcxx::persona_scope *ps_master = nullptr;
int *local_v_ptr;

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

void incr(upcxx::global_ptr<int> value, upcxx::global_ptr<const int> addend)
{
  /*
  char msg[128];
  int iexecuted = executed.fetch_add();
  if (!(iexecuted % 20)) {
    int local_tmp = value;
    const int tmp = addend;
    sprintf(msg, "P%d[%ld] %d+=%d e%d", myrank, myrank + (value.raw_ptr() - local_v_ptr) * nranks, local_tmp, tmp, iexecuted);
    LOG(msg);
  }
  */
  upcxx::future<int> a_f = upcxx::rget(addend); //future<const int> breaks
  upcxx::future<int> v_f = upcxx::rget(value);
  upcxx::future<int, int> both = upcxx::when_all(v_f, a_f);
  upcxx::future<int> result = both.then([](int a, int b) { return a + b; });
  const int res = result.wait();
  upcxx::rput(res, value).wait();
}

void cached_incr(upcxx::global_ptr<int> value, upcxx::cached_global_ptr<const int> addend)
{
  //if(myrank) LOG('P' << myrank << ' ' << value.where() << "+=" << addend.where());
  upcxx::future<int> a_f = upcxx::rget(addend); //future<const int> breaks
  upcxx::future<int> v_f = upcxx::rget(value);
  upcxx::future<int, int> both = upcxx::when_all(v_f, a_f);
  upcxx::future<int> result = both.then([](int a, int b) { return a + b; });
  const int res = result.wait();
  upcxx::rput(res, value).wait();
}

void opt_loop(SharedArray<int>&v, int begin, upcxx::global_ptr<int> marker, int end, upcxx::global_ptr<const int> addend)
{
  //printf("P%d add: %d begin:%d end:%d\n", myrank, (int)addend, begin, end);
  const int cached_added = upcxx::rget(addend).wait();
  for (int j = begin; j < end; ++j) {
    if ( (j % nranks) == myrank) { //Each one takes care of its local updates
      auto res = upcxx::rget(v[j]).wait() + cached_added;
      upcxx::rput(res, v[j]).wait();
    }
  }
}

void cached_opt_loop(SharedArray<int>&v, int begin, upcxx::global_ptr<int> marker, int end, upcxx::cached_global_ptr<const int> addend)
{
  //printf("P%d add: %d begin:%d end:%d\n", myrank, (int)addend, begin, end);
  const int cached_added = upcxx::rget(addend).wait();
  for (int j = begin; j < end; ++j) {
    if ( (j % nranks) == myrank) { //Each one takes care of its local updates
      auto res = upcxx::rget(v[j]).wait() + cached_added;
      upcxx::rput(res, v[j]).wait();
    }
  }
}

bool check(SharedArray<int>&v, int ntest, std::chrono::high_resolution_clock::time_point time1, const char *description)
{ std::chrono::high_resolution_clock::time_point time2;
  
  upcxx::barrier();

  time2 = std::chrono::high_resolution_clock::now();
  
  // Do they match?
  
  bool test_ok = true;
  if (!myrank) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
    
    for (int i = 0; i < N; ++i) {
      test_ok = test_ok && (upcxx::rget(v[i]).wait() == upcxx::rget(seq_test_v[i]).wait());
    }
    std::cout << "TEST " << std::setw(2) << ntest << ' ' << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
    std::cout << std::setw(6) << duration << " ms. for " << description << std::endl;
  }
  
  upcxx::barrier(); // Wait for test before restoring v for next test
  
  for (int i = 0; i < N; i++) {
    if ( (i % nranks) == myrank) {
      const auto tmp = upcxx::rget(golden_v[i]).wait();
      upcxx::rput(tmp, v[i]).wait();
    }
  }
  
  upcxx::barrier();
  
  return test_ok;
}

bool test(const int ntest)
{
  const char *description;

  if ((ntest < 0) && !myrank) {
    upcxx::rput(0, sh_flag_v[0]).wait();
  }
  
  upcxx::barrier();
  executed = 0; //For debugging
  
  auto time1 = std::chrono::high_resolution_clock::now();
  
  switch (ntest) {
      
    case -3: description = "UPC++ parallel test based on barriers";
      
      for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
          if ( (j % nranks) == myrank) { //Each one takes care of its local updates
            incr(v[j], v[i]);
          }
        }
        /* In the next iteration everyone will use the data v[i+1],
         so it is enough to actually flag to everyone that v[i+1] was updated */
        upcxx::barrier();
      }
      break;

    case -2: description = "UPC++ parallel test based on a progress flag";

      for (int i = 0; i < N; ++i) {
        while (upcxx::rget(sh_flag_v[0]).wait() < i) {
          // Wait for sh_flag to indicate that v[i] is ready
          upcxx::progress();
        }
        for (int j = i+1; j < N; ++j) {
          if ( (j % nranks) == myrank) { //Each one takes care of its local updates
            incr(v[j], v[i]);
            if (j == (i+1)) {
              upcxx::rput(j, sh_flag_v[0]).wait();
            }
          }
        }
      }
      break;
    
    case -1: description = "UPC++ parallel test based on a progress flag with cached value";
      
      for (int i = 0; i < N; ++i) {
        while (upcxx::rget(sh_flag_v[0]).wait() < i) {
          // Wait for sh_flag to indicate that v[i] is ready
          upcxx::progress();
        }
        const int cached_value = upcxx::rget(v[i]).wait();
        for (int j = i+1; j < N; ++j) {
          if ( (j % nranks) == myrank) { //Each one takes care of its local updates
            const auto vj = upcxx::rget(v[j]).wait();
            upcxx::rput(vj + cached_value, v[j]).wait(); // v[j] += cached_value;
            if (j == (i+1)) {
              upcxx::rput(j, sh_flag_v[0]).wait();
            }
          }
        }
      }
      break;
  
    case 1: description = "UPC++ DepSpawn standard test";
      disable_ps_master();
      for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
          upcxx_spawn(incr, v[j], v[i]); //P1 runs T0 dep
        }
      }
      break;
      
    case 2: description = "UPC++ DepSpawn test using a cache";
      disable_ps_master();
      for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
          upcxx_spawn(cached_incr, v[j], v[i]); //P1 runs T0 dep
        }
      }
      break;
      
    case 3: description = "UPC++ DepSpawn test using a tuned function";
      disable_ps_master();
      for (int i = 0; i < N-1; ++i) {
        for (int j = i+1; (j < N) && ((j-i) <= nranks); ++j) {
          upcxx_spawn(opt_loop, v, std::move(j), v[j], N, v[i]);
        }
      }
      break;
      
    case 4: description = "UPC++ DepSpawn test using a tuned function and cache";
      disable_ps_master();
      for (int i = 0; i < N-1; ++i) {
        for (int j = i+1; (j < N) && ((j-i) <= nranks); ++j) {
          upcxx_spawn(cached_opt_loop, v, std::move(j), v[j], N, v[i]);
        }
      }
      break;

    default:

      std::cerr << "Unknown case test number " << ntest << std::endl;
      return false;
  }
  
  if (ntest > 0) { // UPC++ DepSpawn test
    upcxx_wait_for_all();
    enable_ps_master();
  } else { // Standard UPC++ test
    upcxx::barrier();
  }
  
  return check(v, ntest, time1, description);
}

int main(int argc, char **argv)
{ RunsToMake runs_selection = AllExceptCachedDepSpawn;
  bool test_ok;

  const int nthreads = (argc > 1) ? strtoul(argv[1],0,0) : 4;

  set_threads(nthreads);
  
  upcxx::init();
  
  golden_v.init(N);
  seq_test_v.init(N);
  v.init(N);
  sh_flag_v.init(1);

  myrank = upcxx::rank_me();
  nranks = upcxx::rank_n();
  
  if (!myrank) {
    std::cout << "testCache [nthreads] [a|c|d]\n Can only specify [a|c|d] after nthreads\n";
    std::cout << "By default all except the cached DepSpawn tests are run\n";
    std::cout << "a -> all tests; d-> Only DepSpawn tests; c-> Only cached DepSpawn tests\n";
    std::cout << "Using " << nranks << " Procs x " << nthreads << " threads for a vector of " << N << " elements. Cache: " << upcxx::cached_global_ptr<const int>::cacheTypeName() << std::endl;
  }

  if (argc > 2) {
    const char c = argv[2][0];
    switch (c) {
      case 'a':
      case 'A':;
        if (!myrank) {
          std::cout << "Will run all tests\n";
        }
        runs_selection = All;
        break;
      case 'd':
      case 'D':
        if (!myrank) {
          std::cout << "Will run only DepSpawn tests\n";
        }
        runs_selection = DepSpawn;
        break;
      case 'c':
      case 'C':
        if (!myrank) {
          std::cout << "Will run only Cached DepSpawn test\n";
        }
        runs_selection = CachedDepSpawn;
        break;
      default:
        if (!myrank) {
          std::cout << "Ignored algorithm selection " << c << std::endl;
        }
        break;
    }
  }

  local_v_ptr = v[myrank].local(); // for debugging

  // Make some random numbers in the interval 0 -- DATA_RANGE -1
  std::random_device rd;
  std::uniform_int_distribution<int> dist(0, DATA_RANGE - 1);
    
  for (int i = 0; i < N; i++) {
    if ( (i % nranks) == myrank) {
      const int tmp = i; //dist(rd);
      if (v[i].where() != myrank) {
        std::cerr << "Wrong placement!\n";
        return -1;
      }
      *(golden_v[i].local()) = tmp;
      *(seq_test_v[i].local()) = tmp;
      *(v[i].local()) = tmp;
    }
  }
  
  upcxx::barrier(); //Make sure all the arrays are correctly initialized
  
  if (!myrank) {
    std::cout << "Data built" << std::endl;
    /*
    for (int i = 0; i < std::min(10, N/10); ++i) {
      std::cout << "V[" << i << "]=" << v[i] << std::endl;
    }
    */

    // Seq test 1 is only run in process 0
    auto time1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
      for (int j = i+1; j < N; ++j) {
        // seq_test_v[j] += seq_test_v[i];
        upcxx::future<int> a_f = upcxx::rget(seq_test_v[i]);
        upcxx::future<int> v_f = upcxx::rget(seq_test_v[j]);
        upcxx::future<int, int> both = upcxx::when_all(v_f, a_f);
        upcxx::future<int> result = both.then([](int a, int b) { return a + b; });
        const int res = result.wait();
        upcxx::rput(res, seq_test_v[j]).wait();
      }
    }
    
    auto time2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
    std::cout << "UPC++ sequential test in single process done in " << duration << " ms." << std::endl;
  }
  
  switch (runs_selection) {
    case All:
      test_ok = test(-3) && test(-2) && test(-1) && test(1) && test(2) && test(3) && test(4);
      break;
    case AllExceptCachedDepSpawn:
      test_ok = test(-3) && test(-2) && test(-1) && test(1) && test(3);
      break;
    case DepSpawn:
      test_ok = test(1) && test(2) && test(3) && test(4);
      break;
    case CachedDepSpawn:
      test_ok = test(2) && test(4);
      break;
    default:
      return -1; // This should never happen
      break;
  }

  
  upcxx::finalize();
  
  return !test_ok;
}
