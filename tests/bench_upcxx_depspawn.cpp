/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     bench_upcxx_depspawn.cpp
/// \brief    Benchmarking
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"
#include "common_io.cpp"  // This is only for serializing parallel prints

constexpr size_t N = 32768;

using namespace depspawn;

typedef void (*voidfptr)();

std::chrono::high_resolution_clock::time_point t0, t1, t2;

double total_time = 0.;

int Myrank, Nranks;
constexpr size_t BCK = 128/sizeof(int);
SharedArray<int> mx;

struct log_line_t { char buffer[128]; };
SharedArray<log_line_t> Log;

size_t nsize = N;
int nthreads = -1;
int queue_limit = -1;
int retstate = 0;
int global_ntest; ///Used by pr()
upcxx::persona_scope *ps_master = nullptr;

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
    //std::cerr << 'P' << upcxx::rank_me() << "owns\n"; // used for myrank !=0 at startup
  }
}


void cleanmx()
{
  for (size_t i = 0; i < nsize * BCK; i++) {
    auto ptr = mx[i];
    if (ptr.where() == Myrank) {
      *(ptr.local()) = 0;
    }
  }
  upcxx::barrier();
}

void pr(const char * s)
{
  char * const buff = Log[Myrank].local()->buffer;
  const double spawn_time = std::chrono::duration <double>(t1 - t0).count();
  const double running_time = std::chrono::duration <double>(t2 - t1).count();
  const double tot_time = std::chrono::duration <double>(t2 - t0).count();
  
  int tmp = sprintf(buff, "T%2d P%2d %31s. ", global_ntest, Myrank, s);
  sprintf(buff + tmp, "Spawning : %8lf Running: %8lf T:%8lf", spawn_time, running_time, tot_time);
  upcxx::barrier();
  if(!Myrank) {
    for (int i = 0; i < Nranks; i++) {
      log_line_t tmp = upcxx::rget(Log[i]).wait();
      puts(tmp.buffer);
    }
  }
  upcxx::barrier();
  total_time += tot_time;
}

void doerror()
{
  retstate = -1;
  std::cerr << "Error\n";
}

void check_global_error()
{
  int global_retstate = upcxx::reduce_all(retstate, upcxx::op_fast_add).wait();
  if (global_retstate < 0) {
    if (!Myrank) {
      std::cerr << "Exiting due to errors in " << (-global_retstate) << " ranks\n";
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
}

/****************************************************************/

int test_0_grain;

void fvec(upcxx::global_ptr<int> r)
{
  assert(r.where() == Myrank);
  int * const p = r.local();
  for (int i = 0; i < test_0_grain; i++) {
    p[i] = p[i] + 1;
  }
}

constexpr bool isPowerOf2(int i) noexcept {
  return !(i & (i-1));
}

void check_and_zero(const bool global = true)
{
  if (global) {
    for(size_t i = 0; i < (nsize * BCK); i++) {
      auto ptr = mx[i];
      if (ptr.where() == Myrank) {
        if( (*(ptr.local())) != 1 ) {
          doerror();
          break;
        }
        *(ptr.local()) = 0;
      }
    }
  } else {
    upcxx::global_ptr<int> my_ptr = mx[Myrank * BCK];
    int * const p = my_ptr.local();
    const size_t max_grain = (nsize * BCK) / Nranks;
    for(size_t i = 0; i < max_grain; i++) {
      if( p[i] != 1 ) {
        doerror();
        break;
      }
      p[i] = 0;
    }
  }
  
}

/**
  Each invocation to fvec processes test_0_grain elements, so it sweeps on all the
  nsize * BCK elements of mx with test_0_grain step so that all the elements are processed.
  max_grain is the number of elements per rank, and thus the total size of the memory chunk
  owned by each rank, while ptr[i] gives the pointer to the shared chunk of rank i.
 */
void body_test_0(int max_grain,
                 const upcxx::global_ptr<int> * const ptr,
                 std::vector<double>& t_upcxx,
                 std::vector<double>& t_threads)
{ char msg[128];
  
  if (max_grain % test_0_grain) {
    if(!Myrank) {
      printf("max_grain=%d not divisible by grain %d\n", max_grain, test_0_grain);
    }
    return;
  }

  upcxx::barrier();

  disable_ps_master();

  t0 = std::chrono::high_resolution_clock::now();

  if (!isPowerOf2(max_grain)) {
    for(size_t i = 0; i < (nsize * BCK); i += test_0_grain) {
      upcxx_spawn(fvec, ptr[i/max_grain] + (i % max_grain));
    }
  } else {
    for(size_t i = 0; i < (nsize * BCK); i += test_0_grain) {
      upcxx_spawn(fvec, ptr[i/max_grain] + (i & (max_grain - 1)));
    }
  }

  t1 = std::chrono::high_resolution_clock::now();

  upcxx_wait_for_all();

  t2 = std::chrono::high_resolution_clock::now();
  enable_ps_master();

  t_upcxx.push_back(std::chrono::duration <double>(t2 - t0).count());
  check_and_zero(false);

  sprintf(msg, "DepSpawn parallel f grain %d", test_0_grain);
  pr(msg);

  upcxx::barrier();

  t0 = t1 = std::chrono::high_resolution_clock::now();

  if (test_0_grain < max_grain) {
    get_task_pool().parallel_for(0, max_grain, test_0_grain, [ptr](int begin) {
      int * const p = ptr[Myrank].local() + begin;
      for (int i = 0; i < test_0_grain; i++) {
        p[i] = p[i] + 1;
      }
    }, false);
  } else {
    int * const p = ptr[Myrank].local();
    for (int i = 0; i < max_grain; i++) {
      p[i] = p[i] + 1;
    }
  }

  upcxx::barrier();

  t2 = std::chrono::high_resolution_clock::now();

  t_threads.push_back(std::chrono::duration <double>(t2 - t0).count());
  check_and_zero(false);

  sprintf(msg, "parallel_for f grain %d", test_0_grain);
  pr(msg);

}

/**
  Measures the cost of spawning independent tasks on a distributed vector compared
  to doing it by hand with SPMD + threads. Different task granularities are tried
  and the best results are provided.
 */
void test0()
{ std::vector<double> t_upcxx, t_threads;

  if (nsize % Nranks) {
    if(!Myrank) {
      printf("TEST 0 skipped because problem size %zu is not divisible by %d NRanks\n", nsize, Nranks);
    }
    return;
  }

  const int max_grain = (nsize * BCK) / Nranks; // Number of elements per rank
  
  if(!Myrank) {
    printf("nsize=%zu BCK=%zu TOT=%zu Nranks=%d max_grain=%d\n", nsize, BCK, nsize * BCK, Nranks, max_grain);
  }
  
  upcxx::global_ptr<int> * const ptr = new upcxx::global_ptr<int>[Nranks];
  for (int i = 0; i < Nranks; i++) {
    ptr[i] = mx[i * BCK];
  }

  for (test_0_grain = 32; test_0_grain <= max_grain; test_0_grain = test_0_grain * 2) {
    body_test_0(max_grain, ptr, t_upcxx, t_threads);
  }

  // tries other test_0_grain for cases where max_grain / nthreads is not a power of 2
  if( (nthreads > 1) && !(max_grain % nthreads) ) {
    test_0_grain = max_grain / nthreads;
    if (!isPowerOf2(test_0_grain)) {
      test_0_grain = test_0_grain * 16; // Try from 16 times equal grain down to up to 512
      do {
        body_test_0(max_grain, ptr, t_upcxx, t_threads);
        if(test_0_grain & 1) {
          break;
        }
        test_0_grain = test_0_grain / 2;
      } while (test_0_grain >= 512);
    }
  }
  
  if(!Myrank) {
    printf("Global best times P0: DepSpawn: %lf Threads: %lf\n", *std::min_element(t_upcxx.begin(), t_upcxx.end()), *std::min_element(t_threads.begin(), t_threads.end()));
  }

  delete [] ptr;
}

void f(upcxx::global_ptr<int> r)
{
  int tmp = upcxx::rget(r).wait();
  upcxx::rput(tmp + 1, r).wait();
}

/// TRANSFORMED TO BE BASED ON INDEPENDENT TASKS
/** A single main thread spawns nsize tasks with a single independent input */
void test1() 
{

  disable_ps_master();
  t0 = std::chrono::high_resolution_clock::now();
  
  for(size_t i = 0; i < nsize; i++)
    upcxx_spawn(f, mx[i * BCK]);
  
  t1 = std::chrono::high_resolution_clock::now();
  
  upcxx_wait_for_all();
  
  t2 = std::chrono::high_resolution_clock::now();
  enable_ps_master();

  for(size_t i = 0; i < nsize; i++) {
    auto ptr = mx[i * BCK];
    if (ptr.where() == Myrank) {
      if( (*(ptr.local())) != 1 ) {
        doerror();
        break;
      }
      *(ptr.local()) = 0;
    }
  }
  
  pr("f(int&) without overlaps");
}

void g(upcxx::global_ptr<int> i0, upcxx::global_ptr<int> i1, upcxx::global_ptr<int> i2, upcxx::global_ptr<int> i3, upcxx::global_ptr<int> i4, upcxx::global_ptr<int> i5, upcxx::global_ptr<int> i6, upcxx::global_ptr<int> i7) {
  auto fi0 = upcxx::rget(i0);
  auto fi1 = upcxx::rget(i1);
  auto fi2 = upcxx::rget(i2);
  auto fi3 = upcxx::rget(i3);
  auto fi4 = upcxx::rget(i4);
  auto fi5 = upcxx::rget(i5);
  auto fi6 = upcxx::rget(i6);
  auto fi7 = upcxx::rget(i7);
  auto fi01 = fi0.then([i0](int i){ return upcxx::rput(i + 1, i0); });
  auto fi11 = fi1.then([i1](int i){ return upcxx::rput(i + 1, i1); });
  auto fi21 = fi2.then([i2](int i){ return upcxx::rput(i + 1, i2); });
  auto fi31 = fi3.then([i3](int i){ return upcxx::rput(i + 1, i3); });
  auto fi41 = fi4.then([i4](int i){ return upcxx::rput(i + 1, i4); });
  auto fi51 = fi5.then([i5](int i){ return upcxx::rput(i + 1, i5); });
  auto fi61 = fi6.then([i6](int i){ return upcxx::rput(i + 1, i6); });
  auto fi71 = fi7.then([i7](int i){ return upcxx::rput(i + 1, i7); });
  upcxx::when_all(fi01, fi11, fi21, fi31, fi41, fi51, fi61, fi71).wait();
}

/** A single main thread spawns nsize tasks with 8 arguments by reference.
    All the tasks are independent.
 */
void test2() 
{
  disable_ps_master();
  t0 = std::chrono::high_resolution_clock::now();
  
  for(size_t i = 0; i < nsize; i++)
    upcxx_spawn(g, mx[i * BCK], mx[i * BCK + 7], mx[i * BCK + 1], mx[i * BCK + 6], mx[i * BCK + 2], mx[i * BCK + 5], mx[i * BCK + 3], mx[i * BCK + 4]);
  
  t1 = std::chrono::high_resolution_clock::now();
  
  upcxx_wait_for_all();
  
  t2 = std::chrono::high_resolution_clock::now();
  enable_ps_master();

  for(size_t i = 0; i < nsize; i++) {
    for (size_t j = 0; j < 8; ++j) {
      auto ptr = mx[i * BCK + j];
      if (ptr.where() == Myrank) {
        if( (*(ptr.local())) != 1) {
          doerror();
          i = nsize; //Break outer loop too
          break;
        }
        *(ptr.local()) = 0;
      }
    }
  }
  
  pr("f(8 int&) without overlaps");
  
}


void depprev(upcxx::cached_global_ptr<const int> prev, upcxx::global_ptr<int> out) {
  int tmp = upcxx::rget(prev).wait();
  upcxx::rput(tmp + 1, out).wait();
}

//#define DEBUG
#ifdef DEBUG
void depprev_dbg(upcxx::cached_global_ptr<const int> prev, upcxx::global_ptr<int> out, const int i) {
  int tmp = upcxx::rget(prev).wait();
  if (tmp != i) {
    fprintf(stderr, "err at task %d\n", i);
    abort();
  }
  upcxx::rput(tmp + 1, out).wait();
}
#endif

/** A single main thread spawns nsize tasks in which task i depends
    on task i-1 (on an arg it writes by reference).
 */
void test3()
{
  const bool default_run = (nsize == N);
  if(default_run) { //Shorten problem size because of slow test
    nsize = 20000;
    if(!Myrank) {
      printf("RUNNING T3 WITH PROBLEM SIZE %zu\n", nsize);
    }
  }

  ///First a simple run in UPC++, for comparison
  t0 = std::chrono::high_resolution_clock::now();
  
  if (!Myrank) {
    int tmp = upcxx::rget(mx[0]).wait();
    upcxx::rput(tmp + 1, mx[0]).wait();
  }
  
  for(size_t i = 1; i < nsize; i++) {
    upcxx::barrier();
    int tmp = upcxx::rget(mx[(i-1) * BCK]).wait();
    upcxx::rput(tmp + 1, mx[i * BCK]).wait();
  }
  
  t1 = std::chrono::high_resolution_clock::now();
  
  upcxx::barrier();
  
  t2 = std::chrono::high_resolution_clock::now();
  
  for(size_t i = 0; i < nsize; i++) {
    auto ptr = mx[i * BCK];
    if (ptr.where() == Myrank) {
      int tmp = *(ptr.local());
      if(tmp != (i+1)) {
        fprintf(stderr, "%zu -> %d\n", i, tmp);
        doerror();
        break;
      }
      *(ptr.local()) = 0;
    }
  }
  
  pr("UPC++ f(int, int&) dep->");

  upcxx::barrier();

  /// Actual upcxx_spawn test
  if((nthreads < 0) || (nthreads > 2)) {
    if (!Myrank) {
      printf("**RUNNING T3 WITH 2 THREADS INSTEAD OF %d**\n", nthreads);
    }
    set_threads(2);
  }
  
  disable_ps_master();
  t0 = std::chrono::high_resolution_clock::now();

#ifdef DEBUG
  upcxx_spawn(depprev_dbg, mx[0], mx[0], 0);
  
  for(size_t i = 1; i < nsize; i++) {
    upcxx_spawn(depprev_dbg, mx[(i-1) * BCK], mx[i * BCK], std::move(i));
  }
#else
  upcxx_spawn(depprev, mx[0], mx[0]);
  
  for(size_t i = 1; i < nsize; i++) {
    upcxx_spawn(depprev, mx[(i-1) * BCK], mx[i * BCK]);
  }
#endif

  t1 = std::chrono::high_resolution_clock::now();
  
  upcxx_wait_for_all();
  
  t2 = std::chrono::high_resolution_clock::now();
  enable_ps_master();

  if((nthreads < 0) || (nthreads > 2)) { //restore threads
    set_threads(nthreads);
  }

  for(size_t i = 0; i < nsize; i++) {
    auto ptr = mx[i * BCK];
    if (ptr.where() == Myrank) {
      int tmp = *(ptr.local());
      if(tmp != (i+1)) {
        fprintf(stderr, "%zu -> %d\n", i, tmp);
        doerror();
        break;
      }
      *(ptr.local()) = 0;
    }
  }

  pr("f(int, int&) dep->");
  
  if (default_run) { //Restore default problem size
    nsize = N;
  }
}

void depsame(upcxx::global_ptr<int> iout) {
  //LOG("B " << (&out - (int*)mx) / (128/sizeof(int)) );
  int tmp = upcxx::rget(iout).wait();
  upcxx::rput(tmp + 1, iout).wait();
  //LOG("E " << (&out - (int*)mx) / (128/sizeof(int)) );
}

/** A single main thread spawns nsize tasks all of which depend on the same element by reference */
void test4()
{
  disable_ps_master();
  t0 = std::chrono::high_resolution_clock::now();
  
  for(size_t i = 0; i < nsize; i++) {
    upcxx_spawn(depsame, mx[0]);
  }
  
  t1 = std::chrono::high_resolution_clock::now();
  
  upcxx_wait_for_all();
  
  t2 = std::chrono::high_resolution_clock::now();
  enable_ps_master();

  if(!Myrank){
    int result = upcxx::rget(mx[0]).wait();
    if( result != nsize ) {
      std::cerr << " -> " << result;
      doerror();
    }
    upcxx::rput(0, mx[0]).wait();
  }
  
  pr("f(int&) depsame");
  
}

/** Simulates pattern of matrix_inverse */
void test5()
{ size_t dim;
  
  const bool default_run = (nsize == N);
  if(default_run) { //Shorten problem size because of slow test
    nsize = 1000;
    if(!Myrank) {
      printf("RUNNING T5 WITH PROBLEM SIZE %zu\n", nsize);
    }
  }

  for(dim = 1; (dim * dim) <= nsize; dim++);
  dim--;
  
  if(!Myrank) {
    printf("%zu x %zu blocks\n", dim, dim);
  }

  auto index_mx = [dim](int i, int j) { return mx[(i * dim + j) * BCK]; };

  disable_ps_master();
  t0 = std::chrono::high_resolution_clock::now();
  
  for(size_t n = 0; n < dim; n++) {

    auto pivot_inverse = index_mx(n, n);

    upcxx_spawn( [](upcxx::global_ptr<int> pivot) {
                    int tmp = upcxx::rget(pivot).wait();
                    upcxx::rput(tmp + 1, pivot).wait();
                 }, pivot_inverse);

    for (int j = 0; j < dim; j++)
    {
      if (j == n) continue;
      
      upcxx_spawn( [](upcxx::global_ptr<int> dest_and_right, upcxx::cached_global_ptr<const int> left) {
                     int tmp = upcxx::rget(left).wait();
                     int tmp2 = upcxx::rget(dest_and_right).wait();
                     upcxx::rput(tmp + tmp2, dest_and_right).wait();
                  }, index_mx(n, j), pivot_inverse);
    }
    
    for (int i = 0; i < dim; i++)
    {
      if (i == n) continue;
      
      auto tin = index_mx(i, n);  //This is a upcxx::global_ptr<int>
      
      for (int j = 0; j < dim; j++)
      {
        if (j == n) continue;
        
        //spawn(&tile::multiply_subtract_in_place, b.m_tiles[dim*i+j], tin, b.m_tiles[dim*n+j]);
        upcxx_spawn( [](upcxx::global_ptr<int> dest, upcxx::cached_global_ptr<const int> left, upcxx::cached_global_ptr<const int> right) {
          int tmp = upcxx::rget(left).wait();
          int tmp2 = upcxx::rget(right).wait();
          int tmp3 = upcxx::rget(dest).wait();
          upcxx::rput(tmp + tmp2 + tmp3, dest).wait();
        }, index_mx(i, j), tin, index_mx(n, j));
      }
      
      //spawn(dsp_multiply_negate, tin, pivot_inverse);
      upcxx_spawn( [](upcxx::global_ptr<int> dest, upcxx::cached_global_ptr<const int> right) {
        int tmp2 = upcxx::rget(right).wait();
        int tmp3 = upcxx::rget(dest).wait();
        upcxx::rput(tmp2 + tmp3, dest).wait();
      }, tin, pivot_inverse);
    }

  }
  
  t1 = std::chrono::high_resolution_clock::now();
  
  upcxx_wait_for_all();
  
  t2 = std::chrono::high_resolution_clock::now();
  enable_ps_master();

  for(size_t i = 0; i < (dim * dim); i++) {
    auto ptr = mx[i * BCK];
    if (ptr.where() == Myrank) {
      *(ptr.local()) = 0;
    }
  }
  
  pr("matrix_inverse sim (no test)");
  
  if (default_run) { //Restore default problem size
    nsize = N;
  }

}


//////////////////////////////////////////////////////
/////               Common part                  /////
//////////////////////////////////////////////////////

constexpr voidfptr tests[] =
 {test0, test1, test2, test3, test4, test5};

constexpr int NTESTS = sizeof(tests) / sizeof(tests[0]);
bool dotest[NTESTS];

void show_help()
{
  puts("bench_spawn [-h] [-q limit] [-t nthreads] [-T ntest] [problemsize]");
  puts("-h          Display help and exit");
  puts("-q limit    # of pending ready tasks that makes a spawning thread steal one");
  puts("-t nthreads Run with nthreads threads. Default is automatic (-1)");
  printf("-T ntest    Run test ntest in [0, %u]\n", NTESTS - 1);
  puts("            Can be used multiple times to select several tests");
  puts("            By default all the tests except 0 are run\n");
  printf("problemsize defaults to %zu\n", N);
}

int process_arguments(int argc, char **argv)
{ int c;
  bool tests_specified = false;

  upcxx::init();

  Myrank = upcxx::rank_me();
  Nranks = upcxx::rank_n();
  
  std::fill(dotest, dotest + NTESTS, false);

  while ( -1 != (c = getopt(argc, argv, "hq:t:T:")) ) {
    switch (c) {
        
      case 'q' : /* queue limit */
        queue_limit = atoi(optarg);
        break;
        
      case 't': /* threads */
	nthreads = atoi(optarg);
	break;
        
      case 'T': /* tests */
	c = atoi(optarg);
	if(c < 0 || c >=  NTESTS) {
	  if (!Myrank) printf("The test number must be in [0, %u]\n", NTESTS - 1);
	  return -1;
	}
	dotest[c] = true;
	tests_specified = true;
	break;
	
      case 'h':
	if (!Myrank) show_help();
	
      default: /* unknown or missing argument */
	return -1;     
    }
  }

  if (optind < argc) {
    nsize = atoi(argv[optind]); /* first non-option argument */
  }

//  if(nsize > N) {
//    if (!Myrank) printf("The problem size cannot exceed %zu\n", N);
//    return -1;
//  }

  if(!tests_specified) {
    // By default do not runt Test 0
    std::fill(dotest + 1, dotest + NTESTS, true);
  }

  set_threads(nthreads); //You must set the number of threads before default runtime setup happens
  
  if (!Myrank) {
    printf("Running problem size %zu with %i procs x %i threads (sizeof(int)=%zu) Cache: %s\n", nsize, Nranks, nthreads, sizeof(int), upcxx::cached_global_ptr<const int>::cacheTypeName());
    print_upcxx_depspawn_runtime_setup(); // This gives place to the default runtime setup in rank 0
  }
  
  enable_ps_master();

  upcxx::barrier();
  
  return 0;
}

int main(int argc, char **argv)
{
  if(process_arguments(argc, argv) == -1) {
    return -1;
  }
  
  assert(upcxx::master_persona().active_with_caller());

  if (queue_limit >= 0) {
    if (!Myrank) printf("Setting queue limit to %d\n", queue_limit);
    set_task_queue_limit(queue_limit);
  }
  
  assert(upcxx::master_persona().active_with_caller());

  mx.init(nsize * BCK, BCK);
  Log.init(Nranks);
  cleanmx();
  
  for(global_ntest = 0; global_ntest < NTESTS; global_ntest++) {
    if(dotest[global_ntest]) {
      if (!Myrank) { printf("NO EXACT_MATCH TEST %d:\n", global_ntest); }
      set_UPCXX_DEPSPAWN_EXACT_MATCH(false);
      enable_ps_master();
      assert(upcxx::master_persona().active_with_caller());
      (*tests[global_ntest])();
      check_global_error(); //upcxx::barrier();
      if (!Myrank) { printf("EXACT_MATCH TEST %d:\n", global_ntest); }
      set_UPCXX_DEPSPAWN_EXACT_MATCH(true);
      enable_ps_master();
      assert(upcxx::master_persona().active_with_caller());
      (*tests[global_ntest])();
      check_global_error(); //upcxx::barrier();
    }
  }
  
  if (!Myrank) printf("Total : %8lf\n", total_time);
  
  upcxx::finalize();

  return retstate;
}
