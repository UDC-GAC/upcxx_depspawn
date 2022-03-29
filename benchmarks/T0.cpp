/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     T0.cpp
/// \brief    Benchmarking
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <chrono>
#include <thread>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"

constexpr int N = 32768;
constexpr int NReps = 4;
constexpr double InitValue = 1.0;

SharedArray<double> mx;

bool TuneMode = false;
bool TryPar = false;
bool LocalMeasurement = false;
bool Tricky = false;
bool Filtered = false;
int Myrank, Nranks;
int cost = (1 << 20);
int nsize = N;
int nthreads = -1;
int queue_limit = -1;
int retstate = 0;
int max_grain; ///< Maximum task size, i.e., total number of elements per rank
double serial_time, serial_value;
upcxx::persona_scope *ps_master = nullptr;

std::chrono::high_resolution_clock::time_point t0, t2;


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
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  }
}

void init_mx()
{
  for (size_t i = 0; i < nsize; i++) {
    auto ref = mx[i];
    if (ref.where() == Myrank) {
      *(ref.local()) = InitValue;
    }
  }
  upcxx::barrier();
}

void show_help()
{
  puts("T0 [-h] [-q limit] [-t nthreads] [-c cost] [-l] [-T] [-P] [-x] [problemsize]");
  puts("-h          Display help and exit");
  puts("-q limit    # of pending ready tasks that makes a spawning thread steal one");
  puts("-t nthreads Run with nthreads threads. Default is automatic (-1)");
  puts("-c cost     Flops per task");
  puts("-f          Filtered DepSpawn that only spawns local tasks");
  puts("-l          Local measurement within each process before barrier/upcxx_wait_for_all");
  puts("-T          Tune cost [1..-c cost]");
  puts("-P          Try parallelism within each task");
  puts("-x          Enqueue empty tasks for non-local tasks");
  printf("problemsize defaults to %d and cost to %d. %d repetitions per experiment\n", N, cost, NReps);
}

double task(double v)
{
  assert(v == InitValue);
  const int lim = cost >> 1;
  for (int i = 0; i < lim; i++) {
    v = 1.000134 * v + 1e-6;
  }
  return v;
}


/// Number of elements processed
/// within each individual DepSpawn task or top level parallel_for iteration
int test_0_grain;

/// Number of parallel subtasks (<=test_0_grain) allowed
//  within each individual DepSpawn task or top level parallel_for iteration
int test_0_subtasks = 1;
int test_0_subblock;

void fvec(upcxx::global_ptr<double> r)
{
  assert(r.where() == Myrank);
  double * const p = r.local();
  for (int i = 0; i < test_0_grain; i++) {
    p[i] = task(p[i]);
  }
}


void fvecpar(upcxx::global_ptr<double> r)
{
  assert(r.where() == Myrank);
  double * const p = r.local();
  depspawn::get_task_pool().hp_soft_parallel_for(0, test_0_grain, test_0_subblock, [p](int begin) {
    const int end = std::min(begin + test_0_subblock, test_0_grain);
    for (int i = begin; i < end; i++) {
      p[i] = task(p[i]);
    }
  });
}

constexpr bool isPowerOf2(int i) noexcept {
  return !(i & (i-1));
}

void verify_reset(double * const res, const double expected, const int index)
{
  if (*res != expected) {
    retstate = -1;
    fprintf(stderr, "grain=%d subtasks=%d subblock=%d\n", test_0_grain, test_0_subtasks, test_0_subblock);
    fprintf(stderr, "Err: P%d [%d] got %lf!=%lf\n",  Myrank, index, *res, expected);
    exit(EXIT_FAILURE);
  }
  *res = InitValue;
}

void check_and_reset(const bool global = true)
{
  upcxx::barrier();

  if (global) {
    for(size_t i = 0; i < nsize; i++) {
      const auto ptr = mx[i];
      if (ptr.where() == Myrank) {
        verify_reset(ptr.local(), serial_value, i);
      }
    }
  } else {
    const upcxx::global_ptr<double> my_ptr = mx[Myrank * max_grain];
    double * const p = my_ptr.local();
    for(size_t i = 0; i < max_grain; i++) {
      verify_reset(&p[i], serial_value, i);
    }
  }

  upcxx::barrier();
}

void voidf() noexcept {}

template<typename F>
double depspawn_run(F& f, const upcxx::global_ptr<double> * const ptr)
{ double ret;

  upcxx::barrier();
  
  disable_ps_master();

  t0 = std::chrono::high_resolution_clock::now();
  
  if (Filtered) {
    if (Tricky) {
      for(size_t i = 0; i < nsize; i += test_0_grain) {
        const int rank = i/max_grain;
        if (rank == Myrank) {
          depspawn::upcxx_spawn(f, ptr[rank] + (i % max_grain));
        } else {
          depspawn::upcxx_cond_spawn(false, voidf);
        }
      }
    } else {
      if (!isPowerOf2(max_grain)) {
        for(size_t i = 0; i < nsize; i += test_0_grain) {
          const auto tmp_ptr = ptr[i/max_grain] + (i % max_grain);
          depspawn::upcxx_cond_spawn(tmp_ptr.where() == Myrank, f, std::move(tmp_ptr));
        }
      } else {
        for(size_t i = 0; i < nsize; i += test_0_grain) {
          const auto tmp_ptr = ptr[i/max_grain] + (i & (max_grain - 1));
          depspawn::upcxx_cond_spawn(tmp_ptr.where() == Myrank, f, std::move(tmp_ptr));
        }
      }
    }
  } else {
    if (Tricky) {
      for(size_t i = 0; i < nsize; i += test_0_grain) {
        const int rank = i/max_grain;
        if (rank == Myrank) {
          depspawn::upcxx_spawn(f, ptr[rank] + (i % max_grain));
        } else {
          depspawn::upcxx_spawn(voidf);
        }
      }
    } else {
      if (!isPowerOf2(max_grain)) {
        for(size_t i = 0; i < nsize; i += test_0_grain) {
          depspawn::upcxx_spawn(f, ptr[i/max_grain] + (i % max_grain));
        }
      } else {
        for(size_t i = 0; i < nsize; i += test_0_grain) {
          depspawn::upcxx_spawn(f, ptr[i/max_grain] + (i & (max_grain - 1)));
        }
      }
    }
  }
  
  if (LocalMeasurement) {
    t2 = std::chrono::high_resolution_clock::now();
    depspawn::upcxx_wait_for_all();
    double tmp = std::chrono::duration <double>(t2 - t0).count();
    enable_ps_master();
    ret = upcxx::reduce_one(tmp, upcxx::op_fast_add, 0).wait() / Nranks;
  } else {
    depspawn::upcxx_wait_for_all();
  
    t2 = std::chrono::high_resolution_clock::now();
    ret = std::chrono::duration <double>(t2 - t0).count();
    enable_ps_master();
  }
  
  return ret;
}

template<typename F>
double multithread_run(F& f, const upcxx::global_ptr<double> * const ptr)
{ double ret;
  
  upcxx::barrier();
  
  t0 = std::chrono::high_resolution_clock::now();
  
  if (test_0_grain < max_grain) {
    // &f instead of f : https://stackoverflow.com/questions/34815698/c11-passing-function-as-lambda-parameter
    auto fs = typename std::decay<F>::type(f);
    depspawn::get_task_pool().parallel_for(0, max_grain, test_0_grain, [fs, ptr](int begin) {
      fs(ptr[Myrank] + begin);
    }, false);
  } else {
    f(ptr[Myrank]);
  }
  
  if (LocalMeasurement) {
    t2 = std::chrono::high_resolution_clock::now();
    double tmp = std::chrono::duration <double>(t2 - t0).count();
    ret = upcxx::reduce_one(tmp, upcxx::op_fast_add, 0).wait() / Nranks; // implicit barrier
    ret = ret / Nranks;
  } else {
    upcxx::barrier();
  
    t2 = std::chrono::high_resolution_clock::now();
    ret = std::chrono::duration <double>(t2 - t0).count();
  }
  
  return ret;
}


void body_test_0(const upcxx::global_ptr<double> * const ptr,
                 std::vector<double>& t_upcxx_noem,
                 std::vector<double>& t_upcxx_em,
                 std::vector<double>& t_mt,
                 std::vector<int>& s_upcxx_noem,
                 std::vector<int>& s_upcxx_em,
                 std::vector<int>& s_mt,
                 std::vector<int>& grains)
{ double upcxx_times_noem[NReps], upcxx_times_em[NReps], mt_times[NReps];
  std::vector<double> t_upcxx_noem_subtasks, t_upcxx_em_subtasks, t_mt_subtasks;
  std::vector<int> subtasks(1, 1);

  if (max_grain % test_0_grain) {
    if(!Myrank) {
      printf("max_grain=%d not divisible by grain %d\n", max_grain, test_0_grain);
    }
    return;
  }
  
  grains.push_back(test_0_grain);
  
  if (TryPar) {
    const int actual_threads = (nthreads == -1) ? std::thread::hardware_concurrency() : nthreads;
    const int max_sub_tasks = std::min(actual_threads, test_0_grain);
    for (test_0_subtasks = 2; test_0_subtasks <= max_sub_tasks; test_0_subtasks *= 2) {
      subtasks.push_back(test_0_subtasks);
    }
    if (!(max_sub_tasks % 3)) {
      for (test_0_subtasks = 3; test_0_subtasks <= max_sub_tasks; test_0_subtasks *= 2) {
        subtasks.push_back(test_0_subtasks);
      }
    }
  }
  
//  for (int num_subtasks : subtasks) {
//    printf("S%d ", num_subtasks);
//  }
//  printf("\n");
  
  for (int num_subtasks : subtasks) {

    test_0_subtasks = num_subtasks;
    test_0_subblock = (test_0_grain + test_0_subtasks - 1) / test_0_subtasks;
    void (* const f) (upcxx::global_ptr<double> r) = (test_0_subtasks == 1) ? fvec : fvecpar;
    
    for (int nrep = 0; nrep < NReps; nrep++) {
      
      depspawn::set_UPCXX_DEPSPAWN_EXACT_MATCH(false);
      enable_ps_master();
      upcxx_times_noem[nrep] = depspawn_run(*f, ptr);
      
      check_and_reset(false); //includes barrier
      //////////////////////////////////////////
      
      depspawn::set_UPCXX_DEPSPAWN_EXACT_MATCH(true);
      upcxx_times_em[nrep] = depspawn_run(*f, ptr);
      
      check_and_reset(false); //includes barrier
      //////////////////////////////////////////
      
      mt_times[nrep] = multithread_run(*f, ptr);
      
      check_and_reset(false); // includes barrier
    }
    
    const double best_upcxx_time_noem = *std::min_element(upcxx_times_noem, upcxx_times_noem + NReps);
    const double best_upcxx_time_em = *std::min_element(upcxx_times_em, upcxx_times_em + NReps);
    const double best_mt_time = *std::min_element(mt_times,   mt_times + NReps);
    
    t_upcxx_noem_subtasks.push_back(best_upcxx_time_noem);
    t_upcxx_em_subtasks.push_back(best_upcxx_time_em);
    t_mt_subtasks.push_back(best_mt_time);
  }
  
  const auto best_upcxx_time_noem_it = std::min_element(t_upcxx_noem_subtasks.begin(), t_upcxx_noem_subtasks.end());
  const auto best_upcxx_time_em_it = std::min_element(t_upcxx_em_subtasks.begin(), t_upcxx_em_subtasks.end());
  const auto best_mt_time_it = std::min_element(t_mt_subtasks.begin(), t_mt_subtasks.end());

  t_upcxx_noem.push_back(*best_upcxx_time_noem_it);
  t_upcxx_em.push_back(*best_upcxx_time_em_it);
  t_mt.push_back(*best_mt_time_it);
  
  s_upcxx_noem.push_back(subtasks[std::distance(t_upcxx_noem_subtasks.begin(), best_upcxx_time_noem_it)]);
  s_upcxx_em.push_back(subtasks[std::distance(t_upcxx_em_subtasks.begin(), best_upcxx_time_em_it)]);
  s_mt.push_back(subtasks[std::distance(t_mt_subtasks.begin(), best_mt_time_it)]);
}


void test0()
{ std::vector<double> t_upcxx_noem, t_upcxx_em, t_mt;
  std::vector<int> s_upcxx_noem, s_upcxx_em, s_mt;
  std::vector<int> grains;
  
  upcxx::global_ptr<double> * const ptr = new upcxx::global_ptr<double>[Nranks];
  for (int i = 0; i < Nranks; i++) {
    ptr[i] = mx[i * max_grain];
  }
  
  for (test_0_grain = 1; test_0_grain <= max_grain; test_0_grain = test_0_grain * 2) {
    body_test_0(ptr, t_upcxx_noem, t_upcxx_em, t_mt, s_upcxx_noem, s_upcxx_em, s_mt, grains);
  }
  
  if (!(max_grain % 3)) { // for 24 cores
    for (test_0_grain = 3; test_0_grain <= max_grain; test_0_grain = test_0_grain * 2) {
      body_test_0(ptr, t_upcxx_noem, t_upcxx_em, t_mt, s_upcxx_noem, s_upcxx_em, s_mt, grains);
    }
  }
  
  if(!Myrank) {
    std::vector<int> grains_sort(grains);
    std::sort(grains_sort.begin(), grains_sort.end());
    for (const int v: grains_sort) {
      const auto it = std::find(grains.begin(), grains.end(), v);
      assert(it != grains.end());
      const int dif = std::distance(grains.begin(), it);
      if (!TryPar) {
        assert( (s_upcxx_noem[dif] * s_upcxx_em[dif] * s_mt[dif]) == 1 );
        printf("Grain=%6d  Tasks_rank=%6d  T_UPCxx_Dsp_NOEM=%.9lf  T_UPCxx_Dsp_EM=%.9lf  T_MT=%.9lf\n", v, max_grain/v, t_upcxx_noem[dif], t_upcxx_em[dif], t_mt[dif]);
      } else {
      printf("Grain=%6d  Tasks_rank=%6d  T_UPCxx_Dsp_NOEM=%.9lf (%2ds) T_UPCxx_Dsp_EM=%.9lf (%2ds) T_MT=%.9lf (%2ds)\n", v, max_grain/v, t_upcxx_noem[dif], s_upcxx_noem[dif], t_upcxx_em[dif], s_upcxx_em[dif], t_mt[dif], s_mt[dif]);
      }
    }
    const auto best_upcxx_time_noem_it = std::min_element(t_upcxx_noem.begin(), t_upcxx_noem.end());
    const int best_upcxx_noem_exp = std::distance(t_upcxx_noem.begin(), best_upcxx_time_noem_it);
    const auto best_upcxx_time_em_it = std::min_element(t_upcxx_em.begin(), t_upcxx_em.end());
    const int best_upcxx_em_exp = std::distance(t_upcxx_em.begin(), best_upcxx_time_em_it);
    const auto best_mt_time_it = std::min_element(t_mt.begin(), t_mt.end());
    const int best_mt_exp = std::distance(t_mt.begin(), best_mt_time_it);
    
    const auto best_upcxx_time_it = (*best_upcxx_time_noem_it < *best_upcxx_time_em_it) ? best_upcxx_time_noem_it : best_upcxx_time_em_it;
    const int best_upcxx_exp = (*best_upcxx_time_noem_it < *best_upcxx_time_em_it) ? best_upcxx_noem_exp : best_upcxx_em_exp;
    const int best_upcxx_subtasks = (*best_upcxx_time_noem_it < *best_upcxx_time_em_it) ? s_upcxx_noem[best_upcxx_noem_exp] : s_upcxx_em[best_upcxx_em_exp];
  
    if (TuneMode) {
      printf("cost=%d\n", cost);
    }
    printf("Best T_UPCxx_Dsp=%.9lf for Grain=%6d Tasks_rank=%6d Subtasks=%2d\n", *best_upcxx_time_it, grains[best_upcxx_exp], max_grain/grains[best_upcxx_exp], best_upcxx_subtasks);
    printf("Best        T_MT=%.9lf for Grain=%6d Tasks_rank=%6d Subtasks=%2d\n", *best_mt_time_it, grains[best_mt_exp], max_grain/grains[best_mt_exp], s_mt[best_mt_exp]);
  }
  
  delete [] ptr;
}

int process_arguments(int argc, char **argv)
{ int c;

  upcxx::init();

  Myrank = upcxx::rank_me();
  Nranks = upcxx::rank_n();

  while ( -1 != (c = getopt(argc, argv, "c:fhlPq:Tt:x")) ) {
    switch (c) {

      case 'f':
        Filtered = true;
        break;

      case 'l':
        LocalMeasurement = true;
        break;

      case 'T':
        TuneMode = true;
        break;

      case 'P':
        TryPar = true;
        break;

      case 'q' : /* queue limit */
        queue_limit = atoi(optarg);
        break;

      case 't': /* threads */
	nthreads = atoi(optarg);
	break;

      case 'c': /* cost */
	cost = atoi(optarg);
	break;

      case 'x':
        Tricky = true;
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
  
  if (nsize % Nranks) {
    if (!Myrank) printf("%d elems not divisible by %d ranks\n", nsize, Nranks);
    exit(EXIT_FAILURE);
  }

  max_grain = nsize / Nranks;
  upcxx::barrier();
  
  return 0;
}


int main(int argc, char **argv)
{ int begin_cost, end_cost;

  if(process_arguments(argc, argv) == -1)
    return -1;
	
  depspawn::set_threads(nthreads);

  if (queue_limit >= 0) {
    if (!Myrank) printf("Setting queue limit to %d\n", queue_limit);
    depspawn::set_task_queue_limit(queue_limit);
  }
  
  if (!Myrank) {
    depspawn::print_upcxx_depspawn_runtime_setup();
  }

  enable_ps_master();
  
  mx.init(nsize, max_grain);
  
  if (TuneMode) {
    begin_cost = 1;
    if (!Myrank) {
      printf("Tune mode cost=%d to %d\n", begin_cost, cost);
    }
  } else {
    begin_cost = cost;
  }
  end_cost = cost;

  for (cost = begin_cost; cost <= end_cost; cost = cost * 2) {
    t0 = std::chrono::high_resolution_clock::now();
    serial_value = task(InitValue);
    t2 = std::chrono::high_resolution_clock::now();
    serial_time = nsize * std::chrono::duration <double>(t2 - t0).count();
    
    if (!Myrank) {
      printf("--------------------------\nnsize=%d Nranks=%d threads=%d max_grain=%d cost=%d local=%c serial_time=%.9lf Tricky=%c Filtered=%c\n", nsize, Nranks, nthreads, max_grain, cost, LocalMeasurement ? 'Y' : 'N', serial_time, Tricky ? 'Y' : 'N', Filtered ? 'Y' : 'N');
      //printf("serial_value=%lf\n", serial_value);
    }
    
    init_mx();

    test0();
  }
  
  return retstate;
}
