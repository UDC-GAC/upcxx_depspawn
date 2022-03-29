/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     bench_sched_perf.cpp
/// \brief    Tests scheduling performance in a single process
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

/* Needs a shared heap of 1300MB: upcxx-run -n 1 -shared-heap 1300MB ./bench_sched_perf */

#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"

using namespace depspawn;

constexpr size_t MAX_TILE_SZ = 100;
constexpr size_t MAX_NTASKS = (1 << 14);

struct Tile {
  double data_[MAX_TILE_SZ][MAX_TILE_SZ];
  
  void clear() {
    memset((void *)data_, 0, sizeof(data_));
  }
};

int Nthreads = 4;
int NReps = 1;
size_t MaxTileSize = MAX_TILE_SZ;
size_t MaxNTasks = MAX_NTASKS;
size_t cur_tile_sz;
bool UseSameTile = false;
bool ClearCaches = false;
bool OnlyParallelRun = false; // This is for profiling
bool ParallelBaseline = false;
int Queue_limit = -1;
int Verbosity = 0;
SharedArray<Tile> Input1, Input2, Destination;
std::vector<double> TSeq, TPar;

void clearDestination()
{
  for (size_t i = 0; i < MAX_NTASKS; i++) {
    Destination[i].local()->clear();
  }
}

// I tried working on a local tmp tile instead of on dest.data in order to avoid
//the extra memory cost of the remote access in NUMA machines. But the compiler
//detected that the output was not written and skipped the computations.
// input1 and input2 should be fine because they are a single read-only tile.
void seq_mult(Tile& __restrict__ dest, const Tile& __restrict__ input1, const Tile& __restrict__ input2)
{
  for (size_t i = 0; i < cur_tile_sz; i++) {
    for (size_t j = 0; j < cur_tile_sz; j++) {
      for (size_t k = 0; k < cur_tile_sz; k++) {
        dest.data_[i][j] += input1.data_[i][k] * input2.data_[k][j];
      }
    }
  }
}

void upcxx_mult(upcxx::global_ptr<Tile> dest, const Tile& input1, const Tile& input2)
{
  seq_mult(*(dest.local()), input1, input2);
}

void upcxx_mult_opt(upcxx::global_ptr<Tile> dest, upcxx::global_ptr<const Tile> input1, upcxx::global_ptr<const Tile> input2)
{
  seq_mult(*(dest.local()), *(input1.local()), *(input2.local()));
}

double bench_serial_time(const size_t ntasks, int nreps)
{ double ret_time = 0.;

  //upcxx::barrier();

  for (int nr = 0; nr < nreps; nr++) {
    
    if (ClearCaches) {
      clearDestination();
    }
    
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    if (ParallelBaseline) {
      const auto task =[&](size_t i) { seq_mult(*(Destination[UseSameTile ? 0 : i].local()), *(Input1[0].local()), *(Input2[0].local())); };
      get_task_pool().parallel_for((size_t)0, ntasks, (size_t)1, task, false);
//      TP->launch_threads();
//      for (size_t i = 0; i < ntasks; i++) {
//        TP->enqueue([&, i] { seq_mult(*(Destination[UseSameTile ? 0 : i].local()), *(Input1[0].local()), *(Input2[0].local())); });
//      }
//      TP->wait(false);
    } else {
      for (size_t i = 0; i < ntasks; i++) {
        seq_mult(*(Destination[UseSameTile ? 0 : i].local()), *(Input1[0].local()), *(Input2[0].local()));
      }
    }

    TSeq[nr] =  std::chrono::duration <double>(std::chrono::high_resolution_clock::now() - t0).count();
    ret_time += TSeq[nr];
  }
  
  return ret_time;
}

double bench_parallel_time(const size_t ntasks, int nreps)
{ double ret_time = 0.;

  //upcxx::barrier();

  for (int nr = 0; nr < nreps; nr++) {
    
    if (ClearCaches) {
      clearDestination();
    }
    
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < ntasks; i++) {
      upcxx_spawn(upcxx_mult_opt, Destination[UseSameTile ? 0 : i], Input1[0], Input2[0]);
    }
    upcxx_wait_for_all();
    
    TPar[nr] =  std::chrono::duration <double>(std::chrono::high_resolution_clock::now() - t0).count();
    ret_time += TPar[nr];
  }

  return ret_time;
}

double bench_sched_perf(const size_t ntasks)
{ double serial_time = 0.;
  
  if (!OnlyParallelRun) {
    
    // heat the caches if they are not going to be cleaned.
    //It is not done in profiling runs because of the runtime impact
    if (!ClearCaches) {
      bench_parallel_time(ntasks, 1);
    }
    
    serial_time = bench_serial_time(ntasks, NReps);
    if(Verbosity) {
      if (Verbosity > 1) {
        for (double t : TSeq) {
          std::cout << std::setw(8) << t << ' ';
        }
      }
      std::cout << "  serial time=" << serial_time << std::endl;
    }
  }

  const double parallel_time = bench_parallel_time(ntasks, NReps);
  if(Verbosity) {
    if (Verbosity > 1) {
      for (double t : TPar) {
        std::cout << std::setw(8) << t << ' ';
      }
    }
    std::cout << "parallel time=" << parallel_time << std::endl;
  }
  
  if (OnlyParallelRun) {
    return 1.0;
  } else {
    // if UseSameTile the runtime cannot scale linearly, and if ParallelBaseline we just want a one-to-one comparison
    //So only in sequential baselines with different tiles do we compare with the theoretical Time/Nthreads scaling
    const double baseline_time = (UseSameTile || ParallelBaseline) ? serial_time : (serial_time / Nthreads);
    return parallel_time / baseline_time;
  }
}

void show_help()
{
  puts("bench_sched_perf [-c] [-h] [-n reps] [-p] [-P] [-q limit] [-s] [-t nthreads] [-T maxtilesize] [-v level]");
  puts("-c          clear caches before test");
  puts("-h          Display help and exit");
  puts("-N ntasks   Maximum number of tasks");
  puts("-n reps     Repeat each measurement reps times");
  puts("-p          Parallel baseline");
  puts("-P          Only depspawn parallel run (for profiling)");
  puts("-q limit    queue limit");
  puts("-s          Work always on same tile (no parallelism)");
  puts("-T tilesz   Maximum tile size");
  puts("-t nthreads Run with nthreads threads. Default is 4");
  puts("-v level    Verbosity level");
}

void process_arguments(int argc, char **argv)
{ int c;

  while ( -1 != (c = getopt(argc, argv, "chN:n:Ppq:sT:t:v:")) ) {
    switch (c) {
      case 'c':
        ClearCaches = true;
        break;
      case 'h':
        show_help();
        //upcxx::init();
        //upcxx::finalize();
        exit(EXIT_SUCCESS);
        break;
      case 'N':
        c = strtoul(optarg, 0, 0);
        if (c < MAX_NTASKS) {
          MaxNTasks = c;
        }
        break;
      case 'n':
        NReps = strtoul(optarg, 0, 0);
        break;
      case 'P':
        OnlyParallelRun = true;
        break;
      case 'p':
        ParallelBaseline = true;
        break;
      case 'q' : /* queue limit */
        Queue_limit = strtoul(optarg, 0, 0);
        break;
      case 's':
        UseSameTile = true;
        break;
      case 'T':
        MaxTileSize = std::min(strtoul(optarg, 0, 0), MAX_NTASKS);
        break;
      case 't':
        Nthreads = strtoul(optarg, 0, 0);
        break;
      case 'v':
        Verbosity = strtoul(optarg, 0, 0);
        break;
      default:
        fprintf(stderr, "Unknown argument %c. Try -h\n", (char)c);
    }
  }
}

int main(int argc, char **argv)
{
  process_arguments(argc, argv);
  
  set_threads(Nthreads);
  
  if (Queue_limit >= 0) {
    set_task_queue_limit(Queue_limit);
  }
  
  TSeq.resize(NReps);
  TPar.resize(NReps);

  upcxx::init();
  
  const int myrank = upcxx::rank_me();

  if(upcxx::rank_n() != 1) {
    if (!myrank) {
      std::cout << "\n  This benchmark must be run with a single process\n\n";
    }
    exit(EXIT_FAILURE);
  }

  assert(upcxx::master_persona().active_with_caller());

  Input1.init(1);
  Input2.init(1);
  Destination.init(MAX_NTASKS, MAX_NTASKS);

  std::cout << upcxx::rank_n() << " Procs x " << Nthreads << " threads max_tile_size=" << MaxTileSize << " NReps=" << NReps << " ClearCache=" << (ClearCaches ? 'Y' : 'N') << " QueueLimit=" << Queue_limit << " EnqueueTasks=" << (depspawn::internal::EnqueueTasks ? 'Y' : 'N') << " Baseline=" << (ParallelBaseline ? "Parallel" : "Sequential") << std::endl;
  
  print_upcxx_depspawn_runtime_setup();
  
  if(!OnlyParallelRun) {
    const auto task = [](size_t i) { Destination[i].local()->clear(); };

    get_task_pool().parallel_for((size_t)0, MAX_NTASKS, (size_t)1, task, false);
//    TP->launch_threads();
//    for (size_t i = 0; i < MAX_NTASKS; i++) {
//      TP->enqueue(task, i);
//    }
//    TP->wait(false);
    
    //1-time runtime preheat (for memory pools)
    bench_parallel_time(MaxNTasks, 1);
  }
  
  //upcxx::barrier();
  
  std::cout << "size %";
  for (size_t ntasks = (1 << 7); ntasks <= MaxNTasks; ntasks *= 2) {
    std::cout << std::setw(5) << ntasks << "    ";
  }
  std::cout << "tasks\n";
  
  for (size_t tile_sz=10; tile_sz <= MaxTileSize; tile_sz += 5) {
    if (Verbosity) {
      std::cout << "Tests for TSZ=";
    }
    std::cout << std::right << std::setw(3) << tile_sz << ' ' << std::left;
    if (Verbosity) {
      std::cout << std::endl;
    }
    cur_tile_sz = tile_sz;
    for (size_t ntasks = (1 << 7); ntasks <= MaxNTasks; ntasks *= 2) {
      const double normalized_par_to_seq_ratio = bench_sched_perf(ntasks);
      if (Verbosity) {
        std::cout << "TSZ=" << tile_sz << " NTASKS=" << ntasks << " r=";
      }
      std::cout << std::setw(8) << normalized_par_to_seq_ratio << ' ';
      if (Verbosity) {
        std::cout << std::endl;
      }
    }
    //std::cout << '%' << tile_sz;
    std::cout << std::endl;
  }

  upcxx::persona_scope scope_e(upcxx::master_persona());

  upcxx::finalize();
  
  return 0;
}

