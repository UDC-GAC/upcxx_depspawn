/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     test_central.cpp
/// \brief    Tests performance of despawn vs a centralized solution.
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

//#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"
#include "common_io.cpp"  // This is only for serializing parallel prints

using namespace depspawn;

//constexpr size_t MAX_TILE_SZ = 200000;

int Myrank, Nranks;
int NTiles, NReps = 1, NThreads = 4;
size_t FlopsPerElem = 64;
double alpha = 2.f;
bool Do_Seq_Test = false;
bool Do_Return = false;
std::atomic<int> Ntasks_mine;
std::chrono::high_resolution_clock::time_point t0, t1;
upcxx::persona_scope *ps_master = nullptr;

template<typename F> void enqueue_task(F&& t) { get_task_pool().enqueue(std::forward<F>(t)); };

SharedArray<int> Signals;

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

void deep_comp(double * const yp, const double * const xp, const double alpha, const int MAX_TILE_SZ)
{
  if (FlopsPerElem == 2) {
    for (int j = 0; j < MAX_TILE_SZ; j++) {
      yp[j] = alpha * xp[j] + yp[j];
    }
  } else {
    for (int j = 0; j < MAX_TILE_SZ; j++) {
      for (size_t r = 0; r < (FlopsPerElem /2); r++) {
        yp[j] = alpha * xp[j] + yp[j];
      }
    }
  }
}


template<size_t MAX_TILE_SZ>
struct Tile {

  double data_[MAX_TILE_SZ];
  
  void clear() {
    memset((void *)data_, 0, sizeof(data_));
  }
  
  double  operator[] (size_t i) const { return data_[i]; }
  double& operator[] (size_t i)       { return data_[i]; }

};

//template<size_t MAX_TILE_SZ>
//bool operator==(const Tile& lhs, const Tile& rhs)
//{ int i;
//
//  for(i = 0; (i < MAX_TILE_SZ) && (lhs[i] == rhs[i]); i++);
//
//  return (i == MAX_TILE_SZ);
//}

template<size_t MAX_TILE_SZ>
void local_computation(Tile<MAX_TILE_SZ>& y, const Tile<MAX_TILE_SZ>& x, double alpha)
{
  const double * const xp = x.data_;
  double * const yp = y.data_;
  deep_comp(yp, xp, alpha, MAX_TILE_SZ);
}

template<size_t MAX_TILE_SZ>
void computation(upcxx::global_ptr<Tile<MAX_TILE_SZ>> y, upcxx::global_ptr<const Tile<MAX_TILE_SZ>> x, double alpha)
{
  const double * const xp = x.local()->data_;
  double * const yp = y.local()->data_;
  deep_comp(yp, xp, alpha, MAX_TILE_SZ);
}

template<size_t MAX_TILE_SZ>
void enqueue_computation(upcxx::global_ptr<Tile<MAX_TILE_SZ>> y, upcxx::global_ptr<const Tile<MAX_TILE_SZ>> x, double alpha)
{
  enqueue_task([y,x,alpha]() {
    auto yt = upcxx::new_<Tile<MAX_TILE_SZ>>();
    auto xt = upcxx::new_<Tile<MAX_TILE_SZ>>();
    auto yf = upcxx::rget(y, yt.local(), 1);
    auto xf = upcxx::rget(x, xt.local(), 1);
    yf.wait();
    xf.wait();
    local_computation<MAX_TILE_SZ>(*yt.local(), *xt.local(), alpha);
    if (Do_Return) {
      upcxx::rput(yt.local(), y, 1).wait(); // return result
    }
    upcxx::delete_<Tile<MAX_TILE_SZ>>(yt);
    upcxx::delete_<Tile<MAX_TILE_SZ>>(xt);
  });

  Ntasks_mine--;
}

template<size_t MAX_TILE_SZ>
struct CentralTest {
  
  using Tile_t = Tile<MAX_TILE_SZ>;

  static SharedArray<Tile_t> X, Y, X1, Y1;

  void parallel_init(SharedArray<Tile_t>& X, SharedArray<Tile_t>& Y)
  {
    upcxx::barrier();
    
    for (int i = 0; i < NTiles; i++) {
      if (X[i].where() == Myrank) {
        double * const xp = X[i].local()->data_;
        double * const yp = Y[i].local()->data_;
        for (int j = 0; j < MAX_TILE_SZ; j++) {
          yp[j] = 1.f;
          xp[j] = (double)(i * MAX_TILE_SZ + j);
        }
      }
    }
    
    upcxx::barrier();
  }

  bool equal_array_contents(SharedArray<Tile_t>& V, SharedArray<Tile_t>& V2)
  { Tile_t t1, t2;
  
    for (int i = 0; i < NTiles; i++) {
      auto vf = upcxx::rget(V[i], &t1, 1);
      auto v2f = upcxx::rget(V2[i], &t2, 1);
      vf.wait();
      v2f.wait();
      if (memcmp(&t1, &t2, sizeof(Tile_t))) {
        return false;
      }
    }
    return true;
  }
  
  bool dotest()
  { std::vector<double> t_seq, t_dsp, t_central;
    bool test_ok = true;
    
    X.init(NTiles);
    Y.init(NTiles);
    X1.init(NTiles, NTiles);
    Y1.init(NTiles, NTiles);
    
    parallel_init(X, Y);
    parallel_init(X1, Y1);
    
    // Sequential run
    /////////////////////////////////////
    if (!Myrank) {
      std::cout << "Test for tiles of " << MAX_TILE_SZ << '\n';
      
      for(int nr = 0; nr < NReps; nr++) {
        if(Do_Seq_Test) {
          t0 = std::chrono::high_resolution_clock::now();
          for (int i = 0; i < NTiles; i++) {
            computation<MAX_TILE_SZ>(Y1[i], X1[i], alpha);
          }
          t_seq.push_back(std::chrono::duration <double>(std::chrono::high_resolution_clock::now() - t0).count());
        } else {
            t_seq.push_back(0.);
        }
      }

      for(double t: t_seq) {
        std::cout << "Sequential  time: " << t << '\n';
      }
    }
    
    // Depspawn run
    /////////////////////////////////////
    for(int nr = 0; nr < NReps; nr++) {
      upcxx::barrier();
    
      disable_ps_master();
      t0 = std::chrono::high_resolution_clock::now();
    
      for (int i = 0; i < NTiles; i++) {
        upcxx_spawn(computation<MAX_TILE_SZ>, Y[i], X[i], alpha);
      }
    
      upcxx_wait_for_all();
      t_dsp.push_back(std::chrono::duration <double>(std::chrono::high_resolution_clock::now() - t0).count());
      enable_ps_master();
    }
    
    if (!Myrank) {

      for(double t: t_dsp) {
        std::cout << "Depspawn    time: " << t << '\n';
      }

      if(Do_Seq_Test) {
        test_ok = equal_array_contents(Y, Y1);
        if (!test_ok) {
          std::cout << "Depspawn test FAILED!\n";
          exit(-1);
        }
      } else {
        std::cout << "Depspawn test skipped (no sequential run)\n";
      }
    }
    
    parallel_init(X1, Y1);
    
    // centralized run
    /////////////////////////////////////

    auto& UPCxx_DepSpawn_Task_Pool = get_task_pool();

    for(int nr = 0; nr < NReps; nr++) {

      UPCxx_DepSpawn_Task_Pool.launch_threads();
      
      upcxx::rput(0, Signals[Myrank]).wait();
      Ntasks_mine = 0;

      upcxx::barrier();
      
      t0 = std::chrono::high_resolution_clock::now();
    
      for (int i = 0; i < NTiles; i++) {
        const int dest = i % Nranks;
        
        if(dest == Myrank) {
          Ntasks_mine++;
        }

        if (!Myrank) {
          if(dest) { // remote
            upcxx::rpc_ff(dest, enqueue_computation<MAX_TILE_SZ>, Y1[i], X1[i], alpha);
          } else { //local
            enqueue_task([i]() { computation<MAX_TILE_SZ>(Y1[i], X1[i], alpha); } );
          }
        }
          
      }
      
      if (!Myrank) {
        auto f = upcxx::rput(1, Signals[Myrank]);
        for (int i = 1; i < Nranks; i++) {
          f = upcxx::when_all(f, upcxx::rput(1, Signals[i]));
        }
        f.wait();
      } else {
        while( !upcxx::rget(Signals[Myrank]).wait() || Ntasks_mine.load() ) { upcxx::progress(); }
      }

      UPCxx_DepSpawn_Task_Pool.wait(false);

      upcxx::barrier();
    
      t_central.push_back(std::chrono::duration <double>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    
    if (!Myrank) {

      for(double t: t_central) {
        std::cout << "Centralized time: " << t << '\n';
      }

      if (Do_Return) {
        test_ok = equal_array_contents(Y, Y1);
        if (!test_ok) {
          std::cout << "Centralized test FAILED!\n";
          exit(-1);
        }
      }
      else {
        std::cout << "Centralized test skipped (no return)\n";
      }
    }
    
    return test_ok;
  }

};

template<size_t MAX_TILE_SZ>
SharedArray<Tile<MAX_TILE_SZ>> CentralTest<MAX_TILE_SZ>::X;

template<size_t MAX_TILE_SZ>
SharedArray<Tile<MAX_TILE_SZ>> CentralTest<MAX_TILE_SZ>::Y;

template<size_t MAX_TILE_SZ>
SharedArray<Tile<MAX_TILE_SZ>> CentralTest<MAX_TILE_SZ>::X1;

template<size_t MAX_TILE_SZ>
SharedArray<Tile<MAX_TILE_SZ>> CentralTest<MAX_TILE_SZ>::Y1;

void parse_command_line(int argc, char **argv)
{ char c;
  
  while ((c = getopt (argc, argv, "ht:F:R:rs")) != -1)
    switch (c) {
      case 'h':
        if (!Myrank) {
          printf("   test_central [-h] [-t threads] [-F flops] [-r] [-R reps] [-s]\n"
                 "\n"
                 "Options:\n"
                 "   -h         Print this help message.\n"
                 "   -t threads Number of thtreads (default: %d)\n"
                 "   -F flops   Number of flops per element (default: %zu)\n"
                 "   -R reps    Number of repetitions per experiment\n"
                 "   -r         Return the result to process 0\n"
                 "   -s         Perform sequential test"
                 "\n", NThreads, FlopsPerElem);
        }
        upcxx::finalize();
        exit (0);
      case 't':
        NThreads = (int)strtoul(optarg, NULL, 0);
        break;
      case 'F':
        FlopsPerElem = (int)strtoul(optarg, NULL, 0);
        break;
      case 'r':
        Do_Return = true;
        break;
      case 'R':
        NReps = (int)strtoul(optarg, NULL, 0);
        break;
      case 's':
        Do_Seq_Test = true;
        break;
      default:
        fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        upcxx::finalize();
        exit(1);
    }
  
  if (!Myrank) {
    std::cout << "Using " << Nranks << " Procs x " << NThreads << " threads Flops/elem: " << FlopsPerElem << " NReps: " << NReps << '\n';
  }
}

int main(int argc, char **argv)
{
  
  upcxx::init();

  Myrank = upcxx::rank_me();
  Nranks = upcxx::rank_n();
  NTiles = NThreads * Nranks;
  
  parse_command_line(argc, argv);
  
  set_threads(NThreads);
  
  Signals.init(NTiles);
  
  const bool test_ok =
  CentralTest<1024>().dotest()
  && CentralTest<2048>().dotest()
  && CentralTest<4096>().dotest()
  && CentralTest<8192>().dotest()
  && CentralTest<16384>().dotest()
  && CentralTest<32768>().dotest()
  && CentralTest<65536>().dotest()
  && CentralTest<131072>().dotest();
  //&& CentralTest<262144>().dotest()
  //&& CentralTest<524288>().dotest()
  
  upcxx::finalize();

  if (!Myrank) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  return !test_ok;
}
