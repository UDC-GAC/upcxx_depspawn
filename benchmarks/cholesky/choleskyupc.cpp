/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cctype>
#include <sys/types.h>
#include "upcxx_depspawn/upcxx_depspawn.h"

#include "tile.h"

//#ifdef _OPENMP
//#include <omp.h>
//#endif

//#ifdef USE_MKL
//#include "mkl.h"
//#endif
extern bool BinaryFiles;

int Row_cyc, Col_cyc, NReps;
bool MidProfiling = false;
char Algorithm = 'D';
upcxx::persona_scope *ps_master = nullptr;

extern void cholesky_seq(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname);
extern void cholesky(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname);
extern void cholesky_barriered(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname);
extern void cholesky_filtered(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname);

// Generate matrix
extern void matrix_init(double *A, int n);

// Read the matrix from the input file
extern void matrix_init(DistributedTiled2DMatrix<tile>& A, int in_n, const char *fname);

// write matrix to file
extern void matrix_write(double *A, int n, const char *fname);

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

void report_time(const char *mode, int ntest, tile &a, double time)
{
  std::cout << mode << " test " << ntest << " Total Time: " << time << " sec" << std::endl;
  //float Gflops = ((float)2*a.size()*a.size()*a.size())/((float)1000000000);
  //if (Gflops >= .000001) printf("Floating-point operations executed: %f billion\n", Gflops);
  //if (time >= .001) printf("Floating-point operations executed per unit time: %6.2f billions/sec\n", Gflops/time);
}

void config_test(const int value, const char *msg)
{
  const int test = upcxx::reduce_all(value, upcxx::op_fast_add).wait();
  if (test != (value * upcxx::rank_n())) {
    if (!upcxx::rank_me()) {
      fprintf(stderr, "No match on %s\n", msg);
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[])
{
  
  //#ifdef USE_MKL
  //    mkl_set_num_threads(1);
  //#else
  //openblas_set_num_threads( 1 );
  //#endif
  //   mkl_set_num_threads(1);
  
  upcxx::init();
  
  //#ifdef _OPENMP
  //omp_set_num_threads(nthreads);
  //#endif
  
  const uint32_t imyrank = upcxx::rank_me();
  Row_cyc = upcxx::rank_n();
  Col_cyc = 1;
  NReps = 1;
  
  int n;
  const int b = TILESIZE;
  const char *fname = NULL;
  const char *oname = NULL;
  const char *mname = NULL;
  int argi;
  
  // Command line: cholesky n b filename [out-file]
  if (argc < 2 || argc > 15) {
    if(!imyrank) {
      fprintf(stderr, "Incorrect number of arguments, expected N [-b] [-P] [-a alg] [-r row_cyc] [-c col_cyc] [-n reps] [-i infile] [-o outfile] [-w mfile]\n");
      fprintf(stderr, "-b: binary files\n-P: profiling before wait\n");
      fprintf(stderr, "-a alg: Use algorithm [D]=DepSpawn [B]=Barriers [F]=Filtered DepSpawn [S]=seq\n");
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  argi = 1;
  n = atol(argv[argi++]);
  
  if(!n) {
    if(!imyrank) {
      fprintf(stderr, "input N=0!\n");
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  //DIEGO: The tile size is specified in definition of TILESIZE in tile.h
  //    b = atol(argv[argi++]);
  
  while (argi < argc) {
    if (!strcmp(argv[argi], "-o"))
      oname = argv[++argi];
    else if (!strcmp(argv[argi], "-i"))
      fname = argv[++argi];
    else if (!strcmp(argv[argi], "-w"))
      mname = argv[++argi];
    else if (!strcmp(argv[argi], "-r"))
      Row_cyc = static_cast<int>(strtol(argv[++argi], (char **)NULL, 10));
    else if (!strcmp(argv[argi], "-c"))
      Col_cyc = static_cast<int>(strtol(argv[++argi], (char **)NULL, 10));
    else if (!strcmp(argv[argi], "-n"))
      NReps = static_cast<int>(strtol(argv[++argi], (char **)NULL, 10));
    else if (!strcmp(argv[argi], "-b"))
      BinaryFiles = true;
    else if (!strcmp(argv[argi], "-P"))
      MidProfiling = true;
    else if (!strcmp(argv[argi], "-a"))
      Algorithm = toupper(argv[++argi][0]);
    else {
       if (!imyrank) {
         fprintf(stderr, "Skipping unknown argument %s\n", argv[argi]);
       }
    }
    ++argi;
  }
  
  const char * const nthreads_env = getenv("UD_NUM_THREADS");
  const int nthreads = (nthreads_env == NULL) ? -1 : static_cast<int>(strtol(nthreads_env, (char **)NULL, 10));

  depspawn::set_threads(nthreads);
  depspawn::upcxx_depspawn_runtime_setup(); // Only so that the get_UPCXX_DEPSPAWN_* functions get right values
  enable_ps_master();  // The setup disables the master_persona

  const int exact_match_on = (int)depspawn::get_UPCXX_DEPSPAWN_EXACT_MATCH();
  const int prefetch_on = (int)depspawn::get_UPCXX_DEPSPAWN_PREFETCH();
  const int yield_on = (int)depspawn::get_UPCXX_DEPSPAWN_YIELD();
  const int active_wait_on = (int)depspawn::get_UPCXX_DEPSPAWN_ACTIVE_WAIT();
  
  if (!imyrank) {
    printf("nthreads=%d n=%d Row_cyc=%d Col_cyc=%d TILESIZE=%d alg=%c NReps=%d ExactMatch=%d Prefetch=%d ActWait=%d Yield=%d\n", nthreads, n, Row_cyc, Col_cyc, TILESIZE, Algorithm, NReps, exact_match_on, prefetch_on, active_wait_on, yield_on);
    fflush(stdout);
  }
  
  if ( (Row_cyc * Col_cyc) != upcxx::rank_n() ) {
    if (!imyrank) {
      fprintf(stderr, "(Row_cyc * Col_cyc) != upcxx::rank_n()\n");
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  //#ifdef _OPENMP
  //    omp_set_num_threads(nthreads);
  //#endif
  
  /* Managed by DEPSPAWN_TASK_QUEUE_LIMIT
  char *qlimit_env = getenv("DSP_QLIMIT");
  if (qlimit_env != NULL)
  {
    int l = static_cast<int>(strtol(qlimit_env, (char **)NULL, 10));
    printf("Queue limit=%d\n", l);
    depspawn::set_task_queue_limit(l);
  }
  */

  if (n % TILESIZE != 0) {
    if (!imyrank) {
      fprintf(stderr, "The tile size is not compatible with the given matrix\n");
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  config_test(exact_match_on, "UPCXX_DEPSPAWN_EXACT_MATCH");
  config_test(prefetch_on, "UPCXX_DEPSPAWN_PREFETCH");
  config_test(yield_on, "UPCXX_DEPSPAWN_YIELD");
  config_test(active_wait_on, "UPCXX_DEPSPAWN_ACTIVE_WAIT");
  
  upcxx::barrier(); // For the sake of I/O

  if (mname) {
    if (!imyrank) {
      double * const AA = new double[static_cast<size_t>(n) * static_cast<size_t>(n)];
      if( AA == nullptr) {
        fprintf(stderr, "Not enough memory for matrix %d x %d\n", n, n);
        upcxx::finalize();
        exit(EXIT_FAILURE);
      }
      matrix_init(AA, n);
      matrix_write(AA, n, mname);
      delete [] AA;
    }
  } else {
    DistributedTiled2DMatrix<tile> AA(n, n, Row_cyc, Col_cyc, imyrank, MatrixType::LowerTriangular);
    matrix_init(AA, n, fname);
    switch (Algorithm) {
      case 'B':
        cholesky_barriered(AA, n, b, oname);
        break;
      case 'D':
        cholesky(AA, n, b, oname);
        break;
      case 'F':
        cholesky_filtered(AA, n, b, oname);
        break;
      case 'S':
        cholesky_seq(AA, n, b,  oname);
        break;
      default:
        fprintf(stderr, "Uknown algorithm %c\n", Algorithm);
        break;
    }
  }

  upcxx::finalize();
  
  return 0;
}
