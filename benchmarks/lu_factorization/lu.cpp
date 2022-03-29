/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     lu.cpp
/// \brief    Main implementation file for LU
/// \author   Diego Andrade       <diego.andrade@udc.es>
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>
///


#include <cstdlib>
#include <cstdio>
#include <limits.h>
#include <cstring>
#include <cmath>
#include <cctype>
#include "tile.h"
#include "DistrBlockMatrix.h"
#include "upcxx_depspawn/upcxx_depspawn.h"

int imyrank;
int Row_cyc, Col_cyc, NReps;
const char *Outfile = nullptr;
char Algorithm = 'D';
upcxx::persona_scope *ps_master = nullptr;

extern int lu_factorization_seq(DistrBlockMatrix<tile>&, const int);
extern int lu_factorization(DistrBlockMatrix<tile> &, int * const *, int, int);
extern int lu_factorization_filtered(DistrBlockMatrix<tile> &, int * const *, int, int);

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

void task11(tile &sub11)
{
  int z;
  const int ld=TILESIZE;
  const int nx=TILESIZE;
  
  for (z = 0; z < nx; z++)
  {
    double pivot;
    pivot = sub11.get(z,z);
    //STARPU_ASSERT(pivot != 0.0);
    
    //CPU_SCAL(nx - z - 1, (1.0/pivot), &sub11[z+(z+1)*ld], ld);
    int nxz1 = nx-z-1;
    double pivot1  = 1.0/pivot;
    double *posz1z = &(sub11.m_tile[((z+1)*TILESIZE)+z]);
    cblas_dscal(nxz1, pivot1, posz1z, ld);
    /*
     CPU_GER(nx - z - 1, nx - z - 1, -1.0,
     &sub11[(z+1)+z*ld], 1,
     &sub11[z+(z+1)*ld], ld,
     &sub11[(z+1) + (z+1)*ld],ld);
     */
    
    double *poszz1  = &(sub11.m_tile[(z*TILESIZE)+z+1]);
    double *posz1z1 = &(sub11.m_tile[((z+1)*TILESIZE)+z+1]);
    cblas_dger(CblasColMajor,nxz1, nxz1, -1.0,
               poszz1, 1,posz1z
               , ld,
               posz1z1,ld);
    
  }
  
}

void task12(const tile &sub11, tile &sub12)
{
  int ld11,ld12,nx12,ny12;
  
  ld11=ld12=nx12=ny12=TILESIZE;
  
  //cblas_dtrsm("L", "L", "N", "N", nx12, ny12,(double)1.0, (double *) sub11.m_tile, ld11, sub12.m_tile, ld12);
  cblas_dtrsm(CblasColMajor,CblasLeft, CblasLower, CblasNoTrans,CblasNonUnit,nx12, ny12,(double)1.0, (double *) sub11.m_tile, ld11, sub12.m_tile, ld12);
  
}

void task21(const tile &sub11, tile &sub21)
{
  int ld11,ld21,nx21,ny21;
  
  ld11=ld21=nx21=ny21=TILESIZE;
  
  //cblas_dtrsm("R", "U", "N", "U", nx21, ny21,(double)1.0, (double *) sub11.m_tile, ld11, sub21.m_tile, ld21);
  cblas_dtrsm(CblasColMajor,CblasRight, CblasUpper, CblasNoTrans, CblasUnit, nx21, ny21,(double)1.0, (double *) sub11.m_tile, ld11, sub21.m_tile, ld21);
  
}

void task22(const tile &right, const tile &left, tile &center)
{
  int dx,dy,dz,ld12,ld21,ld22;
  dx=dy=dz=ld12=ld21=ld22=TILESIZE;
  
  //cblas_dgemm("N","N", dy, dx, dz,  (double)-1.0, (double *) right.m_tile, ld21, (double *) left.m_tile, ld12,  (double)1.0, center.m_tile, ld22);
  cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, dy, dx, dz,  (double)-1.0, (double *) right.m_tile, ld21, (double *) left.m_tile, ld12,  (double)1.0, center.m_tile, ld22);
}

void task11(upcxx::global_ptr<tile> a)
{
  task11(*a.local());
}

void task12(upcxx::cached_global_ptr<const tile> a1, upcxx::global_ptr<tile> a2)
{
  task12(*a1.local(), *a2.local());
}

void task21(upcxx::cached_global_ptr<const tile> a1, upcxx::global_ptr<tile> a2)
{
  task21(*a1.local(), *a2.local());
}

void task22(upcxx::cached_global_ptr<const tile> a1, upcxx::cached_global_ptr<const tile> a2, upcxx::global_ptr<tile> a3)
{
  task22(*a1.local(), *a2.local(), *a3.local());
}

/// Must be run by all the processes
int init(DistrBlockMatrix<tile> *A, double * const localA, int * const P, int m_size, int m_dim)
{
  /*
   m_size = dimension;
   m_dim = (m_size + TILESIZE - 1)/TILESIZE; // Size/TILESIZE rounded up
   m_tiles = new tile[m_dim*m_dim];
   */
  
  int dim = m_dim;
  int size = m_size;

  //init the pivot array
  //for(int i = 1; i <= size; ++i)
  // *P[i] = i;

  srand(1234);

  int ii = 0;
  double e;
  for (int I = 0; I < dim; I++)
  {
    for (int i = 0; i < TILESIZE; i++)
    {
      int jj = 0;
      for (int J = 0; J < dim; J++)
      {
        auto bck = (*A)(I, J);

        for (int j = 0; j < TILESIZE; j++)
        {
          // Everyone calls all the rand() so that the result is == a sequential initialization
          e = (double)(rand()%999999+999999)/(999999*2);
          /*if ((ii < size)&(jj < size)) e = (double)(rand()%999999+999999)/(999999*2);
          else if (ii == jj) e = 1; // On-diagonal padding
          else e = 0; // Off-diagonal padding*/
          
          if (localA != nullptr) {
            localA[(I*TILESIZE*TILESIZE*dim)+(J*TILESIZE*TILESIZE)+(i*TILESIZE)+j] = e;
          }
          
          if (bck.where() == imyrank) {
            bck.local()->set(i,j,e);
          }
          jj++;
        }
      }
      ii++;
    }
  }
  
  // This is to ensure it was (1) run in SPMD and (2) initialized everywhere
  upcxx::barrier();

  return m_dim;
}

void copy_distr_matrix(DistrBlockMatrix<tile> *in, double **out, int ntiles)
{
  double e;
  for (int I = 0; I < ntiles; I++)
  {
    for (int i = 0; i < TILESIZE; i++)
    {
      for (int J = 0; J < ntiles; J++)
      {
        auto bck = (*in)(I, J);

        for (int j = 0; j < TILESIZE; j++)
        {
          
          if (bck.where() == imyrank) {
            //printf("out index %d\n",(I*TILESIZE*TILESIZE*ntiles)+(J*TILESIZE*TILESIZE)+(i*TILESIZE)+j);
            //fflush(stdout);
            (*out)[(I*TILESIZE*TILESIZE*ntiles)+(J*TILESIZE*TILESIZE)+(i*TILESIZE)+j]=bck.local()->get(i,j);
          }
        }
      }
    }
  }
  
  // This is to ensure it was (1) run in SPMD and (2) initialized everywhere
}

void print_local_matrix(double *A,int ntiles)
{
  int dim = ntiles * TILESIZE;

  for(int i=0;i<dim;i++)
  {
    for(int j=0;j<dim;j++)
    {
      printf("%f ",A[i*dim+j]);
    }
    printf("\n");
  }
}

int arg_parse(int argc, char **argv)
{ int ch;

  while ( -1 != (ch = getopt(argc, argv,"a:B:c:n:o:r:")) ) {
    switch (ch) {
      case 'a':
        Algorithm = toupper(*optarg);
        break;
      case 'c':
        Col_cyc= (int)strtoul(optarg, NULL, 0);
        break;
      case 'n':
        NReps = (int)strtoul(optarg, NULL, 0);
        break;
      case 'o':
        Outfile = optarg;
        break;
      case 'r':
        Row_cyc =(int)strtoul(optarg, NULL, 0);
        break;
      default: // unknown or missing argument
        if (!imyrank) {
          printf("Unknown flag %c\n", ch);
        }
        exit( EXIT_FAILURE );
    }
  }
  
  if (Row_cyc * Col_cyc > upcxx::rank_n()) {
    if(!imyrank) {
      std::cerr << "Dsitribution -r x -c > number of processes available\n";
    }
    exit(EXIT_FAILURE);
  }
  
  if (argc > optind && 0 != atoi(argv[optind]))  {
    // std::cout << "Generating matrix of size " << argv[optind] << std::endl;
  } else {
    if(!imyrank) {
      std::cout << "Usage: lu  [-a alg] [-r row_cyc] [-c col_cyc] [-n reps] [-o outfile] dim" << std::endl
                << "-n reps  Number of times to repeat each experiment\n"
                << "-a alg: Use algorithm [D]=DepSpawn [F]=Filtered DepSpawn [S]=seq\n";
    }
    exit(EXIT_FAILURE);
  }
  
  return atoi(argv[optind]); //the dimension
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

int main(int argc, char** argv)
{
  int M = 8, N = 8;
  int info, lda;
  
  upcxx::init();
  
  imyrank = upcxx::rank_me();

  Row_cyc = upcxx::rank_n();
  Col_cyc = 1;
  NReps = 1;

  
  M = N = arg_parse(argc, argv);
  
  //mkl_set_num_threads(1);
 
  char *nthreads_env = getenv("UD_NUM_THREADS");
  const int nthreads = (nthreads_env == NULL) ? -1 : static_cast<int>(strtol(nthreads_env, (char **)NULL, 10));
  
  depspawn::set_threads(nthreads);
  depspawn::upcxx_depspawn_runtime_setup(); // Only so that the get_UPCXX_DEPSPAWN_* functions get right values
  enable_ps_master();  // The setup disables the master_persona

  const int exact_match_on = (int)depspawn::get_UPCXX_DEPSPAWN_EXACT_MATCH();
  const int prefetch_on = (int)depspawn::get_UPCXX_DEPSPAWN_PREFETCH();
  const int yield_on = (int)depspawn::get_UPCXX_DEPSPAWN_YIELD();
  const int active_wait_on = (int)depspawn::get_UPCXX_DEPSPAWN_ACTIVE_WAIT();

  if (!imyrank) {
    printf("nthreads=%d n=%d Row_cyc=%d Col_cyc=%d TILESIZE=%d alg=%c NReps=%d ExactMatch=%d Prefetch=%d ActWait=%d Yield=%d\n", nthreads, N, Row_cyc, Col_cyc, TILESIZE, Algorithm, NReps, exact_match_on, prefetch_on, active_wait_on, yield_on);
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
  char * const qlimit_env = getenv("DSP_QLIMIT");
  if (qlimit_env != NULL)
  {
      int l = static_cast<int>(strtol(qlimit_env, (char **)NULL, 10));
      printf("Queue limit=%d\n", l);
      depspawn::set_task_queue_limit(l);
  }
  */

  if ((N % TILESIZE) != 0)
  {
    fprintf(stderr, "The tile size is not compatible with the given matrix\n");
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }

  config_test(exact_match_on, "UPCXX_DEPSPAWN_EXACT_MATCH");
  config_test(prefetch_on, "UPCXX_DEPSPAWN_PREFETCH");
  config_test(yield_on, "UPCXX_DEPSPAWN_YIELD");
  config_test(active_wait_on, "UPCXX_DEPSPAWN_ACTIVE_WAIT");

  upcxx::barrier(); // For the sake of I/O

  //BBF: I deleted this matrix since it is only used for debugging
  // double * const locA = new double[M * N];
  
  const int ntiles = (M + TILESIZE - 1)/TILESIZE;
  int * const ipiv = nullptr; // (int *) malloc(sizeof(int) * M);

  DistrBlockMatrix<tile> *A{nullptr};

  try {
    A =  new DistrBlockMatrix<tile>(ntiles, Layout(std::make_pair(Row_cyc, Col_cyc)));
  } catch(std::runtime_error& e) {
    std::cerr << e.what() << " at col_row_sum test\n";
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }

  /*
  if (!imyrank) {
    printf("Floating point elements per matrix: %i x %i\n", M, M);
    printf("Floating point elements per tile: %i x %i\n", TILESIZE, TILESIZE);
    printf("tiles per matrix: %i x %i\n", ntiles, ntiles);
  }
  */

  for (int i = 0; i < NReps; i++) {

    // printf("Creating...\n");
    init(A, /*locA*/ nullptr, ipiv, M, ntiles);

    upcxx::barrier();

/*
 if (!imyrank)
 {
    //copy_distr_matrix(&A, &locA, ntiles);
    printf("Input Matrix\n");
    print_local_matrix(locA, ntiles);
  }
*/
    switch (Algorithm) {
      case 'D':
        lu_factorization(*A, &ipiv, M, ntiles);
        break;
      case 'F':
        lu_factorization_filtered(*A, &ipiv, M, ntiles);
        break;
      case 'S':
        lu_factorization_seq(*A, ntiles);
        break;
      default:
        if (!imyrank) {
          fprintf(stderr, "Uknown algorithm %c\n", Algorithm);
        }
        break;
    }
    

/*
  if (!imyrank)
  {
    copy_distr_matrix(&A, &locA, ntiles);
    printf("Output Matrix\n");  
    print_local_matrix(locA, ntiles);
  }
*/
  }
  
  if (Outfile != nullptr) {
    if (!imyrank) {
      FILE * const fout = strcmp(Outfile, "-") ? fopen(Outfile, "w") : stdout;
      tile * const row_tiles_buf = new tile[ntiles];
      for (int i = 0; i < ntiles; i++) {
        
        upcxx::promise<> all_done;
        // fetch remote data
        for (int j = 0; j < ntiles; j++) {
          upcxx::rget((*A)(i, j), row_tiles_buf + j, 1,
                      upcxx::operation_cx::as_promise(all_done));
        }
        all_done.finalize().wait();
        
        for(int i_b = 0; i_b < TILESIZE; i_b++) {
          for (int j = 0; j < ntiles; j++) {
            const tile& _tmp = row_tiles_buf[j];
            for(int j_b = 0; j_b < TILESIZE; j_b++) {
              fprintf(fout, "%lf ", _tmp.get( i_b, j_b ));
            }
          }
          fprintf(fout, "\n" );
        }
      }
      
      delete [] row_tiles_buf;
      fclose(fout);
    }
    upcxx::barrier();
  }

  delete A;

  upcxx::finalize();

  return 0;
}
