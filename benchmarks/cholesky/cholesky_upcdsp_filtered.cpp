/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/time.h>
#include <cstdlib>
#include <chrono>

#include "DistrBlockMatrix.h"
#include "tile.h"

#include "upcxx_depspawn/upcxx_depspawn.h"

using namespace depspawn;
using namespace upcxx;


extern void S1_compute(global_ptr<tile> a);
extern void S2_compute(cached_global_ptr<const tile> li, global_ptr<tile> a);
extern void S_dsyrk(cached_global_ptr<const tile> l2, global_ptr<tile> a);
extern void S_dgemm(cached_global_ptr<const tile> l1, cached_global_ptr<const tile> l2, global_ptr<tile> a);
extern void save_matrix(DistrBlockMatrix<tile> &A, const char * const oname);

/* Debugging functions
void S1_compute_d(global_ptr<tile> a, int id)
{
  std::cerr << upcxx::rank_me() << ' ' << id << std::endl;
  S1_compute(a);
}
void S2_compute_d(cached_global_ptr<const tile> li, global_ptr<tile> a, int id)
{
  std::cerr << upcxx::rank_me() << ' ' << id << std::endl;
  S2_compute(li, a);
}
void S_dsyrk_d(cached_global_ptr<const tile> l2, global_ptr<tile> a, int id)
{
  std::cerr << upcxx::rank_me() << ' ' << id << std::endl;
  S_dsyrk(l2, a);
}
void S_dgemm_d(cached_global_ptr<const tile> l1, cached_global_ptr<const tile> l2, global_ptr<tile> a, int id)
{
  std::cerr << upcxx::rank_me() << ' ' << id << std::endl;
  S_dgemm(l1, l2, a);
}
*/

extern bool MidProfiling;
extern int Row_cyc, Col_cyc, NReps;

extern void disable_ps_master();
extern void enable_ps_master();

void cholesky_filtered(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname)
{
  int p;
  int k;

  const int dim = (n + b - 1) / b;

  const uint32_t  imyrank = upcxx::rank_me();

  DistrBlockMatrix<tile> matrix(dim, Layout(std::make_pair(Row_cyc, Col_cyc)), MatrixType::LowerTriangular);

  const int my_proc_row = imyrank / Col_cyc;
  const int my_proc_col = imyrank % Col_cyc;

  // internal::upcxx_depspawn_start_system();

  for (int rep = 0; rep < NReps; rep++) {
    
    depspawn::get_task_pool().parallel_for((size_t)0, static_cast<size_t>(dim), (size_t)1, [&](size_t I) {
      for (size_t J = 0; J <= I; J++) { // we only operate on the lower triangular matrix (and main diagonal)
        auto bck = matrix(I, J);
        if (bck.where() == imyrank) {
          assert(A.is_local_tile(I, J));
          *bck.local() = *(A.get_tile_ptr(I, J));
        }
      }
    }, false);
    
    upcxx::barrier();

    disable_ps_master();
    auto t0 = std::chrono::high_resolution_clock::now();

    // algorithm
    for (int k = 0; k < dim; ++k)
    {
      const bool in_col_k = (k % Col_cyc) == my_proc_col;
      
      //S1_compute(matrix[k][k]));
      upcxx_cond_spawn(in_col_k,
                       S1_compute, matrix(k, k));

      for (int m = k + 1; m < dim; ++m)
      {
        //S2_compute(matrix[k][k], matrix[m][k]);
        upcxx_cond_spawn(in_col_k || ((m % Row_cyc) == my_proc_row) || ((m % Col_cyc) == my_proc_col),
                         S2_compute, matrix(k, k), matrix(m, k));
      }

      for (int n = k + 1; n < dim; ++n)
      {
        const bool in_row_n = (n % Row_cyc) == my_proc_row;
        const bool in_col_n = (n % Col_cyc) == my_proc_col;

        //S_dsyrk(matrix[n][k], matrix[n][n]);
        upcxx_cond_spawn(in_row_n && (in_col_n || in_col_k),
                         S_dsyrk, matrix(n, k), matrix(n, n));

        for (int m = n + 1; m < dim; ++m)
        {
          const bool in_row_m = (m % Row_cyc) == my_proc_row;
          
          // S_dgemm (matrix[m][k], matrix[n][k], matrix[m][n]);
          upcxx_cond_spawn((in_row_m && (in_col_n || in_col_k)) || (in_row_n && in_col_k),
                           S_dgemm, matrix(m, k), matrix(n, k), matrix(m, n));
        }
      }
    }

    if (MidProfiling) {
      auto t1 = std::chrono::high_resolution_clock::now();
      print_upcxx_depspawn_profile_results();
      if (!imyrank) {
        std::cout << "Rep " << rep <<  " Mid Time: " << (std::chrono::duration<double>(t1 - t0).count()) << " sec" << std::endl;
        puts("===================== FINAL PROFILING: =====================");
      }
    }

    // Wait for all steps to finish
    upcxx_wait_for_all();

    auto t1 = std::chrono::high_resolution_clock::now();
    enable_ps_master();
    
    if (!imyrank) {
      double time = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "Rep " << rep <<  " Total Time: " << time << " sec" << std::endl;
    }
    
  }
  
  if (oname != nullptr) {
    save_matrix(matrix, oname);
  }

}
