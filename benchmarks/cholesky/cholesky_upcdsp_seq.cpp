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

using namespace upcxx;

extern void S1_compute(global_ptr<tile> a);
extern void S2_compute(cached_global_ptr<const tile> li, global_ptr<tile> a);
extern void S_dsyrk(cached_global_ptr<const tile> l2, global_ptr<tile> a);
extern void S_dgemm(cached_global_ptr<const tile> l1, cached_global_ptr<const tile> l2, global_ptr<tile> a);
extern void save_matrix(DistrBlockMatrix<tile> &A, const char * const oname);

extern int Row_cyc, Col_cyc, NReps;

void cholesky_seq(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname)
{

  const int dim = (n + b - 1) / b;

  const uint32_t  imyrank = upcxx::rank_me();

  DistrBlockMatrix<tile> matrix(dim, Layout(std::make_pair(Row_cyc, Col_cyc)), MatrixType::LowerTriangular);

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
    
    if(!upcxx::rank_me()) {

      auto t0 = std::chrono::high_resolution_clock::now();
      
      // algorithm
      for (int k = 0; k < dim; ++k)
      {
        //S1_compute(matrix[k][k]));
        S1_compute(matrix(k, k));
        
        for (int m = k + 1; m < dim; ++m)
        {
          //S2_compute(matrix[k][k], matrix[m][k]);
          S2_compute(matrix(k, k), matrix(m, k));
        }
        
        for (int n = k + 1; n < dim; ++n)
        {
          S_dsyrk(matrix(n, k), matrix(n, n));
          
          for (int m = n + 1; m < dim; ++m)
          {
            // S_dgemm (matrix[m][k], matrix[n][k], matrix[m][n]);
            S_dgemm(matrix(m, k), matrix(n, k), matrix(m,n));
          }
        }
      }
      
      auto t1 = std::chrono::high_resolution_clock::now();
      
      double time = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "Rep " << rep <<  " Total Time: " << time << " sec" << std::endl;
    }

    upcxx::barrier();

  }
  
  
  
  if (oname != nullptr) {
    save_matrix(matrix, oname);
  }
  
  upcxx::barrier();
}
