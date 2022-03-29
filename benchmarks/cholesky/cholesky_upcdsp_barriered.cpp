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

extern void S1_compute(tile &A_block);
extern void S2_compute(const tile &Li_block, tile &dest);
extern void S_dsyrk(const tile &L2_block, tile &A_block);
extern void S_dgemm(const tile &a, const tile &b, tile &c);
extern void save_matrix(DistrBlockMatrix<tile> &A, const char * const oname);

extern int Row_cyc, Col_cyc, NReps;

void cholesky_barriered(DistributedTiled2DMatrix<tile>& A, const int n, const int b, const char *oname)
{
  const int dim = (n + b - 1) / b;

  const uint32_t  imyrank = upcxx::rank_me();

  DistrBlockMatrix<tile> matrix(dim, Layout(std::make_pair(Row_cyc, Col_cyc)), MatrixType::LowerTriangular);

  tile * const remote_tile = new tile[2];

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

    auto t0 = std::chrono::high_resolution_clock::now();

    // algorithm
    for (int k = 0; k < dim; ++k)
    {
      //S1_compute(matrix[k][k]));
      const upcxx::global_ptr<tile> kk = matrix(k, k);

      if (kk.where() == imyrank) {
        S1_compute(*kk.local());
      }

      upcxx::barrier();

      upcxx::rget(kk, &(remote_tile[0]), 1).wait(); // Bring a copy of kk for everyone      

      for (int m = k + 1; m < dim; ++m)
      {
        //S2_compute(matrix[k][k], matrix[m][k]);
        const upcxx::global_ptr<tile> mk = matrix(m, k);
        if (mk.where() == imyrank) {
          S2_compute(remote_tile[0], *mk.local());
        }
      }
      upcxx::barrier();

      for (int n = k + 1; n < dim; ++n)
      {
        auto fut_nk = upcxx::rget(matrix(n, k), &(remote_tile[1]), 1);
        fut_nk.wait();
        
        //S_dsyrk(matrix[n][k], matrix[n][n]);
        const upcxx::global_ptr<tile> nn = matrix(n, n);
        if (nn.where() == imyrank) {
          S_dsyrk(remote_tile[1], *nn.local());
        }

        for (int m = n + 1; m < dim; ++m)
        {
          // S_dgemm (matrix[m][k], matrix[n][k], matrix[m][n]);
          const upcxx::global_ptr<tile> mn = matrix(m, n);
          if (mn.where() == imyrank) {
            upcxx::rget(matrix(m, k), &(remote_tile[0]), 1).wait();
            S_dgemm(remote_tile[0], remote_tile[1], *mn.local());
          }
        }
      }
      
      upcxx::barrier();
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (!imyrank) {
      double time = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "Rep " << rep <<  " Total Time: " << time << " sec" << std::endl;
    }
    
  }
  
  delete [] remote_tile;

  if (oname != nullptr) {
    save_matrix(matrix, oname);
  }

}
