/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     dgetrf.cpp
/// \brief    LU kernel
/// \author   Diego Andrade       <diego.andrade@udc.es>
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>
///

#include <cstdlib>
#include <chrono>

//#include "mkl.h"
#include "upcxx_depspawn/upcxx_depspawn.h"

using namespace depspawn;
using namespace upcxx;

#include "tile.h"
#include "DistrBlockMatrix.h"

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

using namespace depspawn;

extern void task11(global_ptr<tile> a);
extern void task12(cached_global_ptr<const tile> a1, global_ptr<tile> a2);
extern void task21(cached_global_ptr<const tile> a1, global_ptr<tile> a2);
extern void task22(cached_global_ptr<const tile> a1, cached_global_ptr<const tile> a2, global_ptr<tile> a3);
extern void disable_ps_master();
extern void enable_ps_master();

extern int Row_cyc, Col_cyc;

int lu_factorization_filtered(DistrBlockMatrix<tile>& A, int * const *P, int n, int ntiles)
{ int i, j, k;

  const int imyrank = upcxx::rank_me();
  const int my_proc_row = imyrank / Col_cyc;
  const int my_proc_col = imyrank % Col_cyc;
  
  disable_ps_master();
  auto t0 = std::chrono::high_resolution_clock::now();

  for(k=0; k<ntiles; k++)
  {
    const bool in_col_k = (k % Col_cyc) == my_proc_col;
    const bool in_row_k = (k % Row_cyc) == my_proc_row;

    //task11
    upcxx_cond_spawn(in_col_k || in_row_k, task11, A(k, k)) ;

    for(i=k+1; i<ntiles; i++)
    {
      const bool in_col_i = (i % Col_cyc) == my_proc_col;
      const bool in_row_i = (i % Row_cyc) == my_proc_row;
  
      //task12
      upcxx_cond_spawn((in_row_k && in_col_k) || in_col_i, task12, A(k, k), A(k, i)) ;
      //task21
      upcxx_cond_spawn((in_row_k && in_col_k) || in_row_i, task21, A(k, k), A(i, k)) ;
    }

    for(i=k+1; i<ntiles; i++)
    {
      const bool in_row_i = (i % Row_cyc) == my_proc_row;
      
      for(j=k+1; j<ntiles; j++)
      {
        const bool in_col_j = (j % Col_cyc) == my_proc_col;
        
        //task22
        upcxx_cond_spawn((in_row_i && (in_col_k || in_col_j)) || (in_row_k && in_col_j), task22, A(k, j), A(i, k), A(i, j)) ;
      }

    }

  }
  
  // Wait for all steps to finish
  upcxx_wait_for_all();

  auto t1 = std::chrono::high_resolution_clock::now();
  enable_ps_master();
  
  if (!imyrank)
  {
    double time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " Total Time: " << time << " sec" << std::endl;
  }
  return 0;

  /* End of DGETRF */

} /* dgetrf_ */
