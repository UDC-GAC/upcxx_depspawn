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

int lu_factorization(DistrBlockMatrix<tile>& A, int * const *P, int n, int ntiles)
{
  int imyrank = upcxx::rank_me();
  int i,j,k;

  disable_ps_master();
  auto t0 = std::chrono::high_resolution_clock::now();

  for(k=0;k<ntiles;k++)
  {
    //task11
    upcxx_spawn(task11, A(k, k)) ;

    for(i=k+1;i<ntiles;i++)
    {
      //task12
      upcxx_spawn(task12, A(k, k), A(k, i)) ;
      //task21
      upcxx_spawn(task21, A(k, k), A(i, k)) ;
    }

    for(i=k+1;i<ntiles;i++)
    {
      for(j=k+1;j<ntiles;j++)
      {
        //task22
        upcxx_spawn(task22, A(k, j), A(i, k), A(i, j)) ;
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
