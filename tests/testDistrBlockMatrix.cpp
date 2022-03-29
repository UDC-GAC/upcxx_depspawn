/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     testDistrBlockMatrix.cpp
/// \brief    Tests wavefront of function with 3 args on a 2D matrix distributed by rows, colums or blocks provided by class DistrBlockMatrix
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

/* ~/new_upcxx/bin/upcxx -codemode=debug -threadmode=par -o testDistrBlockMatrix  testDistrBlockMatrix.cpp
   GASNET_FREEZE_ON_ERROR=1 GASNET_BACKTRACE=1 ~/new_upcxx/bin/upcxx-run -backtrace -n 2 ./testDistrBlockMatrix
   // GASNET_FREEZE_ON_ERROR=1 ~/new_upcxx/bin/upcxx-run -n 2 /opt/X11/bin/xterm -e lldb ./testDistrBlockMatrix
 */

#include <cstdlib>
#include <thread>
#include <iomanip>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "DistrBlockMatrix.h"

int ntest = 1;
upcxx::persona_scope *ps_master = nullptr;

template <typename T>
void print(const DistrBlockMatrix<T>& mx)
{
  upcxx::barrier();

  if (!upcxx::rank_me()) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.cols(); j++) {
        if (mx.is_valid(i, j)) {
          std::cout << '[' << std::setw(3) << (upcxx::rget(mx(i, j)).wait()) << "] ";
        } else {
          std::cout << "[xxx] ";
        }
        
      }
      std::cout << std::endl;
    }
    std::cout << "=================" << std::endl;
  }

  upcxx::barrier();
}

// the wavefront adds in point (i,j) the values previously computed to its west (left) and north (up)
void g(upcxx::global_ptr<int> value, upcxx::global_ptr<const int> left, upcxx::global_ptr<const int> up)
{
  upcxx::future<int> left_f = upcxx::rget(left); //future<const int> breaks
  upcxx::future<int> up_f = upcxx::rget(up);
  upcxx::future<int, int> both = upcxx::when_all(left_f, up_f);
  upcxx::future<int> result = both.then([](int a, int b) { return a + b; });
  const int res = result.wait();
  upcxx::rput(res, value).wait();
}

template <typename T>
void simple_wavefront(DistrBlockMatrix<T>& mx)
{
  const int maxwave = mx.rows() + mx.cols() - 2;
  
  for (int k = 2; k <= maxwave; k++) {
    for (int i = 1; i < mx.rows(); i++) {
      for (int j = 1; j < mx.cols(); j++) {
        if ((i + j) == k) {
          if (mx.is_valid(i, j) && mx.is_valid(i, j-1) && mx.is_valid(i-1, j)) {
            depspawn::upcxx_spawn(g, mx(i,j), mx(i, j-1), mx(i-1, j));
          }
        }
      }
    }
  }
  
  depspawn::upcxx_wait_for_all();
}

template <typename T>
bool test_simple_wavefront(const DistrBlockMatrix<T>& mx)
{ bool local_ok;
  
  for (int i = 0; i < mx.rows(); i++) {
    for (int j = 0; j < mx.cols(); j++) {
      if (mx.is_valid(i,j)) {
        const int tmp = upcxx::rget(mx(i,j)).wait();
        local_ok = (!mx.is_valid(i, j-1) || !mx.is_valid(i-1, j)) ? (tmp == (i + j + 1)) : (tmp == (upcxx::rget(mx(i, j-1)).wait() + upcxx::rget(mx(i-1, j)).wait()));
        if(!local_ok) {
          std::cerr << "Err at (" << i << ", " << j << ") != " << (upcxx::rget(mx(i, j-1)).wait()) << " + " << (upcxx::rget(mx(i-1, j)).wait()) << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

template<typename DIST>
bool main_test(const DIST &d, const bool docopy = false, const MatrixType matrix_type = MatrixType::Full)
{

  const int MxSize = 2 * upcxx::rank_n();
  if (!upcxx::master_persona().active_with_caller()) {
    std::cerr << 'P' << upcxx::rank_me() << "takes1 " << ntest << std::endl; //never used
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    //std::cerr << 'P' << upcxx::rank_me() << "owns1 " << ntest << std::endl; //std
  }

  DistrBlockMatrix<int> din(MxSize, MxSize, d);
  
  //DistrBlockMatrix<int> &din = * new DistrBlockMatrix<int>(MxSize, MxSize, d, matrix_type);
  
  for (int i = 0; i < din.rows(); i++) {
    for (int j = 0; j < din.cols(); j++) {
      if (din.is_valid(i,j)) {
        auto val = din(i,j);
        if(val.where() == upcxx::rank_me()) {
          *(val.local()) = i + j + 1;
        }
      }
    }
  }

  upcxx::barrier();
  
  DistrBlockMatrix<int> &di = docopy ? (* new DistrBlockMatrix<int>(din)) : din;

  di.test();
  if (ps_master) {
    delete ps_master;
    ps_master = nullptr;
  }
  simple_wavefront(di);
  //BBF: This object may be deallocated before the DistrBlockMatrix, givind place to an error
  //     because only the owner of the master_persona can destroy a dist_object
  //upcxx::persona_scope scope(upcxx::master_persona());
  if (!upcxx::master_persona().active_with_caller()) {
    //std::cerr << 'P' << upcxx::rank_me() << "takes2 " << ntest << std::endl; //std
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    std::cerr << 'P' << upcxx::rank_me() << "owns2 " << ntest << std::endl; // never used
  }
  print(di);
  //upcxx::global_ref<int> tmp = di(1, 0);
  //std::cout <<  tmp << " (" << tmp.where() << ", " << tmp.raw_ptr() << ") in " << upcxx::myrank() << std::endl;
  
  const bool test_result = test_simple_wavefront(di);
  
  upcxx::barrier();

  if (!upcxx::rank_me()) {
    const char *matrix_type_name;
    di.layout().print();
    std::cout << " layout Copied matrix=" << (docopy ? "TRUE" : "FALSE") << " Type: ";
    switch (matrix_type) {
      case MatrixType::Full:
        matrix_type_name = "Full";
        break;
      case MatrixType::LowerTriangular:
        matrix_type_name = "LowerTriangular";
        break;
      case MatrixType::UpperTriangular:
        matrix_type_name = "UpperTriangular";
    }
    std::cout << matrix_type_name;
    std::cout << " TEST" << ntest << ' ' << (test_result ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  ntest++;
  
  upcxx::barrier();
  
  assert(upcxx::master_persona().active_with_caller());

  return test_result;
}

bool test(const int nranks, const MatrixType Test_Matrix_Type)
{
  using LayoutType = Layout::LayoutType;

  return main_test( LayoutType::AT_0, false, Test_Matrix_Type)
  && main_test( LayoutType::CYC_ROWS, false, Test_Matrix_Type)
  && main_test( LayoutType::CYC_COLS, false, Test_Matrix_Type)
  && main_test( std::make_pair(nranks/2, 2), false, Test_Matrix_Type)
  && main_test( std::make_pair(2, nranks/2), false, Test_Matrix_Type)
  && main_test( nranks, false, Test_Matrix_Type)
  && main_test( LayoutType::AT_0, true, Test_Matrix_Type)
  && main_test( LayoutType::CYC_ROWS, true, Test_Matrix_Type)
  && main_test( LayoutType::CYC_COLS, true, Test_Matrix_Type)
  && main_test( std::make_pair(nranks/2, 2), true, Test_Matrix_Type)
  && main_test( std::make_pair(2, nranks/2), true, Test_Matrix_Type)
  && main_test( nranks, true, Test_Matrix_Type);
}

int main(int argc, char **argv)
{
  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 2;
  
  depspawn::set_threads(nthreads);

  upcxx::init();
  
  const int myrank = upcxx::rank_me();
  const int nranks = upcxx::rank_n();

  if (!myrank) {
    std::cout << "Using " << nranks << " Procs\n";
    if (nranks % 2) {
      std::cerr << "This test requires an even number of processors" << std::endl;
      return -1;
    }
  }

  const bool test_ok =
     test(nranks, MatrixType::Full)
  && test(nranks, MatrixType::LowerTriangular)
  && test(nranks, MatrixType::UpperTriangular);

  upcxx::finalize();
  
  if(!myrank && (ntest < 13)) { /* number of tests + 1 */
    assert(!test_ok);
    std::cerr << (13 - ntest) << " tests skipped\n";
  }

  return !test_ok;
}
