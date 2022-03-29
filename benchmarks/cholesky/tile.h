/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include <cmath>
#include <iostream>
#include <atomic>
#ifndef STACK_TILE
#include <memory>
#endif

//#ifdef USE_MKL
//#include "mkl_cblas.h"
//#else
#include "cblas.h"
//#endif

//#include "mkl_cblas.h"
//#include "mkl_lapack.h"

//using namespace depspawn;

//
// A tile is a square matrix, used to tile a larger array
//
#ifndef TILESIZE
#define TILESIZE 200
#endif

#define MINPIVOT 1E-12

class tile
{
  double m_tile[TILESIZE * TILESIZE];
  
public:
  
  static constexpr size_t BlockSize = TILESIZE;
  
#ifdef STACK_TILE
  using tile_handle_t = tile;
#else
  /// Supports creation of temporary tiles in the heap
  struct tile_handle_t {
    std::unique_ptr<tile> p_;
    
    tile_handle_t() : p_(new tile()) {}
    
    tile_handle_t(const tile& other) : p_(new tile(other)) {}
    
    double *data() { return (double *)(p_->m_tile); }
    const double *data() const { return (const double *)(p_->m_tile); }
    
    operator tile& () { return *p_; }
  };
#endif
  
  tile() = default;
  
  ~tile() = default;
  
  
  double *data() { return (double *)m_tile; }
  const double *data() const { return (const double *)m_tile; }
  
  inline void set(const int i, const int j, double d) noexcept { m_tile[i * TILESIZE + j] = d; }
  inline double get(const int i, const int j) const noexcept { return m_tile[i * TILESIZE + j]; }
  
  void dump(double epsilon = MINPIVOT) const
  {
    for (int i = 0; i < TILESIZE; i++)
    {
      for (int j = 0; j < TILESIZE; j++)
      {
        std::cout.width(10);
        double t = get(i, j);
        if (fabs(t) < MINPIVOT)
          t = 0.0;
        std::cout << t << " ";
      }
      std::cout << std::endl;
    }
  }
  
  int identity_check(double epsilon = MINPIVOT) const
  {
    int ecount = 0;
    for (int i = 0; i < TILESIZE; i++)
    {
      for (int j = 0; j < TILESIZE; j++)
      {
        double t = get(i, j);
        if (i == j && (fabs(t - 1.0) < epsilon))
          continue;
        if (fabs(t) < epsilon)
          continue;
        
        std::cout << "(" << i << "," << j << "):" << t << std::endl;
        ecount++;
      }
    }
    return ecount;
  }
  
  int zero_check(double epsilon = MINPIVOT) const
  {
    int ecount = 0;
    for (int i = 0; i < TILESIZE; i++)
    {
      for (int j = 0; j < TILESIZE; j++)
      {
        double t = get(i, j);
        if (fabs(t) < epsilon)
          continue;
        std::cout << "(" << i << "," << j << "):" << t << std::endl;
        ecount++;
      }
    }
    return ecount;
  }
  
  int equal(const tile &t) const
  {
    for (int i = 0; i < TILESIZE; i++)
    {
      for (int j = 0; j < TILESIZE; j++)
      {
        if (get(i, j) != t.get(i, j))
          return false;
      }
    }
    return true;
  }
  
  // c = this * b
  void multiply_(const tile &b, tile &c) const
  {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TILESIZE, TILESIZE, TILESIZE,
                1., (double *)m_tile, TILESIZE, (double *)b.m_tile, TILESIZE, 0., c.m_tile, TILESIZE);
  }
  
  tile_handle_t multiply(const tile &b) const
  {
    tile_handle_t c;
    multiply_(b, c);
    return c;
  }
  
  // c = -(this * b)
  void multiply_negate_(const tile &b, tile &c) const
  {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TILESIZE, TILESIZE, TILESIZE,
                -1., (double *)m_tile, TILESIZE, (double *)b.m_tile, TILESIZE, 0., c.m_tile, TILESIZE);
  }
  
  tile_handle_t multiply_negate(const tile &b) const
  {
    tile_handle_t c;
    multiply_negate_(b, c);
    return c;
  }
  
  // d = this - (b * c)
  tile_handle_t multiply_subtract(const tile &b, const tile &c) const
  {
    tile_handle_t d = *this;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TILESIZE, TILESIZE, TILESIZE,
                -1., b.data(), TILESIZE, c.data(), TILESIZE, 1., d.data(), TILESIZE);
    return d;
  }
  
  // this =  this - (b * c)
  void multiply_subtract_in_place(const tile &b, const tile &c)
  {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TILESIZE, TILESIZE, TILESIZE,
                -1., b.data(), TILESIZE, c.data(), TILESIZE, 1., m_tile, TILESIZE);
  }
  
  // d = this + (b * c)
  tile_handle_t multiply_add(const tile &b, const tile &c) const
  {
    tile_handle_t d = *this;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TILESIZE, TILESIZE, TILESIZE,
                1., b.data(), TILESIZE, c.data(), TILESIZE, 1., d.data(), TILESIZE);
    return d;
  }
  
  // this = this + (b * c)
  void multiply_add_in_place(const tile &b, const tile &c)
  {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TILESIZE, TILESIZE, TILESIZE,
                1., b.data(), TILESIZE, c.data(), TILESIZE, 1., m_tile, TILESIZE);
  }
  
  // this = 0.0;
  void zero()
  {
    for (int i = 0; i < TILESIZE; i++)
      for (int j = 0; j < TILESIZE; j++)
        set(i, j, 0.0);
  }
  
  double hash() const
  {
    double r = m_tile[0];
    int *const pr = (int *)&r;
    const int sz = sizeof(double) / sizeof(int);
    
    for (int i = 0; i < TILESIZE; i++)
      for (int j = 0; j < TILESIZE; j++)
      {
        const int *const p = (const int *)(m_tile + (i * TILESIZE + j));
        for (int k = 0; k < sz; k++)
        {
          pr[k] = pr[k] ^ p[k];
        }
      }
    
    return r;
  }
};


//extern void S1_compute(tile &A_block);
//extern void S2_compute(const tile &Li_block, tile &dest);
//extern void S_dsyrk(const tile &L2_block, tile &A_block);
//extern void S_dgemm(const tile &a, const tile &b, tile &c);

#ifndef DISTR_BLOCK_MATRIX_H
// we replicate this component for the sake of DistributedTiled2DMatrix
enum class MatrixType {Full, LowerTriangular, UpperTriangular};
#endif

///  Only stores the local portion for a matrix divided in tiles and distributed
/// following some2D cyclic distribution.
template<typename T>
struct DistributedTiled2DMatrix {
  
  const size_t rows, cols;             ///< original size in terms of elements
  const size_t row_cyc, col_cyc;       ///< processor mesh for 2D cyclic distribtion
  const size_t myrank;                 ///< rank of this process
  const size_t row_blocks, col_blocks; ///< number of blocks or tiles per dimension
  const MatrixType matrix_type;        ///< kind of full or triangular matrix
  T * * const data;
  
  DistributedTiled2DMatrix(size_t rows_in, size_t cols_in, size_t row_cyc_in, size_t col_cyc_in, size_t myrank_in, MatrixType matrix_type_in = MatrixType::Full) :
  rows(rows_in), cols(cols_in), row_cyc(row_cyc_in), col_cyc(col_cyc_in), myrank(myrank_in),
  row_blocks((rows + T::BlockSize -1) / T::BlockSize), col_blocks((cols + T::BlockSize -1) / T::BlockSize),
  matrix_type(matrix_type_in), data(new T * [row_blocks * col_blocks])
  {
    for (size_t i = 0; i < row_blocks; i++) {
      for (size_t j = 0; j < col_blocks; j++) {
        if( (matrix_type == MatrixType::Full) ||
           ( (matrix_type == MatrixType::LowerTriangular) && (i >= j) ) ||
           ( (matrix_type == MatrixType::UpperTriangular) && (i <= j) ) ) {
          const size_t rank = (i % row_cyc) * col_cyc + (j % col_cyc);
          data[i * row_blocks + j] = (rank == myrank) ? (new T()) : nullptr;
        } else {
          data[i * row_blocks + j] = nullptr;
        }
      }
    }
  }
  
  size_t get_rows() const noexcept { return rows; }
  
  size_t get_cols() const noexcept { return cols; }
  
  size_t get_row_blocks() const noexcept { return row_blocks; }
  
  size_t get_col_blocks() const noexcept { return col_blocks; }
  
  MatrixType get_matrix_type() const noexcept { return matrix_type; }
  
  T *get_tile_ptr(int i, int j) const noexcept { return data[i * row_blocks + j]; }
  
  T *get_tile_ptr_for_element(int i, int j) const noexcept { return get_tile_ptr(i / T::BlockSize, j / T::BlockSize); }
  
  bool is_local_tile(int i, int j) const noexcept { return (get_tile_ptr(i, j) != nullptr); }
  
  ~DistributedTiled2DMatrix()
  {
    for(size_t i = 0; i < row_blocks * col_blocks; i++) {
      if(data[i] != nullptr)
        delete data[i];
    }
    delete [] data;
  }
  
};
