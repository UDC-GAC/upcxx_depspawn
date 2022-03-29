/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     DistrBlockMatrix.h
/// \brief    DistrBlockMatrix class that represents a 2D matrix that can be distributed in several ways among UPC++ processes
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>


#ifndef DISTR_BLOCK_MATRIX_H
#define DISTR_BLOCK_MATRIX_H

#include <cassert>
#include <exception>
#include <iostream>
#include <vector>
#include <upcxx/upcxx.hpp>

/// Represents the distribution of the matrix. It supports no distribution (everything at rank 0)
///and cyclic distributions by rows, columns and 2D by elements (which should be typically tiles).
class Layout {

public:

  enum class LayoutType {AT_0, CYC_ROWS, CYC_COLS, CYC_2D};

private:

  const LayoutType layoutType_;
  const std::pair<int, int> cyc_;
  
  // C++11 has no constexpr constructor for std::pair
  static std::pair<int, int> partition_rows_half(const std::pair<int, int>& inpair) {
    return ((inpair.first > inpair.second) && !(inpair.first % 2)) ? partition_rows_half({inpair.first / 2, inpair.second * 2}) : inpair;
  }

  static std::pair<int, int> default_mesh(const LayoutType layoutType)
  {
    switch (layoutType) {
      case LayoutType::AT_0 :    return {1, 1};
      case LayoutType::CYC_ROWS: return {upcxx::rank_n(), 1};
      case LayoutType::CYC_COLS: return {1, upcxx::rank_n()};
      case LayoutType::CYC_2D:   return partition_rows_half({upcxx::rank_n(), 1});
      default: throw std::logic_error("Impossible Layout type!\n");
    }
  }

public:

  Layout(LayoutType layoutType, const std::pair<int, int>& cyc) :
  layoutType_(layoutType),
  cyc_(cyc)
  {
    switch (layoutType) {
      case LayoutType::AT_0 :
        if(mesh_size() != 1) {
          throw std::logic_error("AT_0 layout with mesh!=(1,1)\n");
        }
        break;
      case LayoutType::CYC_ROWS:
        if (cyc.second != 1) {
          throw std::logic_error("CYC_ROWS layout with mesh!=(x,1)\n");
        }
        break;
      case LayoutType::CYC_COLS:
        if (cyc.first != 1) {
          throw std::logic_error("CYC_COLS layout with mesh!=(1,x)\n");
        }
        break;
      case LayoutType::CYC_2D:
        break;
      default:
        throw std::logic_error("Impossible Layout type!\n");
    }

    if(mesh_size() > upcxx::rank_n()) {
      throw std::logic_error("mesh with more than upcxx::rank_n() processors");
    }

    if(mesh_size() <= 0) {
      throw std::logic_error("mesh with <=0 processors");
    }

  }

  Layout(LayoutType layoutType = LayoutType::AT_0) :
  Layout(layoutType, default_mesh(layoutType))
  { }
  
  /// Simplified constructor for cyclic 2D distribution
  Layout(const std::pair<int, int>& cyc) :
  Layout(LayoutType::CYC_2D, cyc)
  {  }
  
  /// Super simplified constructor for cyclic 2D distribution
  Layout(const int nprocs) :
  Layout(partition_rows_half({nprocs, 1}))
  {  }
  
  /// Copy constructor
  constexpr Layout(const Layout& other) :
  layoutType_(other.layoutType_),
  cyc_(other.cyc_)
  {}
  
  void print(std::ostream& s = std::cout) const {
    switch(layoutType_) {
      case LayoutType::AT_0 :    s << "AT_0"; break;
      case LayoutType::CYC_ROWS: s << "CYC_ROWS"; break;
      case LayoutType::CYC_COLS: s << "CYC_COLS"; break;
      case LayoutType::CYC_2D:   s << "CYC_2D"; break;
      default: throw std::logic_error("Impossible Layout type!\n");
    }
    s << '(' << cyc_.first << ',' << cyc_.second << ')';
  }
  
  /// Returns a transposed layout
  ///
  /// AT_0 stays the same, CYC_ROWS and CYC_COLS are exchanged, and CYC_2D transposes its distribution mesh
  Layout transpose() const {
    switch(layoutType_) {
      case LayoutType::AT_0 :    return Layout(LayoutType::AT_0);
      case LayoutType::CYC_ROWS: return Layout(LayoutType::CYC_COLS, {1, rows()});
      case LayoutType::CYC_COLS: return Layout(LayoutType::CYC_ROWS, {cols(), 1});
      case LayoutType::CYC_2D:   return Layout({cyc_.second, cyc_.first});
      default: throw std::logic_error("Impossible Layout type!\n");
    }
  }
  
  /// Returns a CYC_2D version of the layout
  Layout normalize() const { return Layout(mesh()); }

  int rows() const noexcept { return cyc_.first; }
  
  int cols() const noexcept { return cyc_.second; }

  int mesh_size() const noexcept { return rows() * cols(); }

  const LayoutType& layout_type() const noexcept { return layoutType_; }

  const std::pair<int, int>& mesh() const noexcept { return cyc_; }

  /// Returns the (row, col) location in the mesh of the calling rank
  /// or (-1, -1) if it is outside the mesh
  std::pair<int, int> mesh_pos() const noexcept {
    const upcxx::intrank_t me = upcxx::rank_me();
    const int my_row =  me / cyc_.second;
    return (my_row >= cyc_.first) ? std::pair<int, int>(-1, -1) : std::pair<int, int>(my_row, me % cyc_.second);
  }

};


/// Kind of matrix
enum class MatrixType {Full, LowerTriangular, UpperTriangular};


/// Distributed matrix in which each element is allocated individually
template <typename T>
class DistrBlockMatrix  {

  const int rows_;
  const int cols_;
  const Layout layout_;
  const MatrixType matrix_type_;
  
  upcxx::global_ptr<upcxx::global_ptr<T>> local_data_;          ///< local copy of pointers to all the elements of the matrix
  
  void init_tiles_()
  { std::vector<upcxx::global_ptr<T>*> src;
    std::vector<upcxx::global_ptr<upcxx::global_ptr<T>>> dest;
    int rank; //, valids = 0;

    const int myrank = upcxx::rank_me();
    const int nranks = upcxx::rank_n();

    upcxx::global_ptr<upcxx::global_ptr<T>> data_0 = local_data_;
    upcxx::global_ptr<T>* const local_ptr = data_0.local();

    data_0 = upcxx::broadcast(data_0, 0).wait(); //Everyone gets local_data_ from P0
    
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        if (is_valid(i, j)) {
          //valids++;
          switch (layout_.layout_type()) {
            case Layout::LayoutType::AT_0:
              rank = 0;
              break;
            case Layout::LayoutType::CYC_ROWS:
              rank = i % nranks;
              break;
            case Layout::LayoutType::CYC_COLS:
              rank = j % nranks;
              break;
            case Layout::LayoutType::CYC_2D:
              rank = (i % layout_.rows()) * layout_.cols() + (j % layout_.cols());
              break;
            default:
              throw std::logic_error("Invalid layout");
              break;
          }
        } else {
          rank = -1;
        }

        if(rank == myrank) {
          const int linear_index = i * cols_ + j;
          local_ptr[linear_index] = upcxx::new_<T>();
          if (myrank) {
            src.emplace_back(local_ptr + linear_index);
            dest.emplace_back(data_0 + linear_index);
          }
        }
        
      }
    }
    
    if (!src.empty()) {
      upcxx::rput_regular(src.begin(), src.end(), 1, dest.begin(), dest.end(), 1).wait();
    }

    upcxx::barrier(); // Ensure all rputs finished before the broadcast begins
    
    upcxx::broadcast(local_ptr, rows_ * cols_, 0).wait(); // get all the pointers from 0
    
  }
  
public:

  /// Main constructor
  DistrBlockMatrix(int rows, int cols, const Layout& layout = {}, const MatrixType matrix_type = MatrixType::Full) :
  rows_(rows), cols_(cols), layout_(layout), matrix_type_(matrix_type),
  local_data_(upcxx::new_array<upcxx::global_ptr<T>>(rows * cols))
  {
    init_tiles_();
  }
  
  /// Square matrix constructor
  DistrBlockMatrix(int dim, const Layout& layout = {}, const MatrixType matrix_type = MatrixType::Full) :
  DistrBlockMatrix(dim, dim, layout, matrix_type)
  { }

  /// Copy constructor
  DistrBlockMatrix(const DistrBlockMatrix& other) :
  rows_(other.rows_), cols_(other.cols_), layout_(other.layout_), matrix_type_(other.matrix_type_),
  local_data_(upcxx::new_array<upcxx::global_ptr<T>>(rows_ * cols_))
  {
    init_tiles_();

    upcxx::global_ptr<T>* const other_local_ptr = other.local_data_.local();
    upcxx::global_ptr<T>* const my_local_ptr = local_data_.local();
    const auto my_rank = upcxx::rank_me();

    for(int i = 0; i < rows_ * cols_; i++) {
      const auto& ptr  = other_local_ptr[i];
      if ( !ptr.is_null() && (ptr.where() == my_rank) ) {
        *my_local_ptr[i].local() = *ptr.local();
      }
    }
  }
  
  /// Move constructor
  DistrBlockMatrix(DistrBlockMatrix&& other) :
  rows_(other.rows_), cols_(other.cols_), layout_(other.layout_), matrix_type_(other.matrix_type_),
  local_data_(other.local_data_)
  {
    other.local_data_ = nullptr; // Marks it as moved
  }
  
  int rows() const noexcept { return rows_; }
  
  int cols() const noexcept { return cols_; }

  std::pair<int, int> dimensions() const noexcept { return {rows_, cols_}; }

  const Layout& layout() const noexcept { return layout_; }
  
  MatrixType matrix_type() const noexcept { return matrix_type_; }

  upcxx::global_ptr<const T> operator() (int i, int j) const noexcept {
    return (local_data_.local())[i * cols_ + j];
  }
  
  upcxx::global_ptr<T> operator() (int i, int j) noexcept {
    return (local_data_.local())[i * cols_ + j];
  }
  
  bool is_valid(int i, int j) const noexcept {
    return (i >= 0) && (i < rows_) &&
           (j >= 0) && (j < cols_) &&
           ( (matrix_type_ == MatrixType::Full) ||
             ( (matrix_type_ == MatrixType::LowerTriangular) && (i >= j) ) ||
             ( (matrix_type_ == MatrixType::UpperTriangular) && (i <= j) ) );
  }
  
  // uplimit is the number of processors that print what they see
  void test(int uplimit = 0) const
  { int should_be_proc;
    bool test_ok = true;

    const int nranks = upcxx::rank_n();
    const int k_limit = uplimit ? uplimit : nranks;
    upcxx::global_ptr<T>* const local_ptr = local_data_.local();
  
    for (int k = 0; k < k_limit; k++) {
      if (upcxx::rank_me() == k) {
        for (int i = 0; i < rows_; i++) {
          for (int j = 0; j < cols_; j++) {
            
            const upcxx::global_ptr<T> tmpp = local_ptr[i * cols_ + j];
            
            if (is_valid(i, j)) {
              switch (layout_.layout_type()) {
                case Layout::LayoutType::AT_0:
                  should_be_proc = 0;
                  break;
                case Layout::LayoutType::CYC_ROWS:
                  should_be_proc = i % nranks;
                  break;
                case Layout::LayoutType::CYC_COLS:
                  should_be_proc = j % nranks;
                  break;
                case Layout::LayoutType::CYC_2D:
                  should_be_proc = (i % layout_.rows()) * layout_.cols() + (j % layout_.cols());
                  break;
                default:
                  throw std::logic_error("Invalid layout");
                  break;
              }
            } else {
              should_be_proc = 0; //invalid blocks should point to nullptr in rank 0
            }
            if (should_be_proc != tmpp.where()) {
              std::cerr << "Misplacement in P" << upcxx::rank_me() << '(' << i << ", " << j << "): is in " << tmpp.where() << " should be in " << should_be_proc << std::endl;
              test_ok = false;
            }
            //std::cout << 'P' << upcxx::rank_me() << '(' << i << ", " << j << ") in " << tmpp.where() << " V=" <<  (*tmpp) << std::endl;
          }
        }
        std::cout <<  "P " << upcxx::rank_me() <<" placements tested " << (test_ok ? "OK" : "WRONG") << std::endl;
      }
      upcxx::barrier();
    }
  }

  ~DistrBlockMatrix()
  {
    if (upcxx::initialized() && (local_data_ != nullptr)) {

      upcxx::global_ptr<T>* const local_ptr = local_data_.local();
      assert(local_ptr != nullptr);

      const upcxx::intrank_t my_rank = upcxx::rank_me();
      for (int i = 0; i < rows_ * cols_; i++) {
        const upcxx::global_ptr<T>& ptr = local_ptr[i];
        if (!ptr.is_null() && (ptr.where() == my_rank) ) {
          upcxx::delete_<T>(ptr);
        }
      }

      upcxx::delete_array(local_data_);
    }
  }

};


#endif //DISTR_BLOCK_MATRIX_H
