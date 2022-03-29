/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>

#include "tile.h"
#include "upcxx_depspawn/upcxx_depspawn.h"

bool BinaryFiles = false;

void posdef_gen( double * A, int in_n )
{
  /* Allocate memory for the matrix and its transpose */
  double *L;
  // double *LT;
  double two = 2.0;
  double one = 1.0;
  //	srand( 1 );
  
  const size_t n = static_cast<size_t>(in_n);

  L = (double *) calloc(sizeof(double), n * n);
  if( L == nullptr) {
    fprintf(stderr, "Not enough memory for L\n");
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  /*
  LT = (double *) calloc(sizeof(double), n * n);
  if( LT == nullptr) {
    fprintf(stderr, "Not enough memory for LT\n");
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  */
  
  memset( A, 0, sizeof( double ) * n * n );
  
  /* Generate a conditioned matrix and fill it with random numbers */
  depspawn::get_task_pool().parallel_for((size_t)0, n, (size_t)1, [&] (size_t j) {
    for(size_t k = 0; k <= j; k++) {
      if(k<j) {
        // The initial value has to be between [0,1].
        L[k*n+j] = ( ( (j*k) / ((double)(j+1)) / ((double)(k+2)) * two) - one ) / ((double)n);
      } else if (k == j) {
        L[k*n+j] = 1;
      }
    }
  }, false);
  
  /* Compute transpose of the matrix */
  /*
   // can be parallel
   for(size_t i = 0; i < n; i++) {
    for(size_t j = 0; j < n; j++) {
      LT[j*n+i] = L[i*n+j];
    }
  }
  */
  
  // cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, L, n, LT, n, 0, A, n );
  cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1, L, n, L, n, 0, A, n );

  free (L);
  // free (LT);
}

// Generate matrix
void matrix_init( double *A, int in_n)
{
  posdef_gen( A, in_n );
}

// Read the matrix from the input file
void matrix_init(DistributedTiled2DMatrix<tile>& A, int in_n, const char *fname )
{ bool skip_binary_row;
  
  assert( fname != nullptr);
  assert( A.get_rows() == A.get_cols() );
  assert( A.get_row_blocks() == A.get_col_blocks() );

  const size_t n = static_cast<size_t>(in_n);
  
  double * const row_buffer = new double[n];

  FILE * const fp = fopen(fname, BinaryFiles ? "rb" : "r");
  if(fp == nullptr) {
    fprintf(stderr, "\nFile %s does not exist\n", fname);
    exit(0);
  }
  
  for (size_t i = 0; i < n; i++) {
    
    if (BinaryFiles) { // Skip row if not in this processor
  
      if (!(i % TILESIZE)) {
        size_t j;
        for (j = 0; (j <= i) && (A.get_tile_ptr_for_element(i, j) == nullptr); j += TILESIZE);
        skip_binary_row = (j > i);
        //fprintf(stderr, "P %d will %s row %zu\n", upcxx::myrank(), skip_binary_row ? " NOT READ" : "READ", i);
      }
      
      if (skip_binary_row) {
        fseek (fp, sizeof(double) * (i + 1), SEEK_CUR);
        continue;
      }
  
    }
    
    if (BinaryFiles) {
      const size_t read_vals = fread(row_buffer, sizeof(double), i + 1, fp);
      if (read_vals != (i + 1)) {
        fprintf(stderr,"\nError reading file %s\n", fname);
        upcxx::finalize();
        exit(EXIT_FAILURE);
      }
    } else {
      for (size_t j = 0; j <= i; j++) {
        if( fscanf(fp, "%lf ", row_buffer + j) <= 0) {
          fprintf(stderr,"\nMatrix size incorrect %i %i\n", (int)i, (int)j);
          exit(0);
        }
      }
    }
    
    // offset for this row within a tile
    const size_t local_i_offset = (i % TILESIZE) * TILESIZE;

    for (size_t j = 0; j <= i; j += TILESIZE) {
      tile * const tile_p = A.get_tile_ptr_for_element(i, j);
      if (tile_p != nullptr) {
        memcpy(tile_p->data() + local_i_offset, row_buffer + j, TILESIZE * sizeof(double));
      }
    }
  }
  
  depspawn::get_task_pool().parallel_for((size_t)0, A.get_row_blocks(), (size_t)1, [&](size_t z) {
    tile * const tile_p = A.get_tile_ptr(z, z);
    if (tile_p != nullptr) {
      for (size_t i = 0; i < TILESIZE; i++) {
        for (size_t j = 0; j < i; j++) {
          tile_p->set(j , i, tile_p->get(i, j));
        }
      }
    }
  }, false);
  
  delete [] row_buffer;

  fclose(fp);
}

// write matrix to file
void matrix_write ( double *A, int in_n, const char *fname )
{
  const size_t n = static_cast<size_t>(in_n);
  if( fname ) {
    size_t i;
    FILE *fp;
    
    fp = fopen(fname, "w");
    if(fp == NULL) {
      fprintf(stderr, "\nCould not open file %s for writing.\n", fname );
      upcxx::finalize();
      exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < n; i++) {
      if (BinaryFiles) {
        const size_t written = fwrite(A + (i * n), sizeof(double), i + 1, fp);
        if (written != (i + 1)) {
          fprintf(stderr,"\nError writing file %s\n", fname);
          upcxx::finalize();
          exit(EXIT_FAILURE);
        }
      } else {
        for (size_t j = 0; j <= i; j++) {
          fprintf( fp, "%lf ", A[i*n+j] );
        }
        fprintf( fp, "\n" );
      }
    }
    
    fclose(fp);
  }
}

void S1_compute(tile &A_block)
{
  //char uplo = 'l';
  //int info;
  //dpotf2(&uplo, (const int *) TILESIZE, const_cast< double * >( (double *)A_block.m_tile ), (const int *) TILESIZE, &info);
  
  for (int k_b = 0; k_b < TILESIZE; k_b++) {
    double diag_value = A_block.get(k_b, k_b);
    if (diag_value <= 0) {
      fprintf(stderr, "Not a symmetric positive definite (SPD) matrix\n");
      exit(0);
    }
    diag_value = sqrt(diag_value);
    A_block.set(k_b, k_b, diag_value);
    
    for (int j_b = k_b + 1; j_b < TILESIZE; j_b++) {
      A_block.set(j_b, k_b, A_block.get(j_b, k_b) / diag_value);
    }
    
    for (int i_b = k_b + 1; i_b < TILESIZE; i_b++) {
      double tmp = A_block.get(i_b, k_b);
      for (int j_bb = k_b + 1; j_bb <= i_b; j_bb++) {
        A_block.set(i_b, j_bb, A_block.get(i_b, j_bb) - (tmp * A_block.get(j_bb, k_b)));
      }
    }
  }
}

// Perform triangular system solve on the input tile.
// Input to this step are the input tile and the output tile of the previous step.
void S2_compute(const tile &Li_block, tile &dest)
{
  for (int k_b = 0; k_b < TILESIZE; k_b++) {
    const double tmpk = Li_block.get(k_b, k_b);
    for (int i_b = 0; i_b < TILESIZE; i_b++) {
      dest.set(i_b, k_b, dest.get(i_b, k_b) / tmpk);
    }
    for (int i_b = 0; i_b < TILESIZE; i_b++) {
      const double tmp = dest.get(i_b, k_b);
      for (int j_b = k_b + 1; j_b < TILESIZE; j_b++) {
        dest.set(i_b, j_b, dest.get(i_b, j_b) - (Li_block.get(j_b, k_b) * tmp));
      }
    }
  }
}

void S_dsyrk(const tile &L2_block, tile &A_block)
{
  cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, TILESIZE, TILESIZE, -1., L2_block.data(), TILESIZE, 1., (double *)A_block.data(), TILESIZE);
}

void S_dgemm(const tile &a, const tile &b, tile &c)
{
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TILESIZE, TILESIZE, TILESIZE,
              -1., (double *)a.data(), TILESIZE, (double *)b.data(), TILESIZE, 1., c.data(), TILESIZE);
}

void S1_compute(upcxx::global_ptr<tile> a)
{
  S1_compute(*a.local());
}

void S2_compute(upcxx::cached_global_ptr<const tile> li, upcxx::global_ptr<tile> a)
{
  S2_compute(*li.local(), *a.local());
}

void S_dsyrk(upcxx::cached_global_ptr<const tile> l2, upcxx::global_ptr<tile> a)
{
  S_dsyrk(*l2.local(), *a.local());
}

void S_dgemm(upcxx::cached_global_ptr<const tile> l1, upcxx::cached_global_ptr<const tile> l2, upcxx::global_ptr<tile> a)
{
  S_dgemm(*l1.local(), *l2.local(), *a.local());
}
