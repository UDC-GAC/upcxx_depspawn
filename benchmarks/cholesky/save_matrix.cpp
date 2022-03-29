/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include "DistrBlockMatrix.h"
#include "tile.h"

namespace  {

  constexpr int b = TILESIZE;

  void dump(const int i, FILE * const fout, tile * const row_tiles_buf)
  {
    for(int i_b = 0; i_b < b; i_b++) { //rows of this row of tiles
      for (int j = 0; j <= i; j++) {
        const tile& _tmp = row_tiles_buf[j];
        const int limit = (j != i) ? b : i_b;
        for(int j_b = 0; j_b < limit; j_b++) {
          fprintf(fout, "%lf ", _tmp.get( i_b, j_b ));
        }
      }
      fprintf(fout, "\n" );
    }
  }

}

void save_matrix(DistrBlockMatrix<tile> &A, const char * const oname)
{
  const uint32_t imyrank = upcxx::rank_me();
  const int dim = A.rows();
  assert(A.rows() == A.cols());
  
  if (!imyrank) {
    FILE * const fout = strcmp(oname, "-") ? fopen(oname, "w") : stdout;
    tile * const row_tiles_buf = new tile[dim];
    for (int i = 0; i < dim; i++) {
      
      upcxx::promise<> all_done;
      // fetch remote data
      for (int j = 0; j <= i; j++) {
        upcxx::rget(A(i, j), row_tiles_buf + j, 1,
                    upcxx::operation_cx::as_promise(all_done));
      }
      all_done.finalize().wait();
      
      dump(i, fout, row_tiles_buf);
    }
    
    delete [] row_tiles_buf;
    fclose(fout);
  }
  upcxx::barrier();
}
