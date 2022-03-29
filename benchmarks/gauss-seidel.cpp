/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

/// \file     gauss-seidel.cpp
/// \brief    Gauss-Seidel benchmark
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <tuple>
#include "upcxx_depspawn/upcxx_depspawn.h"

/* Debug with : See https://bitbucket.org/berkeleylab/upcxx/wiki/docs/debugging

   export GASNET_FREEZE_ON_ERROR=1 GASNET_BACKTRACE=1
   ~/new_upcxx/bin/upcxx-run -n  4 ./gauss-seidel  -B 1 -c 2 -r 2 -N 10 -C 2000
*/

#ifdef PRIORITY_TILESIZE
constexpr int TILE =  PRIORITY_TILESIZE;
#else
#ifdef TILESIZE
constexpr int TILE =  TILESIZE;
#else
constexpr int TILE = 1000;
#endif // TILESIZE
#endif // PRIORITY_TILESIZE

using namespace upcxx;

int NUM_ITERS = 10;

intrank_t imyrank;

struct Row {
  double val[TILE+2]; // Includes left + right halos
  
  double  operator[](int i) const noexcept { return val[i]; }
  double& operator[](int i) noexcept { return val[i]; }
};

struct Tile {
  Row val[TILE+2]; // Includes all halos
  
  const Row&  operator[](int i) const noexcept { return val[i]; }
  Row& operator[](int i) noexcept { return val[i]; }
};

struct FullTile {
  upcxx::global_ptr<Tile> tile_ptr_;
  upcxx::global_ptr<Row> top_ptr_, bottom_ptr_;
  upcxx::global_ptr<Row> left_ptr_, right_ptr_;
};


class WaveFrontProbl {
  
  int rows_, cols_;
  bool cyclic_;
  std::pair<int, int> mesh_, tiles_, tiles_per_rank_;
  
  FullTile *tiles_arr_;
  upcxx::global_ptr<Row> *top_most_row_, *bottom_most_row_, *left_most_row_, *right_most_row_;
  
  int * tile_counter_;
  int finish_;

  upcxx::dist_object<WaveFrontProbl> *self_dist_object_;
  
  const FullTile& get_FullTile(int tile_i, int tile_j) const noexcept {
    return tiles_arr_[tile_i * tiles_.second + tile_j];
  }
  
  FullTile& get_FullTile(int tile_i, int tile_j) noexcept {
    return tiles_arr_[tile_i * tiles_.second + tile_j];
  }
  
public:
  
  void init(int rows, int cols, std::pair<int, int> mesh, bool cyclic, upcxx::dist_object<WaveFrontProbl> *self_dist_object);
  void deallocate();
  void fill_in();
  double dsp_block_gauss();
  double dsp_filtered_block_gauss();
  double future_block_gauss();
  double barrier_block_gauss();
  double async_block_gauss();
  bool equal_arr(WaveFrontProbl& other) const;
  
  int get_rows() const noexcept { return rows_; }
  int get_cols() const noexcept { return cols_; }
  const std::pair<int, int>& get_tiles() const noexcept { return tiles_; }
  const std::pair<int, int>& get_tiles_per_rank() const noexcept { return tiles_per_rank_; }
  
  /// Computes rank that owns this tile
  int owner_tile(int tile_i, int tile_j) const noexcept {
    if(cyclic_) {
      return (tile_i % mesh_.first) * mesh_.second + (tile_j % mesh_.second);
    } else {
      return (tile_i / tiles_per_rank_.first) * mesh_.second + (tile_j / tiles_per_rank_.second);
    }
  }
  
  global_ptr<Tile> get_tile(int tile_i, int tile_j) noexcept {
    return get_FullTile(tile_i, tile_j).tile_ptr_;
  }
  
  global_ptr<const Tile> get_tile(int tile_i, int tile_j) const noexcept {
    return get_FullTile(tile_i, tile_j).tile_ptr_;
  }
  
  global_ptr<Row> get_top(int tile_i, int tile_j) noexcept {
    return get_FullTile(tile_i, tile_j).top_ptr_;
  }
  
  global_ptr<Row> get_bottom(int tile_i, int tile_j) noexcept {
    return get_FullTile(tile_i, tile_j).bottom_ptr_;
  }
  
  global_ptr<Row> get_left(int tile_i, int tile_j) noexcept {
    return get_FullTile(tile_i, tile_j).left_ptr_;
  }
  
  global_ptr<Row> get_right(int tile_i, int tile_j) noexcept {
    return get_FullTile(tile_i, tile_j).right_ptr_;
  }
  
  global_ptr<Row> get_top_for(int tile_i, int tile_j) noexcept {
    return tile_i ? get_bottom(tile_i - 1, tile_j) : top_most_row_[tile_j];
  }
    
  global_ptr<Row> get_bottom_for(int tile_i, int tile_j) noexcept {
    return (tile_i < (tiles_.first - 1)) ? get_top(tile_i + 1, tile_j) : bottom_most_row_[tile_j];
  }
    
  global_ptr<Row> get_left_for(int tile_i, int tile_j) noexcept {
    return tile_j ? get_right(tile_i, tile_j - 1) : left_most_row_[tile_i];
  }
      
  global_ptr<Row> get_right_for(int tile_i, int tile_j) noexcept {
    return (tile_j < (tiles_.second - 1))? get_left(tile_i, tile_j + 1) : right_most_row_[tile_i];
  }
  
  int get_tile_counter(int tile_i, int tile_j) const noexcept {
    return tile_counter_[tile_i * tiles_.second + tile_j];
  }

  void set_tile_counter(int tile_i, int tile_j, int val) noexcept {
    tile_counter_[tile_i * tiles_.second + tile_j] = 0;
  }
    
  int inc_tile_counter(int tile_i, int tile_j) noexcept {
    return ++tile_counter_[tile_i * tiles_.second + tile_j];
  }

  void incr_finish() noexcept { finish_++; }

  int get_finish() const noexcept { return finish_; }
};

constexpr int MaxVersion = 4;

int NRanks, NReps, Baseline_Version;
std::vector<int> Versions;
int nthreads;
std::pair<int, int> Mesh;
bool Do_Check = false;
bool TuneMode = false;
bool CyclicDistribution = true;
int SubProblems = 1;
std::atomic<int> Num_live_tasks{0};
//std::atomic<int> try_upcxx_progress_task{0};
upcxx::dist_object<WaveFrontProbl> *probl;
WaveFrontProbl test_probl;
upcxx::persona_scope *ps_master = nullptr;
std::mutex Tile_counter_mutex; // Only used in notify_run for async_block_gauss implementation
double MinTime, MinTimeFactor{1e20};

//void my_debug(WaveFrontProbl * p, int tile_i, int tile_j, int i, int j, const char *extr = "")
//{
//  auto ref = p->get_tile(tile_i, tile_j);
//  if (ref.where() == imyrank) {
//    const Tile *ptr = ref.local();
//    fprintf(stderr, "R%d %s (%d,%d)[%d,%d] = %lf\n", imyrank, extr, tile_i, tile_j, i, j, (*ptr)[i][j]);
//  }
//}

void disable_ps_master()
{
  if (ps_master) {
    delete ps_master;
    ps_master = nullptr;
  }
}

void enable_ps_master()
{
  if (!upcxx::master_persona().active_with_caller()) {
    //std::cerr << 'P' << upcxx::rank_me() << "takes\n"; //std
    ps_master =  new upcxx::persona_scope(upcxx::master_persona());
  } else {
    // std::cerr << 'P' << upcxx::rank_me() << " owns\n";
  }
}

void try_upcxx_progress()
{
  upcxx::progress();
}

/*
void try_upcxx_progress()
{
  static volatile bool no_one_advancing = true;
  
  if (no_one_advancing) {
    no_one_advancing = false;
    upcxx::persona my_persona;
    upcxx::persona_scope scope(my_persona);
    do {
      upcxx::progress();
    } while(upcxx::progress_required());
    no_one_advancing = true;
  }
}

// works while there is still a single live task
void upcxx_advance_helper()
{
  if (!try_upcxx_progress_task.exchange(1)) {
    while (Num_live_tasks.load()) {
      try_upcxx_progress();
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    try_upcxx_progress_task.store(0);
  }
}

// works while WaveFrontProbl has not finished and there < nthread live tasks
void upcxx_advance_helper2(const WaveFrontProbl * const p)
{
  if (!try_upcxx_progress_task.exchange(1)) {
    //fprintf(stderr, "R%d adv\n", imyrank);
    while ( !p->get_finish() && (Num_live_tasks.load() < nthreads) ) {
      try_upcxx_progress();
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    //fprintf(stderr, "R%d adv exit\n", imyrank);
    try_upcxx_progress_task.store(0);
  }
}
*/

void WaveFrontProbl::init(int rows, int cols, std::pair<int, int> mesh, bool cyclic, upcxx::dist_object<WaveFrontProbl> *self_dist_object)
{
  assert(rows && !(rows % TILE));
  assert(cols && !(cols % TILE));
  assert(mesh.first && mesh.second);

  self_dist_object_ = self_dist_object;

  rows_ = rows;
  cols_ = cols;
  cyclic_ = cyclic;
  mesh_ = mesh;

  tiles_.first = rows_ / TILE;
  tiles_.second = cols_ / TILE;
  tiles_per_rank_.first = tiles_.first / mesh_.first;
  tiles_per_rank_.second = tiles_.second / mesh_.second;
  
  assert(tiles_per_rank_.first && tiles_per_rank_.second);

  tiles_arr_       = new FullTile[tiles_.first * tiles_.second];
  top_most_row_    = new upcxx::global_ptr<Row>[tiles_.second];
  bottom_most_row_ = new upcxx::global_ptr<Row>[tiles_.second];
  left_most_row_   = new upcxx::global_ptr<Row>[tiles_.first];
  right_most_row_  = new upcxx::global_ptr<Row>[tiles_.first];
  
  upcxx::barrier();
  
  for (int tile_i = 0; tile_i < tiles_.first; tile_i++) {
    for (int tile_j = 0; tile_j < tiles_.second; tile_j++) {
      FullTile& full_tile = tiles_arr_[tile_i * tiles_.second + tile_j];
      const auto owner = owner_tile(tile_i, tile_j);
      if (imyrank == owner) {
        full_tile.tile_ptr_ = upcxx::new_array<Tile>(1);
        full_tile.top_ptr_ = upcxx::reinterpret_pointer_cast<Row>(full_tile.tile_ptr_) + 1; // Skip halo
        full_tile.bottom_ptr_ = full_tile.top_ptr_ + (TILE - 1);
        full_tile.left_ptr_ = upcxx::new_array<Row>(2);  // also allocates right
        full_tile.right_ptr_ = full_tile.left_ptr_ + 1;
      }
      full_tile = upcxx::broadcast(full_tile, owner).wait();
    }
  }
  
  upcxx::barrier();
  
  
  for (int tile_j = 0; tile_j < tiles_.second; tile_j++) {
    
    auto owner = owner_tile(0, tile_j);
    if (imyrank == owner) {
      top_most_row_[tile_j] = upcxx::new_array<Row>(1);
    }
    top_most_row_[tile_j] = upcxx::broadcast(top_most_row_[tile_j], owner).wait();
    
    owner = owner_tile(tiles_.first - 1, tile_j);
    if (imyrank == owner) {
      bottom_most_row_[tile_j] = upcxx::new_array<Row>(1);
    }
    bottom_most_row_[tile_j] = upcxx::broadcast(bottom_most_row_[tile_j], owner).wait();
    
  }
  
  
  for (int tile_i = 0; tile_i < tiles_.first; tile_i++) {
    
    auto owner = owner_tile(tile_i, 0);
    if (imyrank == owner) {
      left_most_row_[tile_i] = upcxx::new_array<Row>(1);
    }
    left_most_row_[tile_i] = upcxx::broadcast(left_most_row_[tile_i], owner).wait();

    owner = owner_tile(tile_i, tiles_.second - 1);
    if (imyrank == owner) {
      right_most_row_[tile_i] = upcxx::new_array<Row>(1);
    }
    right_most_row_[tile_i] = upcxx::broadcast(right_most_row_[tile_i], owner).wait();
    
  }

  upcxx::barrier();

  fill_in();

//  if (!imyrank) {
//    fprintf(stderr, "00=%d %p\n", get_FullTile(0, 0).tile_ptr_.where(), get_FullTile(0, 0).tile_ptr_.local());
//    fprintf(stderr, "10=%d %p\n", get_FullTile(1, 0).tile_ptr_.where(), get_FullTile(1, 0).tile_ptr_.local());
//    fprintf(stderr, "01=%d %p\n", get_FullTile(0, 1).tile_ptr_.where(), get_FullTile(0, 1).tile_ptr_.local());
//    fprintf(stderr, "20=%d %p\n", get_FullTile(2, 0).tile_ptr_.where(), get_FullTile(2, 0).tile_ptr_.local());
//    fprintf(stderr, "11=%d %p\n", get_FullTile(1, 1).tile_ptr_.where(), get_FullTile(1, 1).tile_ptr_.local());
//    fprintf(stderr, "02=%d %p\n", get_FullTile(0, 2).tile_ptr_.where(), get_FullTile(0, 2).tile_ptr_.local());
//  }
  
}

void WaveFrontProbl::deallocate()
{
  for (int tile_i = 0; tile_i < tiles_.first; tile_i++) {
    for (int tile_j = 0; tile_j < tiles_.second; tile_j++) {
      const auto owner = owner_tile(tile_i, tile_j);
      if (imyrank == owner) {
        FullTile& full_tile = tiles_arr_[tile_i * tiles_.second + tile_j];
        upcxx::delete_array(full_tile.tile_ptr_);
        upcxx::delete_array(full_tile.left_ptr_);
      }
    }
  }
  
  delete [] tiles_arr_;
  //fprintf(stdout, " ---- %d exit ----\n", imyrank);

  for(int tile_j = 0; tile_j < tiles_.second; tile_j++) {
    if (top_most_row_[tile_j].where() == imyrank) {
      upcxx::delete_array(top_most_row_[tile_j]);
    }
    if (bottom_most_row_[tile_j].where() == imyrank) {
      upcxx::delete_array(bottom_most_row_[tile_j]);
    }
  }
  
  for (int tile_i = 0; tile_i < tiles_.first; tile_i++) {
    if (left_most_row_[tile_i].where() == imyrank) {
      upcxx::delete_array(left_most_row_[tile_i]);
    }
    if (right_most_row_[tile_i].where() == imyrank) {
      upcxx::delete_array(right_most_row_[tile_i]);
    }
  }

  delete [] top_most_row_;
  delete [] bottom_most_row_;
  delete [] left_most_row_;
  delete [] right_most_row_;

  upcxx::barrier();
}

void WaveFrontProbl::fill_in()
{
  for (int tile_j = 0; tile_j < tiles_.second; tile_j++) {
    if (top_most_row_[tile_j].where() == imyrank) {
      memset(top_most_row_[tile_j].local(), 0, sizeof(Row));
    }
    if (bottom_most_row_[tile_j].where() == imyrank) {
      memset(bottom_most_row_[tile_j].local(), 0, sizeof(Row));
    }
  }
  
  for (int tile_i = 0; tile_i < tiles_.first; tile_i++) {
    if (left_most_row_[tile_i].where() == imyrank) {
      memset(left_most_row_[tile_i].local(), 0, sizeof(Row));
    }
    if (right_most_row_[tile_i].where() == imyrank) {
      memset(right_most_row_[tile_i].local(), 0, sizeof(Row));
    }
    for (int tile_j = 0; tile_j < tiles_.second; tile_j++) {
      const auto& tmp = tiles_arr_[tile_i * tiles_.second + tile_j];
      if (tmp.tile_ptr_.where() == imyrank) {
        Row * const my_ptr = (Row *)(tmp.tile_ptr_.local());
        Row& my_left_col= *(tmp.left_ptr_.local());
        Row& my_right_col = *(tmp.right_ptr_.local());
        // initialize local tile
        for (int row = 1; row <= TILE; row++) { // skip top halo
          for (int j = 1; j <= TILE; j++) {     // skip left halo
            my_ptr[row][j] = static_cast<double>((tile_i * TILE + row) * cols_ + tile_j * TILE + j);
          }
          my_left_col[row] = my_ptr[row][1]; // Skip left halo
          my_right_col[row] = my_ptr[row][TILE]; // Skip left halo
        }
      }
    }
  }
  
  upcxx::barrier();
}

void compute_bck(Tile& wk_tile, int tile_i, int tile_j, const int bck_sz)
{
  //fprintf(stderr,"(%d,%d)\n", tile_i, tile_j);
  
  const int begin_i = 1 + bck_sz * tile_i;
  const int begin_j = 1 + bck_sz * tile_j;
  const int end_i = std::min(TILE + 1, begin_i + bck_sz);
  const int end_j = std::min(TILE + 1, begin_j + bck_sz);
  
  for(int i = begin_i; i < end_i; i++) {
    for(int j = begin_j; j < end_j; j++) {
      wk_tile[i][j] = (4. * wk_tile[i][j] + wk_tile[i-1][j] + wk_tile[i+1][j] + wk_tile[i][j-1] + wk_tile[i][j+1]) / 8.;
    }
  }
}

void async_bck(Tile& wk_tile, int tile_i, int tile_j,
               const int bck_sz,
               std::atomic<int> * const markers)
{
  auto& tp = depspawn::get_task_pool();
  compute_bck(wk_tile, tile_i, tile_j, bck_sz);
  
  if (tile_j < (SubProblems - 1)) { // move east
    if (!tile_i) {
      tp.hp_enqueue([&wk_tile, tile_i, tile_j, bck_sz, markers] { async_bck(wk_tile, 0, tile_j + 1, bck_sz, markers); });
    } else {
      if ((++markers[tile_i * SubProblems + tile_j + 1]) == 2) {
        tp.hp_enqueue([&wk_tile, tile_i, tile_j, bck_sz, markers] { async_bck(wk_tile, tile_i, tile_j + 1, bck_sz, markers); });
      }
    }
  } else {
    if (tile_i == (SubProblems - 1)) { // Mark end of tasks
      markers[tile_i * SubProblems + tile_j].store(-1, std::memory_order_relaxed);
    }
  }
  
  if (tile_i < (SubProblems - 1)) { // move south
    if (!tile_j) {
      tp.hp_enqueue([&wk_tile, tile_i, tile_j, bck_sz, markers] { async_bck(wk_tile, tile_i + 1, 0, bck_sz, markers); });
    } else {
      if ((++markers[(tile_i + 1) * SubProblems + tile_j]) == 2) {
        tp.hp_enqueue([&wk_tile, tile_i, tile_j, bck_sz, markers] { async_bck(wk_tile, tile_i + 1, tile_j, bck_sz, markers); });
      }
    }
  }

}
    
void basic_block_gauss(global_ptr<const Row> top_row, global_ptr<const Row> bottom_row,
                       global_ptr<const Row> left_col, global_ptr<const Row> right_col,
                       global_ptr<Tile> tile,
                       global_ptr<Row> my_top_row_buf, global_ptr<Row> my_bottom_row_buf,
                       global_ptr<Row> my_left_col_buf, global_ptr<Row> my_right_col_buf)
{
  const bool local_top_row = (top_row.where() == imyrank);
  const bool local_bottom_row = (bottom_row.where() == imyrank);
  const bool local_left_col = (left_col.where() == imyrank);
  const bool local_right_col = (right_col.where() == imyrank);

  Tile& wk_tile = *(tile.local());

  //fprintf(stderr, "CP=%p\n", &upcxx::current_persona()); //sometimes it is nullptr
  //upcxx::persona my_persona;
  //upcxx::persona_scope scope(my_persona);
  
  //fprintf(stderr,"P %d %p\n", tile.where(), tile.local());

  Row in_left_col, in_right_col;
  const Row *in_left_col_ptr, *in_right_col_ptr;
  upcxx::future<> copy_top_event, copy_bottom_event, copy_left_event, copy_right_event;
  
  // Request remote copies
  if (!local_top_row) {
    copy_top_event = upcxx::rget(top_row, reinterpret_cast<Row *>(&wk_tile), 1);
//    upcxx::async_copy(upcxx::global_ptr<void>((void *)top_row.local(), top_row.where()),
//                      upcxx::global_ptr<void>((void *)(&wk_tile), imyrank),
//                      sizeof(Row), &copy_top_event);
  }
  
  if (!local_bottom_row) {
    copy_bottom_event = upcxx::rget(bottom_row, &(wk_tile[TILE + 1]), 1);
//    upcxx::async_copy(upcxx::global_ptr<void>((void *)bottom_row.local(), bottom_row.where()),
//                      upcxx::global_ptr<void>((void *)(&(wk_tile[TILE + 1])), imyrank),
//                      sizeof(Row), &copy_bottom_event);
  }
  
  if (!local_left_col) {
    in_left_col_ptr = &in_left_col;
    copy_left_event = upcxx::rget(left_col, &in_left_col, 1);
//    upcxx::async_copy(upcxx::global_ptr<void>((void *)left_col.local(), left_col.where()),
//                      upcxx::global_ptr<void>((void *)&in_left_col, imyrank),
//                      sizeof(Row), &copy_left_event);
  } else {
    in_left_col_ptr = left_col.local();
  }
    
  if (!local_right_col) {
    in_right_col_ptr = &in_right_col;
    copy_right_event = upcxx::rget(right_col, &in_right_col, 1);
//    upcxx::async_copy(upcxx::global_ptr<void>((void *)right_col.local(), right_col.where()),
//                      upcxx::global_ptr<void>((void *)&in_right_col, imyrank),
//                      sizeof(Row), &copy_right_event);
  } else {
    in_right_col_ptr = right_col.local();
  }
  
  // Wait for copies / perform local copies to halo regions
  if (!local_top_row) {
    copy_top_event.wait();
  } else {
    memcpy(&wk_tile, top_row.local(), sizeof(Row));
  }
  
  if (!local_bottom_row) {
    copy_bottom_event.wait();
  } else {
    memcpy(&(wk_tile[TILE + 1]), bottom_row.local(), sizeof(Row));
  }
  
  if (!local_left_col) {
    copy_left_event.wait();
  }

  assert((*in_left_col_ptr)[0] == wk_tile[0][0]); // must already come from top if wavefront goes NW->SE
  for(int i = 1; i <= (TILE + 1); i++) {
    wk_tile[i][0] = (*in_left_col_ptr)[i];
  }
  
  if (!local_right_col) {
    copy_right_event.wait();
  }
  
  for(int i = 1; i <= (TILE + 1); i++) {
    wk_tile[i][TILE+1] = (*in_right_col_ptr)[i];
  }

  // Compute and update owned halos for other tiles
  Row& my_left_col= *(my_left_col_buf.local());
  Row& my_right_col = *(my_right_col_buf.local());

  if (SubProblems == 1) {
    for(int i = 1; i <= TILE; i++) {
      for(int j = 1; j <= TILE; j++) {
        //fprintf(stderr,"%p %d %d -> %p %d %d\n", &wk_tile, i, j, &(wk_tile[i][j]), &(wk_tile[i][j]) - &(wk_tile[0][0]), TILE);
        wk_tile[i][j] = (4. * wk_tile[i][j] + wk_tile[i-1][j] + wk_tile[i+1][j] + wk_tile[i][j-1] + wk_tile[i][j+1]) / 8.;
      }
      my_left_col[i] = wk_tile[i][1];
      my_right_col[i] = wk_tile[i][TILE];
    }
  } else {
    auto& tp = depspawn::get_task_pool();
    
    const int bck_sz = (TILE + SubProblems - 1) / SubProblems;
    
    compute_bck(wk_tile, 0, 0, bck_sz);
    
    std::atomic<int> markers[SubProblems * SubProblems];
    for (int i = 0; i < SubProblems * SubProblems; i++) {
      markers[i].store(0);
    }
    
    tp.hp_enqueue([&, bck_sz] { async_bck(wk_tile, 0, 1, bck_sz, markers); });
    tp.hp_enqueue([&, bck_sz] { async_bck(wk_tile, 1, 0, bck_sz, markers); });
    
    while (markers[SubProblems * SubProblems - 1].load(std::memory_order_relaxed) != -1) {
      tp.hp_try_run();
    }

    for(int i = 1; i <= TILE; i++) {
      my_left_col[i] = wk_tile[i][1];
      my_right_col[i] = wk_tile[i][TILE];
    }
  }
  

  my_left_col[0] = wk_tile[0][1];
  my_right_col[0] = wk_tile[0][TILE];
}

upcxx::future<> basic_block_gauss1(WaveFrontProbl * const p, const int tile_i, const int tile_j, Row *& in_left_col_ptr, Row *& in_right_col_ptr)
{
  const global_ptr<const Row> top_row = p->get_top_for(tile_i, tile_j);       //in
  const global_ptr<const Row> bottom_row = p->get_bottom_for(tile_i, tile_j); //in
  const global_ptr<const Row> left_col = p->get_left_for(tile_i, tile_j);     //in
  const global_ptr<const Row> right_col = p->get_right_for(tile_i, tile_j);   //in
  const global_ptr<Tile> tile = p->get_tile(tile_i, tile_j);

  const bool local_top_row = (top_row.where() == imyrank);
  const bool local_bottom_row = (bottom_row.where() == imyrank);
  const bool local_left_col = (left_col.where() == imyrank);
  const bool local_right_col = (right_col.where() == imyrank);

  Tile& wk_tile = *(tile.local());

  upcxx::future<> ret;

  // Request remote copies
  if (!local_top_row) {
    //fprintf(stderr, "P%d %d %d ReqN %d\n", imyrank, tile_i, tile_j, top_row.where());
    ret = upcxx::rget(top_row, reinterpret_cast<Row *>(&wk_tile), 1);
  } else {
    ret = upcxx::make_future();
  }
  
  if (!local_bottom_row) {
    //fprintf(stderr, "P%d %d %d ReqS %d\n", imyrank, tile_i, tile_j, bottom_row.where());
    ret = upcxx::when_all(ret, upcxx::rget(bottom_row, &(wk_tile[TILE + 1]), 1));
  }
  
  if (!local_left_col) {
    //fprintf(stderr, "P%d %d %d ReqW %d\n", imyrank, tile_i, tile_j, left_col.where());
    in_left_col_ptr = new Row();
    ret = upcxx::when_all(ret, upcxx::rget(left_col, in_left_col_ptr, 1));
  } else {
    in_left_col_ptr = const_cast<Row*>(left_col.local());
  }
    
  if (!local_right_col) {
    //fprintf(stderr, "P%d %d %d ReqE %d\n", imyrank, tile_i, tile_j, right_col.where());
    in_right_col_ptr = new Row();
    //auto q = upcxx::rget(right_col, in_right_col_ptr, 1);
    //q.then([=]{fprintf(stderr, "P%d %d %d ReqE %d was read\n", imyrank, tile_i, tile_j, right_col.where());});
    //upcxx::rpc(right_col.where(),[tile_i,tile_j,right_col]{
    //  fprintf(stderr, "P%d got req %d %d -> %p %lf %lf %lf\n", imyrank, tile_i, tile_j, right_col.local(), right_col.local()->val[0], right_col.local()->val[1], right_col.local()->val[2]);
    //}).then([=]{fprintf(stderr, "P%d %d %d Remote RPC P%d was run\n", imyrank, tile_i, tile_j, right_col.where());});
    ret = upcxx::when_all(ret, upcxx::rget(right_col, in_right_col_ptr, 1));
  } else {
    in_right_col_ptr = const_cast<Row*>(right_col.local());
  }

  //std::cerr << 'P' << imyrank << ' ' << std::this_thread::get_id() << '\n';
  //ret.then([=]{fprintf(stderr, "Completed %d %d\n", tile_i, tile_j);});
  //fprintf(stderr, "%d %d %d <- T%d B%d L%d R%d\n", imyrank, tile_i, tile_j, top_row.where(), bottom_row.where(), left_col.where(), right_col.where());

  return ret;

}

void basic_block_gauss2(WaveFrontProbl * const p, const int tile_i, const int tile_j,  Row * const in_left_col_ptr, Row * const in_right_col_ptr)
{
  const global_ptr<const Row> top_row = p->get_top_for(tile_i, tile_j);       //in
  const global_ptr<const Row> bottom_row = p->get_bottom_for(tile_i, tile_j); //in
  const global_ptr<const Row> left_col = p->get_left_for(tile_i, tile_j);     //in
  const global_ptr<const Row> right_col = p->get_right_for(tile_i, tile_j);   //in
  const global_ptr<Tile> tile = p->get_tile(tile_i, tile_j);                  //in-out
  //global_ptr<Row> my_top_row_buf = p->get_top(tile_i, tile_j);         //in-out
  //global_ptr<Row> my_bottom_row_buf = p->get_bottom(tile_i, tile_j);   //in-out
  const global_ptr<Row> my_left_col_buf = p->get_left(tile_i, tile_j);       //in-out
  const global_ptr<Row> my_right_col_buf = p->get_right(tile_i, tile_j);     //in-out
  
  Tile& wk_tile = *(tile.local());

  // perform local copies to halo regions
  const bool local_top_row = (top_row.where() == imyrank);
  if(local_top_row) {
    memcpy(&wk_tile, top_row.local(), sizeof(Row));
  }

  const bool local_bottom_row = (bottom_row.where() == imyrank);
  if(local_bottom_row) {
    memcpy(&(wk_tile[TILE + 1]), bottom_row.local(), sizeof(Row));
  }

  assert((*in_left_col_ptr)[0] == wk_tile[0][0]); // must already come from top if wavefront goes NW->SE
  for(int i = 1; i <= (TILE + 1); i++) {
    wk_tile[i][0] = (*in_left_col_ptr)[i];
  }

  for(int i = 1; i <= (TILE + 1); i++) {
    wk_tile[i][TILE+1] = (*in_right_col_ptr)[i];
  }

  const bool local_left_col = (left_col.where() == imyrank);
  if (!local_left_col) {
    delete in_left_col_ptr;
  }

  const bool local_right_col = (right_col.where() == imyrank);
  if (!local_right_col) {
    delete in_right_col_ptr;
  }

  // Compute and update owned halos for other tiles
  Row& my_left_col= *(my_left_col_buf.local());
  Row& my_right_col = *(my_right_col_buf.local());

  if (SubProblems == 1) {
    for(int i = 1; i <= TILE; i++) {
      for(int j = 1; j <= TILE; j++) {
        //fprintf(stderr,"%p %d %d -> %p %d %d\n", &wk_tile, i, j, &(wk_tile[i][j]), &(wk_tile[i][j]) - &(wk_tile[0][0]), TILE);
        wk_tile[i][j] = (4. * wk_tile[i][j] + wk_tile[i-1][j] + wk_tile[i+1][j] + wk_tile[i][j-1] + wk_tile[i][j+1]) / 8.;
      }
      my_left_col[i] = wk_tile[i][1];
      my_right_col[i] = wk_tile[i][TILE];
    }
  } else {
    auto& tp = depspawn::get_task_pool();
    
    const int bck_sz = (TILE + SubProblems - 1) / SubProblems;
    
    compute_bck(wk_tile, 0, 0, bck_sz);
    
    std::atomic<int> markers[SubProblems * SubProblems];
    for (int i = 0; i < SubProblems * SubProblems; i++) {
      markers[i].store(0);
    }
    
    tp.hp_enqueue([&, bck_sz] { async_bck(wk_tile, 0, 1, bck_sz, markers); });
    tp.hp_enqueue([&, bck_sz] { async_bck(wk_tile, 1, 0, bck_sz, markers); });
    
    while (markers[SubProblems * SubProblems - 1].load(std::memory_order_relaxed) != -1) {
      tp.hp_try_run();
    }

    for(int i = 1; i <= TILE; i++) {
      my_left_col[i] = wk_tile[i][1];
      my_right_col[i] = wk_tile[i][TILE];
    }
  }
  

  my_left_col[0] = wk_tile[0][1];
  my_right_col[0] = wk_tile[0][TILE];
}

// Equivalent to basic_block_gauss
void basic_block_gauss12(WaveFrontProbl * const p, const int tile_i, const int tile_j)
{ Row *in_left_col_ptr, *in_right_col_ptr;
  
  basic_block_gauss1(p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr).wait();
  basic_block_gauss2(p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr);
}

// Always in RPC -> always run in main thread
upcxx::future<> submit_basic_block_gauss(upcxx::dist_object<WaveFrontProbl>& obj, const int tile_i, const int tile_j)
{ Row *in_left_col_ptr, *in_right_col_ptr;

  upcxx::promise<> * const promise_ptr = new upcxx::promise<>();
  upcxx::future<> result = promise_ptr->get_future();
  
  WaveFrontProbl * const p = &(*obj);
  Num_live_tasks.fetch_add(1);
  //basic_block_gauss is broken in two because we cannot make wait() inside progress()
  upcxx::future<> res = basic_block_gauss1(p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr);

  res.then([promise_ptr, p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr]{
    depspawn::get_task_pool().enqueue([promise_ptr, p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr] {
      basic_block_gauss2(p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr);
      promise_ptr->fulfill_anonymous(1);
      Num_live_tasks.fetch_sub(1);
      //try_upcxx_progress();
      delete promise_ptr;
    });
  });

  return result;
}

void notify_finish(upcxx::dist_object<WaveFrontProbl>& obj)
{
  obj->incr_finish();
}

double WaveFrontProbl::dsp_block_gauss()
{ int start, end;
  double time;

  assert(tiles_.first == tiles_.second);
  
  upcxx::barrier();

  disable_ps_master();
  auto t0 = std::chrono::high_resolution_clock::now();
  
  const int numblocks = tiles_.first;

  for (int nr = 0; nr < NUM_ITERS; nr++) {
    
    for (int j = 1; j < 2 * numblocks; j++) {

      if ( j <= numblocks ) {
        start = 0;
        end = j;
      }
      else {
        start = j - numblocks;
        end = numblocks;
      }
      
      for (int tile_i = start; tile_i < end; tile_i++) {
        const int tile_j = j - tile_i - 1;
        depspawn::upcxx_spawn(basic_block_gauss,
                    get_top_for(tile_i, tile_j), get_bottom_for(tile_i, tile_j), //in
                    get_left_for(tile_i, tile_j), get_right_for(tile_i, tile_j), //in
                    get_tile(tile_i, tile_j),                                    //in-out
                    get_top(tile_i, tile_j), get_bottom(tile_i, tile_j),         //in-out
                    get_left(tile_i, tile_j), get_right(tile_i, tile_j));        //in-out
      }
    }

  }
  
  depspawn::upcxx_wait_for_all();
  
  auto t1 = std::chrono::high_resolution_clock::now();
  enable_ps_master();
  
  if (!imyrank)
  {
    time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " Total Time: " << time << " sec" << std::endl;
  }

  return time;
}

double WaveFrontProbl::dsp_filtered_block_gauss()
{ int start, end;
  double time;

  assert(tiles_.first == tiles_.second);
  
  upcxx::barrier();

  disable_ps_master();
  auto t0 = std::chrono::high_resolution_clock::now();

  const int numblocks = tiles_.first;

  for (int nr = 0; nr < NUM_ITERS; nr++) {
    
    for (int j = 1; j < 2 * numblocks; j++) {

      if ( j <= numblocks ) {
        start = 0;
        end = j;
      }
      else {
        start = j - numblocks;
        end = numblocks;
      }

      for (int tile_i = start; tile_i < end; tile_i++) {
        const int tile_j = j - tile_i - 1;
        
        depspawn::upcxx_cond_spawn(   (imyrank == owner_tile(tile_i, tile_j))
                         || (tile_i && (imyrank == owner_tile(tile_i - 1, tile_j)))
                         || ((tile_i < (tiles_.first - 1)) && (imyrank == owner_tile(tile_i + 1, tile_j)))
                         || (tile_j && (imyrank == owner_tile(tile_i, tile_j - 1)))
                         || ((tile_j < (tiles_.second - 1)) && (imyrank == owner_tile(tile_i, tile_j + 1))),
                         basic_block_gauss,
                         get_top_for(tile_i, tile_j), get_bottom_for(tile_i, tile_j), //in
                         get_left_for(tile_i, tile_j), get_right_for(tile_i, tile_j), //in
                         get_tile(tile_i, tile_j),                                    //in-out
                         get_top(tile_i, tile_j), get_bottom(tile_i, tile_j),         //in-out
                         get_left(tile_i, tile_j), get_right(tile_i, tile_j));        //in-out
      }
    }

  }
  
  depspawn::upcxx_wait_for_all();
  
  auto t1 = std::chrono::high_resolution_clock::now();
  enable_ps_master();
  
  if (!imyrank)
  {
    time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " Total Time: " << time << " sec" << std::endl;
  }

  return time;
}

double WaveFrontProbl::future_block_gauss()
{ int start, end;
  double time;
  upcxx::future<> * gauss_futures;

  assert(tiles_.first == tiles_.second);
  
  assert(!Num_live_tasks.load());

  upcxx::barrier();
  
  auto t0 = std::chrono::high_resolution_clock::now();
  
  finish_ = 0;

  if (!imyrank) {
    
    const int numblocks = tiles_.first;
    const int matrix_blocks = numblocks * numblocks;
    upcxx::future<> ready_future = upcxx::make_future();
    gauss_futures = new upcxx::future<>[NUM_ITERS * matrix_blocks];
    
    for (int nr = 0; nr < NUM_ITERS; nr++) {
      
      for (int j = 1; j < 2 * numblocks; j++) {
        
        if ( j <= numblocks ) {
          start = 0;
          end = j;
        }
        else {
          start = j - numblocks;
          end = numblocks;
        }
        
        for (int tile_i = start; tile_i < end; tile_i++) {

          const int tile_j = j - tile_i - 1;
          
          upcxx::future<>& left_dep = tile_j
          ? gauss_futures[nr * matrix_blocks + tile_i * numblocks + tile_j - 1]
          : ready_future;
          
          upcxx::future<>& up_dep = tile_i
          ? gauss_futures[nr * matrix_blocks + (tile_i - 1) * numblocks + tile_j]
          : ready_future;
          
          upcxx::future<>& prev_right_completed = (nr && (tile_j != (numblocks - 1)))
          ? gauss_futures[(nr - 1)* matrix_blocks + tile_i * numblocks + tile_j + 1]
          : ready_future;
          
          upcxx::future<>& prev_down_completed = (nr && (tile_i != (numblocks - 1)))
          ? gauss_futures[(nr - 1)* matrix_blocks + (tile_i + 1) * numblocks + tile_j]
          : ready_future;
          
          gauss_futures[nr * matrix_blocks + tile_i * numblocks + tile_j] = upcxx::when_all(left_dep, up_dep, prev_right_completed, prev_down_completed).then([tile_i, tile_j, this] {
            const upcxx::intrank_t owner = get_tile(tile_i, tile_j).where();
            return upcxx::rpc(owner, submit_basic_block_gauss, *self_dist_object_, tile_i, tile_j);
          });
          
        }
      }
    }
    
    gauss_futures[(NUM_ITERS - 1) * matrix_blocks + (numblocks - 1) * numblocks + (numblocks - 1)].then([&]{
      for (int i = 1; i < NRanks; i++) {
        upcxx::rpc_ff(i, notify_finish, *self_dist_object_);
      }
      finish_ = 1;
    });

  }
  
  while (!finish_) {
    while ( !finish_ && (Num_live_tasks.load() < nthreads) ) {
      try_upcxx_progress();
    }
    depspawn::get_task_pool().try_run();
  }
  
  auto t1 = std::chrono::high_resolution_clock::now();
  
  if (!imyrank)
  {
    time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " Total Time: " << time << " sec" << std::endl;
    delete [] gauss_futures;
  }
  
  return time;
}

double WaveFrontProbl::barrier_block_gauss()
{ int start, end;
  double time;

  assert(tiles_.first == tiles_.second);
  
  assert(!Num_live_tasks.load());

  auto& tp = depspawn::get_task_pool();

  upcxx::barrier();
  
  auto t0 = std::chrono::high_resolution_clock::now();

  //Num_live_tasks.store(0);
  
  const int numblocks = tiles_.first;

  for (int nr = 0; nr < NUM_ITERS; nr++) {
    
    for (int j = 1; j < 2 * numblocks; j++) {
      
      if ( j <= numblocks ) {
        start = 0;
        end = j;
      }
      else {
        start = j - numblocks;
        end = numblocks;
      }
      
      bool finish_submit = false;
      
      for (int tile_i = start; tile_i < end; tile_i++) {
        const int tile_j = j - tile_i - 1;
        const auto tile_ref = get_tile(tile_i, tile_j);
        if (tile_ref.where() == imyrank) {
          //Num_live_tasks.fetch_add(1);
          tp.enqueue([&, tile_i, tile_j, tile_ref] {
            basic_block_gauss(get_top_for(tile_i, tile_j), get_bottom_for(tile_i, tile_j), //in
                              get_left_for(tile_i, tile_j), get_right_for(tile_i, tile_j), //in
                              tile_ref,                                                    //in-out
                              get_top(tile_i, tile_j), get_bottom(tile_i, tile_j),         //in-out
                              get_left(tile_i, tile_j), get_right(tile_i, tile_j));        //in-out
            //try_upcxx_progress();
            //Num_live_tasks.fetch_sub(1);
          });
        }
      }

      finish_submit = true;
      
      tp.wait();
      upcxx::barrier();
    }
  }
  
  auto t1 = std::chrono::high_resolution_clock::now();
  
  if (!imyrank)
  {
    time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " Total Time: " << time << " sec" << std::endl;
  }
  
  return time;
}

/* async_block_gauss begins */
    
void notify_run(upcxx::dist_object<WaveFrontProbl>& obj, int tile_i, int tile_j);

void complete_run_async_tile(upcxx::dist_object<WaveFrontProbl>& obj, int tile_i, int tile_j, Row * const in_left_col_ptr, Row * const in_right_col_ptr)
{
  WaveFrontProbl * const p = &(*obj);

  //fprintf(stderr, "%d C %d %d\n", imyrank, tile_i, tile_j);
  basic_block_gauss2(p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr);

  bool is_last = true;
  const int numblocks_end = p->get_tiles().first - 1;
  
  if(tile_i < numblocks_end) {
    const auto dest = p->get_tile(tile_i + 1, tile_j).where();
    if (dest == imyrank) {
      notify_run(obj, tile_i + 1, tile_j);
    } else {
      upcxx::rpc_ff(dest, notify_run, obj, tile_i + 1, tile_j);
    }
    is_last = false;
  }
  
  if(tile_j < numblocks_end) {
    const auto dest = p->get_tile(tile_i, tile_j + 1).where();
    if (dest == imyrank) {
      notify_run(obj, tile_i, tile_j + 1);
    } else {
      upcxx::rpc_ff(dest, notify_run, obj, tile_i, tile_j + 1);
    }
    is_last = false;
  }
  
  if (is_last) {
    for (uint32_t i = 0; i < NRanks; i++) {
      if(i != imyrank) {
        upcxx::rpc_ff(i, notify_finish, obj);
      }
    }
    notify_finish(obj);
  }
}

//#define DEBUG
#ifdef DEBUG
struct DBG { int i, j, val; bool incr; };
std::vector<DBG> DBG_v(10000);
std::atomic<int> DBG_ctr{0};

void dump_DBG(const int limit)
{
  for (int i = 0; i <= limit; i++) {
    fprintf(stderr, "P%d %d %d %d %c\n", imyrank, DBG_v[i].i, DBG_v[i].j, DBG_v[i].val, DBG_v[i].incr ? '+' : '-');
  }
}
#endif

void begin_run_async_tile(upcxx::dist_object<WaveFrontProbl>& obj, const int tile_i, const int tile_j)
{ Row *in_left_col_ptr, *in_right_col_ptr;

  //fprintf(stderr, "%d B %d %d\n", imyrank, tile_i, tile_j);
  WaveFrontProbl * const p = &(*obj);

  //basic_block_gauss is broken in two because we cannot make wait() inside progress()
  upcxx::future<> res = basic_block_gauss1(p, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr);
  
  const auto cont_f = [&obj, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr]{
    //fprintf(stderr, "%d S %d %d\n", imyrank, tile_i, tile_j);
#ifndef DEBUG
    Num_live_tasks.fetch_add(1);
#else
    const int tmp = DBG_ctr.fetch_add(1);
    DBG_v[tmp]= {tile_i, tile_j, Num_live_tasks.fetch_add(1) + 1, true};
    if (tmp && (DBG_v[tmp-1].i >= tile_i) && (DBG_v[tmp-1].j >= tile_j)) {
      dump_DBG(tmp);
      fprintf(stderr, "P%d jumps from %d %d to %d %d f=%d\n", imyrank, DBG_v[tmp-1].i, DBG_v[tmp-1].j, tile_i, tile_j, obj->get_finish());
      //abort();
    }
#endif
    
    depspawn::get_task_pool().enqueue([&obj, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr] {
      complete_run_async_tile(obj, tile_i, tile_j, in_left_col_ptr, in_right_col_ptr);
#ifndef DEBUG
      Num_live_tasks.fetch_sub(1);
#else
      const int tmp = Num_live_tasks.fetch_sub(1);
      const int tmp2 = DBG_ctr++;
      DBG_v[tmp2]= {tile_i, tile_j, tmp - 1, false};
      int i;
      for (i = 0; (i <= tmp2) && ((DBG_v[i].i!=tile_i) || (DBG_v[i].j!=tile_j) || !DBG_v[i].incr); i++);
      if (i > tmp2) {
        dump_DBG(tmp2);
        fprintf(stderr, "P%d where is %d %d +?\n", imyrank, tile_i, tile_j);
        abort();
      }
      if (!tmp) {
        dump_DBG(tmp2);
        fprintf(stderr, "P%d %d %d -1!\n", imyrank, tile_i, tile_j);
        abort();
      }
#endif
    });
  };

  //fprintf(stderr,"P%d R=%d in_P=%d\n", imyrank, res.ready(), upcxx::in_progress());

  if(!upcxx::in_progress() && !upcxx::master_persona().active_with_caller()) {
    res.wait();
    cont_f();
  } else {
    res.then(cont_f);
  }
  
}

void notify_run(upcxx::dist_object<WaveFrontProbl>& obj, int tile_i, int tile_j)
{ //static std::mutex Tile_counter_mutex; //could break with std::system_error
  int counter;

  WaveFrontProbl * const p = &(*obj);
  try {
    std::lock_guard<std::mutex> guard(Tile_counter_mutex);
    counter = p->inc_tile_counter(tile_i, tile_j);
  } catch(const std::system_error& e) {
      std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
  } catch (...) {
    std::cerr << "Uknown exception occurred!\n";
  }

  if ((counter == 2) || !tile_i || !tile_j) {
    p->set_tile_counter(tile_i, tile_j, 0);
    begin_run_async_tile(obj, tile_i, tile_j);
  }
}

double WaveFrontProbl::async_block_gauss()
{ double time;

  upcxx::barrier();

  assert(tiles_.first == tiles_.second);

  assert(!Num_live_tasks.load());

  upcxx::barrier();
  
  auto t0 = std::chrono::high_resolution_clock::now();
  
  const int numblocks = tiles_.first;
  tile_counter_ = new int[numblocks * numblocks];
  memset(tile_counter_, 0, numblocks * numblocks * sizeof(int));

  for (int nr = 0; nr < NUM_ITERS; nr++) {

    // There can be live tasks in the pool arrived during the barrier
#ifdef DEBUG
//    int kk;
//    if ((kk=Num_live_tasks.load())) {
//      fprintf(stderr, "P%d it %d not 0! is %d\n", imyrank, nr, kk);
//    }
//    if (!depspawn::get_task_pool().empty()) {
//      fprintf(stderr, "P%d it %d : !depspawn::get_task_pool().empty()\n", imyrank, nr);
//    }
#endif

    finish_ = 0;

    if (get_tile(0, 0).where() == imyrank) {
      begin_run_async_tile(*self_dist_object_, 0, 0);
    }

    //Num_live_tasks is decreased after the local notify_finish, so we must make sure
    //it is also zero before we leave
    while (!finish_ || Num_live_tasks.load()) {
      while ( !finish_ && (Num_live_tasks.load() < nthreads) ) {
        try_upcxx_progress();
      }
      depspawn::get_task_pool().try_run();
    }

#ifdef DEBUG
    DBG_ctr.store(0);
#endif
    
    //upcxx::barrier();
    if(imyrank) { // seems safer than barrier
      upcxx::rpc_ff(0, notify_finish, *self_dist_object_);
    } else {
      while (finish_ < NRanks) {
        upcxx::progress();
      }
    }

  }
  
  delete [] tile_counter_;

  auto t1 = std::chrono::high_resolution_clock::now();
  
  if (!imyrank)
  {
    time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " Total Time: " << time << " sec" << std::endl;
  }
  
  return time;
}
/* async_block_gauss ends */

// Must be called by all the processes
bool WaveFrontProbl::equal_arr(WaveFrontProbl& other) const
{ int tile_i, tile_j, same_local = 1;
  const Tile *a, *b;

  upcxx::barrier();
  
  for (tile_i = 0; tile_i < tiles_.first; tile_i++) {
    for (tile_j = 0; tile_j < tiles_.second; tile_j++) {
      const auto tile_ref = get_tile(tile_i, tile_j);
      if (tile_ref.where() == imyrank) {
        assert(other.get_tile(tile_i, tile_j).where() == imyrank);
        a = tile_ref.local();
        b = other.get_tile(tile_i, tile_j).local();
        same_local = !memcmp(a, b, sizeof(Tile));
        if (!same_local) {
          break;
        }
      }
    }
    if (!same_local) {
      break;
    }
  }
  
  if (!same_local) {
    std::cerr << imyrank << " != at tile (" << tile_i << ", " << tile_j << ")\n";
    for (int i = 0 ; i < (TILE + 2); i++) {
      for (int j = 0 ; j < (TILE + 2); j++) {
        if ((*a)[i][j] != (*b)[i][j]) {
          std::cerr << (*a)[i][j] << ' ' << (*b)[i][j] << " at [" << i << ", " << j << "]\n";
          exit(-1);
        }
      }
    }
  }

  return upcxx::reduce_all(same_local, upcxx::op_fast_mul).wait();
}


int arg_parse(int argc, char **argv)
{ int ch;
  
  while ( -1 != (ch = getopt(argc, argv,"B:Cc:D:i:N:n:Q:r:T")) ) {
    switch (ch) {
      case 'B':
        Baseline_Version = (int)strtoul(optarg, NULL, 0);
        if ((Baseline_Version < 0) || (Baseline_Version > MaxVersion)) {
          std::cerr << "-B not in [0," << MaxVersion << "]. Value=" << Baseline_Version << std::endl;
          exit( EXIT_FAILURE );
        }
        Versions.push_back(Baseline_Version);
        break;
      case 'C':
        Do_Check = true;
        break;
      case 'c':
        Mesh.second = (int)strtoul(optarg, NULL, 0);
        break;
      case 'D':
        if(*optarg == 'c' || *optarg == 'C') {
          CyclicDistribution = true;
          break;
        }
        if(*optarg == 'b' || *optarg == 'B') {
          CyclicDistribution = false;
          break;
        }
        printf("Unknown -D argument %c", *optarg);
        exit( EXIT_FAILURE );
      case 'i':
        SubProblems = (int)strtoul(optarg, NULL, 0);
        break;
      case 'N':
        NUM_ITERS = (int)strtoul(optarg, NULL, 0);
        break;
      case 'n':
        NReps = (int)strtoul(optarg, NULL, 0);
        break;
      case 'Q':
        MinTimeFactor = strtod(optarg, NULL);
        break;
      case 'r':
        Mesh.first = (int)strtoul(optarg, NULL, 0);
        break;
      case 'T':
        TuneMode = true;
        break;
      default: // unknown or missing argument
        printf("Unknown flag %c\n", ch);
        exit( EXIT_FAILURE );
    }
  }
  
  if (Mesh.first * Mesh.second != upcxx::rank_n()) {
    if(!imyrank) {
      std::cerr << "Mesh -r x -c != number of processes available\n";
    }
    exit(EXIT_FAILURE);
  }
  
  if (argc > optind && 0 != atoi(argv[optind]))  {
    // std::cout << "Generating matrix of size " << argv[optind] << std::endl;
  } else {
    if(!imyrank) {
      printf("Usage: gauss-seidel [-B base] [-C] [-D dist] [-i intl] [-m] [-N iters] [-r row_mesh] [-c col_mesh] [-T] [ -n reps] dim\n"
             "   This version divides the matrix in tiles of %d x %d that are evenly\n"
             "  distributed among the ranks in contiguous square or rectangular blocks of tiles.\n"
             //"-a           Use advance helper for non-DepSpawn version. Default is false\n"
             //"             WARNING: Almost always hangs. Do not use.\n"
             "-B base      Baseline version to use (can be used several times) :\n"
             "             0: Basic DepSpawn version\n"
             "             1: Filtered DepSpawn version\n"
             "             2: Manual SPDM version based on barriers\n"
             "             3: Manual SPDM version based on async messages\n"
             "             4: Manual SPDM version based on futures\n"
             "-C           Check result\n"
             "-c col_mesh  Number of columns of the mesh of processors\n"
             "-D dist      dist=[C]yclic or dist=[B]lock distribution of tiles on processors mesh. s Cyclic.\n"
             "-i intl      Internal subtasks per dimension\n"
             "-N iters     Number iterations of the stencil\n"
             "-n reps      Number of times to repeat each experiment\n"
             "-Q factor    Quit testing combinations with runtimes factor times slower than minimum in tuning\n"
             "-r row_mesh  Number of rows of the mesh of processors\n"
             "-T           Tune mode tests all the versions for the possible combinations of -c -r -D and -i.\n"
             "             If -B is used, only those versions are tried.\n"
             "             DepSpawn version is tried with and without EXACT_MATCH mode.\n"
             "             -i is tried up to 1 subtask/dimension/core or tiles of 25x25 per task\n",
             TILE, TILE);
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  const int dimension = atoi(argv[optind]);
  if (dimension % (Mesh.first * TILE)) {
    if(!imyrank) {
      std::cout << dimension << " rows % (" << Mesh.first << " * " << TILE << ") != 0\n";
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  if (dimension % (Mesh.second * TILE)) {
    if(!imyrank) {
      std::cout << dimension << " cols % (" << Mesh.second << " * " << TILE << ") != 0\n";
    }
    upcxx::finalize();
    exit(EXIT_FAILURE);
  }
  
  if (Do_Check && (NReps > 1)) {
    if(!imyrank) {
      std::cout << "Check can only be performed for an individual iteration. Setting -n 1\n";
    }
    NReps = 1;
  }

  if (Versions.empty()) {
    for(int i = 0; i <= MaxVersion; i++) {
      Versions.push_back(i);
    }
  }
  Baseline_Version = Versions[0];

  return dimension; //the dimension
}

void run(const int nthreads, const int M)
{ double time;
  int i;
  bool do_iters = true;
  
  if (!imyrank) {
    auto tiles = (*probl)->get_tiles();
    auto tiles_per_rank = (*probl)->get_tiles_per_rank();
    const char * subversion_description = "";
    if (Baseline_Version < 2) {
      subversion_description = depspawn::get_UPCXX_DEPSPAWN_EXACT_MATCH() ? "EM" : "NO_EM";
    }
    printf("nthreads=%d TILE=%d Niters=%d NReps=%d n=%dx%d Version=%d%s NRanks=(%d,%d) Distr=%s Subtasks_dim=%d\n", nthreads, TILE, NUM_ITERS, NReps, M, M, Baseline_Version, subversion_description, Mesh.first, Mesh.second,   CyclicDistribution ? "Cyclic" : "Block",  SubProblems);
    if(!TuneMode) {
      printf("Matrix tiles: %dx%d Tiles per rank: %dx%d\n", tiles.first, tiles.second, tiles_per_rank.first, tiles_per_rank.second);
    }
  }

  (*probl)->fill_in();

  for (i = 0; (i < NReps) && do_iters; i++) {
    // if (!imyrank) { puts("Starting"); }
    
    switch (Baseline_Version) {
      case 0:
        if (!imyrank && !i && !TuneMode) { puts("Running UPC++ DepSpawn version"); }
        time = (*probl)->dsp_block_gauss();
        break;
      case 1:
        if (!imyrank && !i && !TuneMode) { puts("Running UPC++ DepSpawn filtered version"); }
        time = (*probl)->dsp_filtered_block_gauss();
        break;
      case 2:
        if (!imyrank && !i && !TuneMode) { puts("Running Manual UPC++ version with barriers"); }
        time = (*probl)->barrier_block_gauss();
        break;
      case 3:
        if (!imyrank && !i && !TuneMode) { puts("Running Manual UPC++ version with async messages"); }
        time = (*probl)->async_block_gauss();
        break;
      case 4:
        if (!imyrank && !i && !TuneMode) { puts("Running Manual UPC++ version with futures"); }
        time = (*probl)->future_block_gauss();
        break;
      default:
        if (!imyrank) {
          std::cerr << "Unsupported Baseline version " << Baseline_Version << std::endl;
        }
    }

    time = upcxx::broadcast(time, 0).wait();
    if(time < MinTime) {
      MinTime = time;
    }

    do_iters = !TuneMode || (time <= (MinTime * MinTimeFactor));

// DEBUGGING
//std::cout << "~=" << 499 * 4000 + 499 << std::endl;
//std::cout << "Golden: " << (*(test_probl.get_tile(0,0).local()))[499][499] << std::endl;
//std::cout << "Result: " << (*(probl.get_tile(0,0).local()))[499][499] << std::endl;
//const bool are_equal = test_probl.equal_arr(probl);
//if (!imyrank) {
//  printf("Checking result: %s\n", are_equal ? "SUCCESS" : "FAIL" );
//}
// END DEBUGGING
    
    //std::cout << C(0,0).get().get(0,1) << std::endl;
    
  }

  if (!imyrank) {
    while (i++ < NReps) {
      std::cout << " Total Time: " << time << " sec" << std::endl;
    }
  }

}
    
void tune_version(const int nthreads, const int M)
{ std::pair<int, int> tiles;
  //std::vector<int> subtasks(1, 1);

  tiles.first = M / TILE;
  tiles.second = M / TILE;
  
  const int lim_distr_type = (NRanks > 1) ? 2 : 1;  // Block == Cyclic distribution for a single rank
  const int max_subproblems = std::min(TILE/25, static_cast<int>(std::thread::hardware_concurrency()));// Do not test subproblems smaller than 25x25

//  for (int test_subtasks = 2; test_subtasks <= max_subproblems; test_subtasks *= 2) {
//    subtasks.push_back(test_subtasks);
//  }

  MinTime = 1e20;
  for (int proc_rows = 1; proc_rows <= NRanks; proc_rows = 2 * proc_rows) {

    const int proc_cols = NRanks / proc_rows;
    if ((tiles.first < proc_rows) || (tiles.second < proc_cols)) {
      if(!imyrank) fprintf(stderr, "Skipping %dx%d tiles on %dx%d ranks because there are not enough tiles\n", tiles.first, tiles.second, proc_rows, proc_cols);
      continue;
    }

    if(tiles.first % proc_rows) {
      if(!imyrank) fprintf(stderr, "Skipping %dx%d tiles on %dx%d ranks because (tiles.first %% proc_rows)\n", tiles.first, tiles.second, proc_rows, proc_cols);
      continue;
    }

    if(tiles.second % proc_cols) {
      if(!imyrank) fprintf(stderr, "Skipping %dx%d tiles on %dx%d ranks because (tiles.second %% proc_cols)\n", tiles.first, tiles.second, proc_rows, proc_cols);
      continue;
    }

    Mesh = {proc_rows, proc_cols};

    for (int distr = 0; distr < lim_distr_type; distr++) {
      CyclicDistribution = (distr == 1);
      (*probl)->deallocate();
      (*probl)->init(M, M, Mesh, CyclicDistribution, probl);
      for (SubProblems = 1; SubProblems <= max_subproblems; SubProblems = 2 * SubProblems) {
        run(nthreads, M);
        if ((SubProblems < max_subproblems) && ((2 * SubProblems) > max_subproblems)) {
          SubProblems = max_subproblems / 2;
        }
      }
    }
  }
}
  
int main(int argc, char** argv)
{
  upcxx::init();

  imyrank = upcxx::rank_me();
  NRanks =  upcxx::rank_n();
  NReps = 1;
  
#ifndef UPCXX_BACKEND_GASNET_PAR
  if (!imyrank) {
    fprintf(stderr, "*** WARNING: UPCXX_BACKEND_GASNET_PAR not defined  ***\n");
  }
#endif
  
  if(NRanks & (NRanks-1)) {
    fprintf(stderr, "Number of ranks must be a power of 2\n");
    exit(EXIT_FAILURE);
  }
  
  int tmp = static_cast<int>(sqrtf(static_cast<float>(NRanks)));
  if ((tmp * tmp) == NRanks) {
    Mesh = {tmp, tmp};
  } else {
    tmp = static_cast<int>(sqrtf(static_cast<float>(NRanks / 2)));
    Mesh = {tmp, tmp * 2};
  }

  int M = arg_parse(argc, argv);
  
  probl = new upcxx::dist_object<WaveFrontProbl>({});
  (*probl)->init(M, M, Mesh, CyclicDistribution, probl);
  
// DEBUGGING
//test_probl.init(M, M, Mesh);
//test_probl.barrier_block_jacobi();
// END DEBUGGING

  const char * const nthreads_env = getenv("UD_NUM_THREADS");
  nthreads = (nthreads_env == NULL) ? -1 : static_cast<int>(strtol(nthreads_env, (char **)NULL, 10));

  depspawn::set_threads(nthreads);
  enable_ps_master();

//  if (!imyrank) {
//    printf("UPCXX_BACKEND_GASNET_SEQ=%d UPCXX_BACKEND_GASNET_PAR=%d\n", UPCXX_BACKEND_GASNET_SEQ, UPCXX_BACKEND_GASNET_PAR);
//  }

  if (TuneMode) {
    for (int tmp  : Versions) {
      Baseline_Version = tmp;
      depspawn::set_UPCXX_DEPSPAWN_EXACT_MATCH(false);
      enable_ps_master();
      tune_version(nthreads, M);
      if (Baseline_Version < 2) { // another run with EXACT_MATCH=ON
        depspawn::set_UPCXX_DEPSPAWN_EXACT_MATCH(true);
        enable_ps_master();
        tune_version(nthreads, M);
      }
    }
  } else {
    run(nthreads, M);
  }

  if (Do_Check) {
    if (!imyrank) { puts("Testing"); }
    SubProblems = 1;
    test_probl.init(M, M, Mesh, CyclicDistribution, nullptr);
    test_probl.barrier_block_gauss();
    //std::cout << "C2:" << C2(0,0).get().get(0,1) << std::endl;
    const bool are_equal = test_probl.equal_arr(*(*probl));
    if (!imyrank) {
      printf("Checking result: %s\n", are_equal ? "SUCCESS" : "FAIL" );
    }
    upcxx::barrier();
  }

  upcxx::finalize();

  return 0;
}
