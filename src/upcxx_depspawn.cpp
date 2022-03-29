/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
 Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     upcxx_depspawn.cpp
/// \brief    Main implementation file for UPCxx_DepSpawn
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>
///

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <csignal> // for hangs
#include <fcntl.h> // for hangs
#include <unistd.h> // for hangs
#include <algorithm>
#include <vector>
#include "depspawn/depspawn_utils.h"
#ifdef UPCXX_DEPSPAWN_NO_CACHE
// We will compile most cache-related stuff anyway
#undef UPCXX_DEPSPAWN_NO_CACHE
// We will skip a few things
#define UPCXX_DEPSPAWN_NO_CACHE2
#endif
#include "upcxx_depspawn/upcxx_depspawn.h"

#ifdef ATOMIC_EXACT_MATCH
#include "upcxx_depspawn/safe_queue_mpsc.h"
#endif


 /* Replicated utils from depspawn.cpp */
namespace {
  
  using namespace depspawn::internal;
  
  DEPSPAWN_PROFILEDEFINITION(size_t
                             upcxx_profile_jobs = 0,
                             upcxx_profile_local_jobs = 0,
                             upcxx_profile_steals = 0,
                             upcxx_profile_steal_attempts = 0,
                             upcxx_profile_stop_steals = 0,
                             upcxx_profile_stop_steal_attempts = 0,
                             upcxx_profile_predead_tasks = 0,
                             upcxx_profile_fast_extl_insert = 0,
                             upcxx_profile_fast_insert = 0);

  DEPSPAWN_PROFILEDEFINITION(float
                             upcxx_profile_avg_workitems = 0.f,
                             upcxx_profile_avg_ready_workitems = 0.f);

  DEPSPAWN_PROFILEDEFINITION(size_t
                             upcxx_profile_stops = 0,
                             upcxx_profile_erases = 0,
                             upcxx_profile_advance_erases = 0,
                             upcxx_profile_stop_advance_erases = 0,
                             upcxx_profile_too_early_fixed = 0,
                             upcxx_profile_too_early_fixedo = 0,
                             upcxx_profile_too_early_checks = 0,
                             upcxx_profile_too_early_cleans = 0);
  
  DEPSPAWN_PROFILEDEFINITION(std::atomic<size_t>
                             upcxx_profile_msgs_sent_ftask(0),
                             upcxx_profile_msgs_sent_otask(0),
                             upcxx_profile_msgs_recv_ftask(0),
                             upcxx_profile_msgs_recv_otask(0),
                             upcxx_profile_too_early_ftask(0));


  /// Whether the runtime variables will get values from the environment or not
  bool DISABLE_RUNTIME_SETUP = false;

  /// Each how many spawns does the main thread invoke upcxx::progress().
  /// Controlled by env var UPCXX_DEPSPAWN_ADV_RATE
  int SPAWN_ADVANCE_RATE;
  
  /// Each how many spawns does the main thread try to clear the worklist.
  /// Controlled by env var UPCXX_DEPSPAWN_CLR_RATE
  int SPAWN_CLEAR_RATE;
  
  /// Maximum number of Ready tasks to store in main_wait()
  constexpr int READY_TASKS_ARR_SZ = 64;
  
  /// Whether complex or simple clean will be performed
  /// Controlled by env var UPCXX_DEPSPAWN_CLEAN_TYPE
  bool COMPLEX_CLEAN;

  ///Minimum progress in Oldest_task that is worth reporting to other processes
  int SPAWN_MIN_TASK_REPORT;

  /// Each how many spawns does the main thread stop new spawnings
  /// Controlled by env var UPCXX_DEPSPAWN_STOP_RATE
  int SPAWN_STOP_RATE;
  
  /// minimum number of available tasks to stop new spawning
  ///Controlled by env var UPCXX_DEPSPAWN_MIN_LOCAL
  int MIN_LOCAL_TASKS;

  /// Controls ExactMatch mode
  bool EXACT_MATCH_MODE;

  /// control whether forward notices of finishsing tasks are sent to other processes
  bool UPCXX_DEPSPAWN_FWD_NOTICE;

  /// Whether idle threads do active wait
  bool UPCXX_DEPSPAWN_ACTIVE_WAIT;

  /// Notify whether someone is running a upcxx::progress
  std::atomic<bool> ADVANCE_IN_PROGRESS;

  constexpr task_ctr_t First_Task_Id = 1;

  DEPSPAWN_DEBUGDEFINITION(task_ctr_t Oldest_local_task_found = First_Task_Id); ///< Last locally found oldest task, may be not yet broadcasted
  upcxx::global_ptr<upcxx::global_ptr<task_ctr_t>> Shared_Oldest_task; ///< Last found oldest task. NRanks elems per rank
  volatile task_ctr_t *Oldest_task = nullptr;         ///< Local portion of Shared_Oldest_task
  UPCxx_Workitem* Bottom_task = nullptr;              ///< Last task in TDG. Only used in token sync mode

  struct ExactMatchNode_t {

#ifdef ATOMIC_EXACT_MATCH
    std::atomic<UPCxx_Workitem *> last_write;
    UPCxx_Workitem *last_read;
    std::atomic<unsigned int> num_uses;
#else
    UPCxx_Workitem *last_write, *last_read;
    unsigned int num_uses;
#endif

    ExactMatchNode_t() noexcept :
    last_write{nullptr}, last_read{nullptr}, num_uses{0}
    { }

  };

  
  using ExactMatchKey_t = upcxx::global_ptr<char>;

  // Only one thread makes the allocations, but all the threads deallocate
  using ExactMatchKeyMap_t = std::unordered_map<ExactMatchKey_t, ExactMatchNode_t>;

  ExactMatchKeyMap_t ExactMatchKeyMap;

#ifdef ATOMIC_EXACT_MATCH
  SafeQueue_MPSC<ExactMatchKey_t, 64> ExactMatchDeleteQueue; ///< Stores keys to delete from ExactMatchKeyMap
  
  // The invoker must have acquired ExactMatchLock
  void dequeue_ExactMatchDeleteQueue()
  { ExactMatchKey_t key;
    
    while (ExactMatchDeleteQueue.try_pop(key)) {
      const auto map_it = ExactMatchKeyMap.find(key); // could have been re-used and erased
      if( (map_it != ExactMatchKeyMap.end()) && !map_it->second.num_uses.load(std::memory_order_relaxed)) {
        ExactMatchKeyMap.erase(map_it);
      }
    }
  }
#endif

  /// Returns true if the intervals [s1, e1] and [s2, e2] overlap
  template<typename T>
  constexpr bool overlaps_intervals(const T s1, const T e1, const T s2, const T e2) noexcept
  {
    return (s1 <= s2) ? (s2 <= e1) : (s1 <= e2);
  }
  
  constexpr bool overlaps(const arg_info* const a, const arg_info* const b) noexcept
  {
    return overlaps_intervals(a->addr, a->addr + a->size - 1,
                              b->addr, b->addr + b->size - 1);
  }
  
  /// Returns true if the interval [s1, e1] contains the interval [s2, e2]
  template<typename T>
  constexpr bool contains_intervals(const T s1, const T e1, const T s2, const T e2) noexcept
  {
    return (s1 <= s2) && (e2 <= e1);
  }
  
  /// Means a contains b
  constexpr bool contains(const arg_info* const a, const arg_info* const b) noexcept
  { //the -1 for the end would be correct, but unnecessary in this test
    return contains_intervals(a->addr, a->addr + a->size, b->addr, b->addr + b->size);
  }

  constexpr bool isPowerOf2(int i) noexcept {
    return !(i & (i-1));
  }
  
  void setPowerOf2Var(int& var, int val, const char *name)
  {
    if(!isPowerOf2(val)) {
      fprintf(stderr, "%s must be a power of 2\n", name ? name : "Something");
      exit(EXIT_FAILURE);
    }
    var = val;
  }

} // anonymous namespace

namespace upcxx  {
  intrank_t PersistentCache::Node_t::here_ = -1;
  PersistentCache GlobalRefCache;
} // namespace upcxx

namespace depspawn {
  
  namespace internal {

    enum class Upcxx_Waiting_t : int { No, PeriodicStop, Final };
    enum class ThreadState_t : int { Idle, Service, TaskRun, Progress, CleanWorklist };

    upcxx::intrank_t MyRank = 0xFFFFFFFF;
    upcxx::intrank_t NRanks = 0;
    UPCxx_Workitem* volatile upcxx_worklist = nullptr;
    task_ctr_t UPCxx_Workitem::Ctr = First_Task_Id;
    upcxx_args_support Static_args_support;
    volatile Upcxx_Waiting_t Upcxx_Waiting = Upcxx_Waiting_t::No; ///< Main thread state w.r.t main_wait

    AtomicFlagAccessControl_t EraserLock, ExactMatchLock;
    LockFlagAccessControl_t DeallocLock;
    std::atomic<int> Upcxx_Done_workitems {First_Task_Id};
    std::atomic<int> Upcxx_local_live_tasks {0};
    
    LockFlagAccessControl_t Early_messages_mutex;
    std::set<task_ctr_t, std::greater<task_ctr_t>> Early_finish_notifications;
    
    std::mutex master_persona_mutex;
    std::mutex prefetch_persona_mutex;
    upcxx::persona prefetch_persona;
    thread_local ThreadState_t ThreadState {ThreadState_t::Service};
    std::vector<UPCxx_Workitem *> ProgressPendingPosts, CleanWorklistPendingPosts;
    thread_local std::vector<UPCxx_Workitem *> LocalPendingPosts;

    /// Controls wheter threads yield or not during waits
    /** It currently only applies to waits in the caches in which several threads wait for the same data item to arrive */
    bool UPCXX_DEPSPAWN_YIELD;
  
  /// Whether prefetchs must be performed
    bool UPCXX_DEPSPAWN_PREFETCH;
    
    static void main_wait(const bool is_final);

    // std::map<int, UPCxx_Workitem *> Ctr_to_UPCxx_Workitem_map;
    
#ifndef DEPSPAWN_POOL_CHUNK_SZ
    ///Number of elements to allocate at once in each new allocation requested by the pools
#define DEPSPAWN_POOL_CHUNK_SZ 32
#endif
    
#ifndef DEPSPAWN_MIN_POOL_ITEM_SZ
    ///Minimum space to allocate for each item in the pool
#define DEPSPAWN_MIN_POOL_ITEM_SZ 0
#endif
    
    Pool_t<UPCxx_Workitem, false> UPCxx_Workitem::Pool(DEPSPAWN_POOL_CHUNK_SZ, DEPSPAWN_MIN_POOL_ITEM_SZ);

    // Mostly used for initialization and FAST_START blocks
    extern int Nthreads;
#ifdef DEPSPAWN_FAST_START
    static const int FAST_ARR_SZ  = 16;
    extern int FAST_THRESHOLD;
#endif

    static void run_pending_posts()
    { static const auto comp_func = [] (UPCxx_Workitem *a, UPCxx_Workitem *b) { return a->ctr_ < b->ctr_; };

      const size_t nelems = LocalPendingPosts.size();
      if (nelems) {
        std::vector<UPCxx_Workitem *> pending_posts;
        std::swap(pending_posts, LocalPendingPosts);
        if (nelems > 1) {
          std::sort(pending_posts.begin(), pending_posts.end(), comp_func);
        }
        for(UPCxx_Workitem * w : pending_posts) {
          w->post();
        }
      }
    }

    static void try_upcxx_advance()
    {
      const ThreadState_t prev_thread_state = ThreadState;

      if (!ADVANCE_IN_PROGRESS.load(std::memory_order_relaxed) && !ADVANCE_IN_PROGRESS.exchange(true)) {
        assert( prev_thread_state != ThreadState_t::Progress ); //Could be Idle, Service or TaskRun
        assert( prev_thread_state != ThreadState_t::CleanWorklist );
        // (prev_thread_state != ThreadState_t::TaskRun) only implies (enum_thr_spec_father == nullptr)
        //if users cannot try to run tasks from the pool from within tasks, i.e., if tasks behave as
        //atomic units where a task cannot begin in the middle of another task run in the same thread
        assert( ( (prev_thread_state == ThreadState_t::TaskRun) && (enum_thr_spec_father != nullptr) ) ||
                ( (prev_thread_state != ThreadState_t::TaskRun) /*&& (enum_thr_spec_father == nullptr)*/ ) );

        ThreadState = ThreadState_t::Progress; // Notice: never >1 threads can be simultaneously at Progress
        {
          upcxx::persona_scope master_scope(master_persona_mutex, upcxx::master_persona());
          if (UPCXX_DEPSPAWN_PREFETCH && prefetch_persona_mutex.try_lock()) {
            std::lock_guard<std::mutex> prefetch_lock(prefetch_persona_mutex, std::adopt_lock);
            {
              upcxx::persona_scope prefetch_scope(prefetch_persona);
              upcxx::progress();
            }
          } else {
            upcxx::progress();
          }
          
          if (!ProgressPendingPosts.empty() && (prev_thread_state == ThreadState_t::Service)) {
            assert(LocalPendingPosts.empty());
            std::swap(LocalPendingPosts, ProgressPendingPosts);
          }
        }
        
        ADVANCE_IN_PROGRESS.store(false);
        
        ThreadState = prev_thread_state;
        run_pending_posts();
      }
    }

    /** @internal this version is only used for main_wait final termination */
    task_ctr_t get_global_oldest_task() noexcept
    {
      //return *std::min_element(Oldest_task, Oldest_task + NRanks);
 
      task_ctr_t ret = *Oldest_task;
      for(upcxx::intrank_t i = 1; i < NRanks; i++) {
        const auto tmp = Oldest_task[i];
        if(tmp < ret) ret = tmp;
      }
      return ret;
    }

    /**** BEGIN debugging code ****/

    // Returns pair with oldest local live Workitem and set of workitems it depends on
    std::pair<UPCxx_Workitem *, std::vector<UPCxx_Workitem *>> oldestLocalLiveWorkitem()
    { UPCxx_Workitem *p, *l = nullptr;
      std::vector<UPCxx_Workitem *> precs;

      for(p = upcxx_worklist; p != nullptr; p = static_cast<UPCxx_Workitem*>(p->next)) {
        if (p->is_local() && (p->status < UPCxx_Workitem::Status_t::Running)) {
          l = p;
        }
      }

      if (l != nullptr) {
        for(p = static_cast<UPCxx_Workitem*>(l->next); p != nullptr; p = static_cast<UPCxx_Workitem*>(p->next)) {
          for (Workitem::_dep* dp = p->deps; dp != nullptr; dp = dp->next) {
            if (dp->w == l) {
              precs.push_back(p);
            }
          }
        }
      }

      return {l, precs};
    }
    
    void print_oldestLocalLiveWorkitem(const std::pair<UPCxx_Workitem *, std::vector<UPCxx_Workitem *>>& pair, char * buffer = nullptr, int fd = 2)
    { char *buff;
      
      const bool dynamic_memory = (buffer == nullptr);
      
      if (dynamic_memory) {
        buff = buffer = (char *)malloc(2048);
      } else {
        buff = buffer;
      }
      
      if (pair.first == nullptr) {
        buff += sprintf(buff, "no local live Workitems\n");
      } else {
        buff += sprintf(buff, "oldest live Workitem %d: %d of %d deps found\n", pair.first->ctr_, (int)(pair.second.size()), (int)(pair.first->ndependencies));
        for (const auto& p : pair.second) {
          buff += sprintf(buff, "   %d from P%d [%d] (S%d)\n", p->ctr_, (int)(p->where_), Oldest_task[p->where_], (int)(p->status));
        }
      }
      
      size_t result = write(fd, buffer, buff - buffer);

      if (dynamic_memory) {
        free(buffer);
      }
    }

//DEPSPAWN_PROFILEDEFINITION(
    void catch_function(int signal)
    { char file_name[32], buffer[2048];
      const char *cp;
    
      sprintf(file_name, "term_%d_%d.deb", NRanks, MyRank);

      const int outfile = open(file_name, O_CREAT|O_TRUNC|O_WRONLY);
      
      switch (Upcxx_Waiting) {
        case Upcxx_Waiting_t::No:
          cp = "No";
          break;
        case Upcxx_Waiting_t::PeriodicStop:
          cp = "PeriodicStop";
          break;
        case Upcxx_Waiting_t::Final:
          cp = "Final";
          break;
      }
      
      char * buff = static_cast<char *>(buffer);
      buff += sprintf(buff, "Upcxx_Waiting=%s Upcxx_local_live_tasks=%d", cp, static_cast<int>(Upcxx_local_live_tasks));
      DEPSPAWN_DEBUGACTION(buff += sprintf(buff, " Oldest_local_task_found=%d", Oldest_local_task_found));
      buff += sprintf(buff, "\n min=%d Ctr=%zu signal=%d in_advance=%c pid=%d\n[", get_global_oldest_task(), (size_t)(UPCxx_Workitem::Ctr - 1), signal, ADVANCE_IN_PROGRESS.load() ? 'Y' : 'N', (int)getpid());

      for(int j = 0; j < NRanks; j++) {
        buff += sprintf(buff, "%d ", Oldest_task[j]);
      }
      
      *(buff - 1) = ']';
      *buff++ = '\n';

      size_t result = write(outfile, buffer, buff - buffer);
      
      print_oldestLocalLiveWorkitem(oldestLocalLiveWorkitem(), buffer, outfile);
      
      close(outfile);

      if(signal == SIGTERM) {
        //abort(); //So we get core dump
        exit(EXIT_FAILURE);
      }
    }
//)

    /**** END debugging code ****/

    /** This function is called by idle threads with from_task = false and
        threads running a task with from_task = true.
     */
    void idle_progress(const bool from_task)
    { volatile static int WorkChooser;
      
      const ThreadState_t prev_thread_state = ThreadState;

      if (from_task) {
        assert(enum_thr_spec_father != nullptr);
        ThreadState = ThreadState_t::TaskRun;
      } else {
        assert(enum_thr_spec_father == nullptr);
        ThreadState = ThreadState_t::Idle;
      }

      const int mywork = WorkChooser++;
      if (mywork & 1) {
        try_upcxx_advance();
      } else {
        if ((Upcxx_Waiting == Upcxx_Waiting_t::No) && EraserLock.try_lock()) {
          UPCxx_Workitem::Clean_worklist();
        }
      }
      
      ThreadState = prev_thread_state;
    }
  
    void upcxx_depspawn_start_system()
    { static task_ctr_t Last_advance, Last_clear;

      //Only needed the very first time
      if(!NRanks) {
        upcxx_depspawn_runtime_setup();
      }
      
      //Needed each time there is a global wait and re-start
      if( (TP == nullptr) || !TP->is_running() ) {
        
        // This is done here instead of in upcxx_depspawn_runtime_setup() because if a single process invokes
        //print_upcxx_depspawn_runtime_setup(), it will in turn invoke upcxx_depspawn_runtime_setup() and hang
        //alone on the shared_array init
        if (Oldest_task == nullptr) {
          const upcxx::intrank_t ncols = NRanks;

          upcxx::persona_scope scope(master_persona_mutex, upcxx::master_persona());

          Shared_Oldest_task = upcxx::new_array<upcxx::global_ptr<task_ctr_t>>(NRanks);
          auto sot0_fut = upcxx::broadcast(Shared_Oldest_task, 0);
          upcxx::global_ptr<task_ctr_t> * const remote_ptrs = Shared_Oldest_task.local();
          remote_ptrs[MyRank] = upcxx::new_array<task_ctr_t>(ncols);
          Oldest_task = remote_ptrs[MyRank].local();

          std::fill(Oldest_task, Oldest_task + ncols, First_Task_Id);
          upcxx::rput<upcxx::global_ptr<task_ctr_t>>(remote_ptrs[MyRank], sot0_fut.wait() + MyRank).wait();
          upcxx::barrier(); // Ensure all rputs finished before the broadcast begins
          upcxx::broadcast(remote_ptrs, NRanks, 0).wait(); // get all the pointers from 0

          // upcxx::barrier(); // needed ?
        }

        start_master();

        if (UPCXX_DEPSPAWN_ACTIVE_WAIT) {
          TP->wait(false);
          TP->set_idle_function( []{ idle_progress(false); } );
          TP->launch_threads();
        }

        //should be ready after initial startup and previous wait
        //upcxx_worklist = nullptr;
        //UPCxx_Workitem::Ctr = First_Task_Id;
        
        Last_advance = 0;
        Last_clear = 0;

        //Avoids potential problem if UPCxx_Workitem::Clean_worklist() is called below, being upcxx_worklist == nullptr (breaks)
        return;
      }

      if(((UPCxx_Workitem::Ctr - Upcxx_Done_workitems) > SPAWN_STOP_RATE) && (Upcxx_local_live_tasks > MIN_LOCAL_TASKS)) {
        DEPSPAWN_PROFILEACTION(upcxx_profile_stops++);
        internal::main_wait(false);
      } else {
        if ( (UPCxx_Workitem::Ctr - Last_advance) >= SPAWN_ADVANCE_RATE ) {
          Last_advance = UPCxx_Workitem::Ctr;
          try_upcxx_advance(); // Help progress just in case?
        }

        if(((UPCxx_Workitem::Ctr - Last_clear) >= SPAWN_CLEAR_RATE) && EraserLock.try_lock() ) {
          Last_clear = UPCxx_Workitem::Ctr;
          DEPSPAWN_PROFILEACTION(upcxx_profile_advance_erases++);
          UPCxx_Workitem::Clean_worklist();
        }
      }
  
    }

    // Orders by rank, then by addr within rank
    static constexpr bool upcxx_arglist_precedes(arg_info* n, arg_info* p) noexcept {
      return (n->rank < p->rank) || ( (n->rank == p->rank) && (n->addr < p->addr) );
    }
    
    static void upcxx_insert_in_arglist(arg_info* n, arg_info*& args);

    static void upcxx_solve_overlap(arg_info* n, arg_info *other)
    { size_t tmp_size;
      
      const bool orwr = n->wr || other->wr;
      
      if(n->size == other->size) {
        other->wr = orwr;
        arg_info::Pool.free(n);
      } else if (n->size < other->size) {
        
        if(orwr == other->wr) {
          arg_info::Pool.free(n);
        } else {
          tmp_size = n->size;
          
          n->addr += tmp_size;
          n->size = other->size - tmp_size;
          n->wr = other->wr; //should be false
          
          other->size = tmp_size;
          other->wr = orwr; //should be true
          
          upcxx_insert_in_arglist(n, other);
        }
        
      } else { //size > other->size
        
        if(orwr == n->wr) {
          other->wr = orwr;
          other->size = n->size;
          arg_info::Pool.free(n);
        } else { //Should be other->wr = true, this->wr = false
          tmp_size = other->size;
          
          n->addr += tmp_size;
          n->size -= tmp_size;
          
          upcxx_insert_in_arglist(n, other);
        }
        
      }
      
    }
    
    static void upcxx_insert_in_arglist(arg_info* n, arg_info*& args)
    { arg_info *p, *prev;
      
      if(!args) {
        n->next = nullptr;
        args = n;
      } else if ( upcxx_arglist_precedes(n, args) ) {
        n->next = args;
        args = n;
      } else {
        prev = args;
        p = args->next;
        while(p) {
          if(upcxx_arglist_precedes(n, p)) {
            if( (n->rank == prev->rank) && (n->addr == prev->addr) ) {
              upcxx_solve_overlap(n, prev);
            } else {
              prev->next = n;
              n->next = p;
            }
            break;
          }
          prev = p;
          p = p->next;
        }
        
        if(!p) {
          if( (n->rank == prev->rank) && (n->addr == prev->addr) ) {
            upcxx_solve_overlap(n, prev);
          } else {
            prev->next = n;
            n->next = nullptr;
          }
        }
      }
      
    }
    
    upcxx_args_support::upcxx_args_support() noexcept :
    pargs_(nullptr), nargs_(0)
    {}

    void upcxx_args_support::insert_arg(arg_info *n) noexcept
    {
      nargs_++;
      add_rank_weight(n->rank, n->wr);
      upcxx_insert_in_arglist(n, pargs_);
    }

    upcxx::intrank_t upcxx_args_support::get_rank() const noexcept
    {
      //assert(!prio_.empty()); Can be empty for f() without arguments.

      RankToWeight_t::key_type rank = 0; // defaults to P0
      RankToWeight_t::mapped_type weight = 0;
      
      const auto itend = prio_.cend();
      for (auto it = prio_.cbegin(); it != itend; ++it) {
        if (it->second > weight) {
          rank   = it->first;
          weight = it->second;
        }
      }

      return static_cast<upcxx::intrank_t>(rank);
    }

    void upcxx_args_support::clear() noexcept
    {
      prio_.clear();
      pargs_ = nullptr;
      nargs_ = 0;
    }

    /** \internal
     These notifications are always useful because they are only sent
     when there is a dependent task detected in the local plan of the sender.
     But they can arrive
     a) before the system is started in the receiver and/or
     b) before that task exists in the receiver
     If they arrive too early this function does not find the task, and thus the
     notification is ignored. The problem is ameliorated because the notify_new_oldest_task
     mechanism should sooner or later inform again that this task finished.
     */
    void upcxx_depspawn_notify_finish_task(task_ctr_t task_ctr)
    { UPCxx_Workitem* p;

      DEPSPAWN_PROFILEACTION(upcxx_profile_msgs_recv_ftask++);
      
      //TODO: Use a map(ctr_ -> UPCxx_Workitem*) to find faster the task?
      if ( (upcxx_worklist == nullptr) || (upcxx_worklist->ctr_ < task_ctr) ) { // Cannot be in the list; but may be in the future
        Early_messages_mutex.lock();
        const bool too_early = (upcxx_worklist == nullptr) || (upcxx_worklist->ctr_ < task_ctr);
        if(too_early) {
          DEPSPAWN_PROFILEACTION(upcxx_profile_too_early_ftask++);
          Early_finish_notifications.insert(task_ctr);
        }
        Early_messages_mutex.unlock();
        if(too_early) {
          return;
        }
      }
      
      //Required to not collide with an eraser while traversing the list
      DeallocLock.lock();
      for (p = upcxx_worklist; (p != nullptr) && (p->ctr_ > task_ctr); p = static_cast<UPCxx_Workitem*>(p->next));
      
      if ( (p != nullptr) && (p->ctr_ == task_ctr) && (p->status < UPCxx_Workitem::Status_t::Done) ) {

        // if this happens to be the upcxx_worklist element and it is still being inserted,
        //its arg list is still in use, and finish_execution() does not wait for the fill
        //because it first sets its status to Done!
        while(p->status == UPCxx_Workitem::Status_t::Filling) { };

        //notice that in principle this could trigger a clean_worklist unless EraserLock is locked
        //which would then self-deadlock on DeallocLock.
        //to avoid it, now finish_execution() can only trigger clean_worklist when finishing local tasks
        p->finish_execution();
      }
      DeallocLock.unlock();
      
    }

  void upcxx_depspawn_send_notify_new_oldest_task(const task_ctr_t new_oldest_task)
    { upcxx::promise<> prom;

      DEPSPAWN_PROFILEACTION(upcxx_profile_msgs_sent_otask += (NRanks - 1));

      upcxx::global_ptr<task_ctr_t> * const remote_ptrs = Shared_Oldest_task.local();
      task_ctr_t * const src_ptr = const_cast<task_ctr_t *>(&Oldest_task[MyRank]);

      for (upcxx::intrank_t i = 0; i < NRanks; i++) {
        if (i != MyRank) {
          upcxx::rput(src_ptr, remote_ptrs[i] + MyRank, 1, upcxx::operation_cx::as_promise(prom));
        }
      }

    }

    UPCxx_Workitem::UPCxx_Workitem() :
    Workitem()
    { }

    UPCxx_Workitem::UPCxx_Workitem(upcxx_args_support& args_support) :
    Workitem(args_support.pargs_, args_support.nargs_),
    where_(args_support.get_rank()), ctr_(Ctr++)
    {
      args_support.clear();
      assert(father == nullptr);
    }

    void UPCxx_Workitem::post()
    {
#ifndef UPCXX_DEPSPAWN_NO_CACHE2
      if (UPCXX_DEPSPAWN_PREFETCH && prefetch_persona_mutex.try_lock()) {
        upcxx::future<> f;
        upcxx::intrank_t rank;
        upcxx::PersistentCache::underlying_global_ptr_t *ptr;
  
        std::lock_guard<std::mutex> prefetch_lock(prefetch_persona_mutex, std::adopt_lock);
        {
          upcxx::persona_scope prefetch_scope(prefetch_persona);
          //TODO: define std::unordered_map (ptr) -> future , link futures and when_all -> post task
          for (arg_info *arg_p = args; arg_p != nullptr; arg_p = arg_p->next) {
            if (!arg_p->wr && (( rank = static_cast<upcxx::intrank_t>(arg_p->rank)) != MyRank)) { //Prefetch read-only non-local arguments
              ptr = (upcxx::PersistentCache::underlying_global_ptr_t *)arg_p->addr;
              upcxx::GlobalRefCache.get(rank, ptr, arg_p->size, &f);
            }
          }
          
          upcxx::progress();
        }
      }
#endif
      Workitem::post();
    }

    void UPCxx_Workitem::insert_depencency_on(UPCxx_Workitem *workitem)
    {
      if(!is_local()) { //only with !this->is_local() && workitem->is_local()
        assert(workitem->is_local());
        if (UPCXX_DEPSPAWN_FWD_NOTICE) {
          workitem->dependent_ranks_.insert(where_);
        }
      } else { //only with this->is_local(). workitem may be local or not
        ndependencies++;
        Workitem::_dep* const newdep = Workitem::_dep::Pool.malloc();
        newdep->next = nullptr;
        newdep->w = this;
        
        if(workitem->deps == nullptr) {
          workitem->deps = workitem->lastdep = newdep;
        } else {
          workitem->lastdep->next = newdep;
          workitem->lastdep = newdep;
        }
      }
    }

    void UPCxx_Workitem::insert_in_worklist(AbstractRunner* itask)
    { UPCxx_Workitem* p;
      arg_info *arg_p, *arg_w;
      bool predead, fast_extl_insert;
  
      task = itask;
      next = upcxx_worklist;
      
      const bool is_local_task = is_local();

      DEPSPAWN_PROFILEDEFINITION(unsigned int workitems_in_list = 0);
      DEPSPAWN_PROFILEACTION(upcxx_profile_jobs++);
      DEPSPAWN_PROFILEACTION(upcxx_profile_local_jobs += (unsigned)is_local_task);
      
      //DEPSPAWN_PROFILEACTION(if (ctr_ < Oldest_task[where_]) { upcxx_profile_predead_tasks++;});

      if (is_local_task) {
        predead = fast_extl_insert = false;
        Upcxx_local_live_tasks++;
      } else {
        predead = (ctr_ < Oldest_task[where_]);
        if (!predead && !Early_finish_notifications.empty()) {
          DEPSPAWN_PROFILEACTION(upcxx_profile_too_early_checks++);
          bool needs_unlock = true;
          Early_messages_mutex.lock();
          auto iter = Early_finish_notifications.lower_bound(ctr_);
          if (iter != Early_finish_notifications.end()) {
            DEPSPAWN_PROFILEACTION(upcxx_profile_too_early_cleans++);
            if ((*iter) == ctr_) {
              DEPSPAWN_PROFILEACTION(upcxx_profile_too_early_fixed++);
              predead = true;
              const auto iter_orix = iter++;
              Early_finish_notifications.erase(iter_orix);
            }
            if(iter != Early_finish_notifications.end()) {
              std::set<task_ctr_t, std::greater<task_ctr_t>> tmp_set(iter, Early_finish_notifications.end());
              Early_finish_notifications.erase(iter, Early_finish_notifications.end());
              Early_messages_mutex.unlock(); // avoids thread0 scenario 4 in TERM_debugging_Notes.txt
              needs_unlock = false;
              p = upcxx_worklist; // It also makes sense to do this latter part when inserting local tasks
              // Locking EraserLock allows to
              // (1) safely traverse the task list and
              // (2) avoid that finish_execution generates a clean_worklist that can generate a upcxx::rpc that may be
              //     also receives messages requiring Early_messages_mutex (or EraserLock), giving place to a deadlock.
              // Also notice that since we are finishing remote tasks, they cannot give place to remote_awake() calls
              //
              // Now that finish_execution() can only trigger clean_worklist when finishing local tasks, it should be
              // enough to lock on DeallocLock
              DeallocLock.lock();
              for(const task_ctr_t notified_ctr : tmp_set) {
                while( (p != nullptr) && (p->ctr_ > notified_ctr) ){
                  p = static_cast<UPCxx_Workitem*>(p->next);
                }
                if( (p != nullptr) && (p->ctr_ == notified_ctr) && (p->status < Status_t::Done)) {
                  DEPSPAWN_PROFILEACTION(upcxx_profile_too_early_fixedo++);
                  p->finish_execution();
                }
              }
              DeallocLock.unlock();
            }
          }
          if (needs_unlock) {
            Early_messages_mutex.unlock();
          }
        }
        fast_extl_insert = !Upcxx_local_live_tasks;
      }

      int nargs = 0;
      arg_info* argv[static_cast<int>(nargs_) + 1];
      arg_w = args;

      // Save original list of arguments
      if (EXACT_MATCH_MODE) {
        p = nullptr; // marks where to begin search
        upcxx_worklist = this; //link workitem
        ExactMatchLock.lock();
        
#ifdef ATOMIC_EXACT_MATCH
        dequeue_ExactMatchDeleteQueue();
#endif

        while(arg_w != nullptr) {
          const ExactMatchKey_t key(upcxx::detail::internal_only(), static_cast<upcxx::intrank_t>(arg_w->rank), (char *)arg_w->addr);
          auto map_it = ExactMatchKeyMap.find(key);
          if (map_it != ExactMatchKeyMap.end()) {
            if (!predead && !fast_extl_insert) {
              UPCxx_Workitem* const wr_p = map_it->second.last_write;
              UPCxx_Workitem* const rd_p = map_it->second.last_read;
              // the condition on older wr_p is not needed because a newer wr_p nulls rd_p
              if( arg_w->wr && (rd_p != nullptr) /*&& ((wr_p == nullptr) || (wr_p->ctr_ < rd_p->ctr_))*/ ) { // write depends on last and following reads
                if( (p == nullptr) || (p->ctr_ < rd_p->ctr_) ) {
                  // We cannot also include the condition on
                  // (rd_p->status < Status_t::Done) && (is_local_task || rd_p->is_local())
                  // because may be rd_p finished or is not local, but it hides older non-finished non-local reads
                  p = rd_p;
                }
                argv[nargs++] = arg_w;
              } else { //read or write only depends on last write
                if( (wr_p != nullptr) && (wr_p->status < Status_t::Done) && (is_local_task || wr_p->is_local()) ) {
                  insert_depencency_on(wr_p);
                }
              }
            }
          } else {
            const auto insert_pair = ExactMatchKeyMap.emplace(std::piecewise_construct, std::make_tuple(key), std::make_tuple());
            assert(insert_pair.second);
            map_it = insert_pair.first;
          }
          
          ExactMatchNode_t& node = map_it->second;
          arg_w->array_range = (array_range_t*)(&(*(map_it)));
          node.num_uses++;
          if(arg_w->wr) { // predead are fine to do this because this basically erases the dependencies on the position
            node.last_write = this;
            node.last_read = nullptr; // older reads do not matter
          } else {
            if(!predead) { // predead do not overwrite the subsequent most recent read
              node.last_read = this;
            }
          }
          arg_w = arg_w->next;
        }

        ExactMatchLock.unlock();
        assert(!nargs || (p != nullptr));
      } else {
        while(arg_w != nullptr) {
          argv[nargs++] = arg_w;
          arg_w = arg_w->next;
        }
        // This is not true because nargs_ does not consider repeated arguments, while arg_w does
        //assert(nargs == static_cast<int>(nargs_));

        p = static_cast<UPCxx_Workitem*>(next);
        upcxx_worklist = this; //link workitem
        // Without this fence the change of head can be unseen by current finish_execution's
        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      // End save original list of arguments

      if (predead) {
        DEPSPAWN_PROFILEACTION(upcxx_profile_predead_tasks++);
        finish_execution();
        return;
      }
      
      if(fast_extl_insert) {
        DEPSPAWN_PROFILEACTION(upcxx_profile_fast_extl_insert++);
        status = Status_t::Waiting;
        return;
      }

#ifdef DEPSPAWN_FAST_START
      bool steal_work = false;
      int nready = 0, nfillwait = 0;
#endif

      if (nargs) {
        argv[nargs] = nullptr;
        while(p != nullptr) {
          DEPSPAWN_PROFILEACTION(workitems_in_list++);

          if((p->status < Status_t::Done) && (is_local_task || p->is_local())) {
            int arg_w_i = 0;
            arg_p = p->args; // preexisting workitem
            arg_w = argv[0]; // New workitem
            while(arg_p && arg_w) {
              
              const bool conflict = (arg_p->wr || arg_w->wr)
              && (arg_p->rank == arg_w->rank)
              && overlaps(arg_p, arg_w);
              
              if(conflict) { // Found a dependency
                
                insert_depencency_on(p);
                
                if (arg_p->wr && contains(arg_p, arg_w)) {
                  nargs--;
                  
                  if (!nargs) {
                    DEPSPAWN_PROFILEACTION(upcxx_profile_fast_insert++);
                    // The optimal thing would be to just make this goto to leave the main loop and insert
                    // the Workitem waiting. But in tests with repeated spawns this leads to very fast insertion
                    // that slows down the performance. Other advantage of continuing down the list is that
                    // it is more likely DEPSPAWN_FAST_START is triggered and it uses oldest tasks
                    
                    goto OUT_MAIN_insert_in_worklist_LOOP;
                    
                  } else {
                    for (int i = arg_w_i; i <= nargs; i++) {
                      argv[i] = argv[i+1];
                    }
                  }
                }
                break;
              } else {
                if(upcxx_arglist_precedes(arg_p, arg_w)) {
                  arg_p = arg_p->next;
                } else {
                  arg_w = argv[++arg_w_i];
                }
              }
            }
            
#ifdef DEPSPAWN_FAST_START
            if (p->is_local()) {
              const auto tmp_status = p->status;
              nfillwait += ( tmp_status < Status_t::Ready );
              nready += ( tmp_status == Status_t::Ready );
            }
#endif
          }
          p = static_cast<UPCxx_Workitem*>(p->next);
        }
        
      OUT_MAIN_insert_in_worklist_LOOP:
        
        DEPSPAWN_PROFILEACTION(upcxx_profile_avg_ready_workitems = (upcxx_profile_avg_ready_workitems * (upcxx_profile_jobs - 1) + nready) / (float)upcxx_profile_jobs);
        
#ifdef DEPSPAWN_FAST_START
        if ((nready > FAST_THRESHOLD) || ((nready > Nthreads) && (nfillwait > FAST_THRESHOLD))) {
          DEPSPAWN_PROFILEACTION(upcxx_profile_steal_attempts++);
          steal_work = true;
        }
#endif
      }

      // This is set after stealing work because this way
      // the fast_arr workitems should not have been deallocated
      //status = (!ndependencies) ? Ready : Waiting;
      if (is_local_task && !ndependencies) {
        post();
      } else {
        status = Status_t::Waiting;
      }

#ifdef DEPSPAWN_FAST_START
      if(steal_work && TP->try_run()) {
        DEPSPAWN_PROFILEACTION(upcxx_profile_steals++);
      }
#endif

      DEPSPAWN_PROFILEACTION(upcxx_profile_avg_workitems = (upcxx_profile_avg_workitems * (upcxx_profile_jobs - 1) + workitems_in_list) / (float)upcxx_profile_jobs);
    }
    
    void UPCxx_Workitem::remote_awake()
    {
      const auto end_ptr = dependent_ranks_.cend();
      
      for(auto ptr = dependent_ranks_.cbegin(); ptr != end_ptr; ++ptr) {
        upcxx::rpc_ff(*ptr, upcxx_depspawn_notify_finish_task, ctr_);
      }
      DEPSPAWN_PROFILEACTION(upcxx_profile_msgs_sent_ftask += dependent_ranks_.size());

    }
    
    void UPCxx_Workitem::finish_execution()
    { bool erase;
      
      UPCxx_Workitem * const current = this;

      // Finish work
      const bool was_local = is_local();
      if (was_local) {
        Upcxx_local_live_tasks--;

        erase = (Upcxx_Waiting == Upcxx_Waiting_t::No)
                 && (this->ctr_ == Oldest_task[MyRank]) // (((((intptr_t)current)>>8)&0xfff) < 32)
                 && EraserLock.try_lock();
        assert(!erase || EraserLock.is_locked());
      } else {
        // One can simultaneously arrive here from clean_worklist, upcxx_depspawn_notify_finish_task and insert_in_worklist for pre_dead tasks
        // so we use the atomic children, which should be 1 on initial finish_execution to avoid conflicts
        if(!current->nchildren.exchange(0)) {
          return;
        }
        erase = false;
      }
  
      current->status = Status_t::Done;
      Upcxx_Done_workitems++;

      // Without this fence the change of status can be unseen by current fill ins
      // It it now commented out because the Upcxx_Done_workitems++ atomic operation should release the write to memory
      // std::atomic_thread_fence(std::memory_order_seq_cst);
    
      UPCxx_Workitem * const worklist_wait_hint = upcxx_worklist;
      
      if (!current->dependent_ranks_.empty()) {
        current->remote_awake();
      }
      
      while(worklist_wait_hint->status == Status_t::Filling) {}
      
      // Notice that ranks could be inserted in dependent_ranks_
      //while or after remote_awake() runs because we are agressively parallelizing
      //it with the wait for worklist_wait_hint. For this reason it must be cleaned later
      if( !current->dependent_ranks_.empty() ) {
        dependent_ranks_.clear();
      }

      arg_info *arg_p = current->args;
      if ((arg_p != nullptr) && !upcxx::GlobalRefCache.empty()) {
        do { // Cache invalidations must be performed before new tasks are spawned
          if (arg_p->wr && (static_cast<upcxx::intrank_t>(arg_p->rank) != MyRank)) {
            const upcxx::PersistentCache::Key_t tmp(upcxx::detail::internal_only(), static_cast<upcxx::intrank_t>(arg_p->rank), (upcxx::PersistentCache::underlying_global_ptr_t *)(arg_p->addr));
            upcxx::GlobalRefCache.invalidate(tmp);
          }
          arg_p = arg_p->next;
        } while (arg_p != nullptr);
      }

      Workitem::_dep *idep = current->deps;
      if (idep != nullptr) {
        do {
          if(idep->w->ndependencies.fetch_sub(1) == 1) {
            UPCxx_Workitem * const workitem = static_cast<UPCxx_Workitem *>(idep->w);
            //cannot run tasks while in ThreadState_t::Progress (inside upcxx::progress)
            //and it is unwise to run the risk to begin to execute a new task while in Clean_worklist
            switch (ThreadState) {
              case ThreadState_t::Service:
                workitem->post();
                break;
              case ThreadState_t::Progress:
                ProgressPendingPosts.emplace_back(workitem);
                break;
              case ThreadState_t::CleanWorklist:
                assert(EraserLock.is_locked()); // || (Upcxx_Waiting != Upcxx_Waiting_t::No));
                CleanWorklistPendingPosts.emplace_back(workitem);
                break;
              default:
                assert(false);
                break;
            }
          }
          idep = idep->next;
        } while (idep != nullptr);
        Workitem::_dep::Pool.freeLinkedList(current->deps, current->lastdep);
        current->deps = current->lastdep = nullptr;
      }

      if (current->args != nullptr) {
        if (EXACT_MATCH_MODE) {
#ifdef ATOMIC_EXACT_MATCH
          // Atomic implementation
          arg_info::Pool.freeLinkedList(current->args,
                                        [current](arg_info * const arg_w) {
                                          ExactMatchKeyMap_t::value_type& v = *((ExactMatchKeyMap_t::value_type*)(arg_w->array_range));
                                          ExactMatchNode_t& node = v.second;
                                          UPCxx_Workitem *tmp = current;
                                          if (/*(node.last_write.load(std::memory_order_relaxed) != nullptr) && */ node.last_write.compare_exchange_strong(tmp, nullptr)) {
                                            assert(arg_w->wr);
                                          }
                                          if (node.num_uses.fetch_sub(1) == 1) {
                                            // We cannot null the node pointers here because it would be a race with the inserter thread
                                            ExactMatchDeleteQueue.push(v.first);
                                            /*
                                            const ExactMatchKey_t key {v.first};
                                            ExactMatchLock.lock();
                                            const auto map_it = ExactMatchKeyMap.find(key); // could have been re-used and erased
                                            if( (map_it != ExactMatchKeyMap.end()) && !map_it->second.num_uses) {
                                              ExactMatchKeyMap.erase(map_it);
                                            }
                                            ExactMatchLock.unlock();
                                            */
                                          }
                                        });

          if ( (ExactMatchDeleteQueue.size() > (ExactMatchDeleteQueue.ReservedSize >> 1)) && ExactMatchLock.try_lock()) {
            dequeue_ExactMatchDeleteQueue();
            ExactMatchLock.unlock();
          }
#else
          ExactMatchLock.lock();
          arg_info::Pool.freeLinkedList(current->args,
                                        [current](arg_info * const arg_w) {
                                          ExactMatchKeyMap_t::value_type& v = *((ExactMatchKeyMap_t::value_type*)(arg_w->array_range));
                                          ExactMatchNode_t& node = v.second;
                                          if (node.num_uses == 1) {
                                            ExactMatchKeyMap.erase(v.first);
                                          } else {
                                            node.num_uses--;
                                            if (node.last_write == current) {
                                              assert(arg_w->wr);
                                              node.last_write = nullptr;
                                            }
                                          }
                                        });
          ExactMatchLock.unlock();
#endif
        } else {
          arg_info::Pool.freeLinkedList(current->args);
        }
        current->args = nullptr;
      }
      
      current->status = Status_t::Deallocatable;

      if (erase) {
        assert(EraserLock.is_locked());
        Clean_worklist();
      }

      // invoking advance() after Clean_worklist() instead of before ensures this thread releases the EraserLock if it previously got it,
      //which avoids a self-deadlock if advance() invokes upcxx_depspawn_notify_finish_task() and it tries to
      //get again EraserLock
      if(was_local) {
        try_upcxx_advance();
      }
  
    }
 
    volatile UPCxx_Workitem *common_clear(volatile UPCxx_Workitem * const debug_p, const task_ctr_t my_tentative_oldest_task, const bool my_tentative_oldest_task_local, UPCxx_Workitem * const lastkeep)
    {
      lastkeep->next = nullptr;

      std::atomic_thread_fence(std::memory_order_seq_cst);

      volatile UPCxx_Workitem * const current_upcxx_worklist = upcxx_worklist;

      assert( (debug_p->next == nullptr) || (static_cast<UPCxx_Workitem*>(debug_p->next)->where_ != MyRank) || (debug_p->next->status == UPCxx_Workitem::Status_t::Deallocatable) );
      assert( debug_p->ctr_ == my_tentative_oldest_task );
      assert(my_tentative_oldest_task >= Oldest_local_task_found);
      assert(Oldest_local_task_found >= Oldest_task[MyRank]);

      DEPSPAWN_DEBUGACTION(Oldest_local_task_found = my_tentative_oldest_task);

      static bool Static_my_tentative_oldest_task_local = false;

      const task_ctr_t oldest_task_delta = my_tentative_oldest_task - Oldest_task[MyRank];
 
      Static_my_tentative_oldest_task_local |= my_tentative_oldest_task_local;

//      if ((oldest_task_delta > SPAWN_MIN_TASK_REPORT) || ((Upcxx_Waiting != Upcxx_Waiting_t::No) && oldest_task_delta)) {
      if ( (Static_my_tentative_oldest_task_local && ((oldest_task_delta > SPAWN_MIN_TASK_REPORT) || ((Upcxx_Waiting != Upcxx_Waiting_t::No) && oldest_task_delta)))
           || ((Upcxx_Waiting == Upcxx_Waiting_t::Final) && oldest_task_delta) ) {
        // We only update locally when we also update remotely because
        //otherwise the local partial updates smaller than the threshold could accumulate,
        //resulting in fewer or even no remote updates
        Oldest_task[MyRank] = my_tentative_oldest_task;
        upcxx_depspawn_send_notify_new_oldest_task(my_tentative_oldest_task);
        Static_my_tentative_oldest_task_local = false;
      }

      return current_upcxx_worklist;
    }
 
    // Implementation that identifies chunks of deletable subitems and deallocates them as
    //the potentially depending Dones finish their execution
    void complex_clean_worklist(UPCxx_Workitem *lastkeep)
    { static std::vector< std::pair<volatile UPCxx_Workitem *, uint32_t> > Dones;
      static std::vector< std::pair<UPCxx_Workitem *, UPCxx_Workitem *> > Deletable_Sublists;
      
      using Status_t = UPCxx_Workitem::Status_t;

      UPCxx_Workitem *last_workitem;
      volatile UPCxx_Workitem *p, *debug_p = lastkeep;
      
      unsigned int deletable_workitems = 0;
      task_ctr_t my_tentative_oldest_task = lastkeep->ctr_;
      bool my_tentative_oldest_task_local = lastkeep->where_ == MyRank;

      for(p = static_cast<volatile UPCxx_Workitem*>(lastkeep->next); p != nullptr; p = static_cast<volatile UPCxx_Workitem*>(p->next)) {

        const upcxx::intrank_t workitem_place = p->where_;
        if ( (p->ctr_ < Oldest_task[workitem_place]) && (p->status < Status_t::Done) ) {
          assert(workitem_place != MyRank);
          const_cast<UPCxx_Workitem*>(p)->finish_execution();
        }

        if (p->status != Status_t::Deallocatable) {

          if (workitem_place == MyRank) {
            assert(my_tentative_oldest_task > p->ctr_);
            // assert(p->status != Status_t::Deallocatable); can become Deallocatable since the if...
            my_tentative_oldest_task = p->ctr_;
            my_tentative_oldest_task_local = true;
            debug_p = p;
          }

          if (deletable_workitems > 4) { //We ask for a minimum that justifies the cost
            // Deleted_workitems += deletable_workitems;
            Deletable_Sublists.emplace_back(static_cast<UPCxx_Workitem *>(lastkeep->next), last_workitem); // Deleted sublist
            lastkeep->next = const_cast<UPCxx_Workitem*>(p);
          }
          
          if(p->status == Status_t::Done) {
            Dones.emplace_back(p, Deletable_Sublists.size());
          }
          
          lastkeep = const_cast<UPCxx_Workitem*>(p);
          deletable_workitems = 0;
          
        } else {
          deletable_workitems++;
          assert(const_cast<UPCxx_Workitem*>(p)->dependent_ranks_.empty());
          assert(p->deps == nullptr);
          assert(p->args == nullptr);
        }
        
        last_workitem = const_cast<UPCxx_Workitem*>(p);
      }
      
      if (lastkeep->next != nullptr) {
        // Deleted_workitems += deletable_workitems;
        Deletable_Sublists.emplace_back(static_cast<UPCxx_Workitem *>(lastkeep->next), last_workitem);
      }
      
      p = common_clear(debug_p, my_tentative_oldest_task, my_tentative_oldest_task_local, lastkeep);
      
      uint32_t cur_deletable_pos = Deletable_Sublists.size();
      if(cur_deletable_pos) {

        while(p->status == Status_t::Filling) { } // Waits until work p has its dependencies

        DeallocLock.lock();
        
        for (int i = Dones.size(); i != 0; i--) {
          const auto& cur_done = Dones[i-1];
          while(cur_deletable_pos > cur_done.second) {
            const auto& tmp = Deletable_Sublists[--cur_deletable_pos];
            UPCxx_Workitem::Pool.freeLinkedList(tmp.first, tmp.second);
          }
          while (cur_done.first->status == Status_t::Done) { }
        }
 
        while(cur_deletable_pos) {
          const auto& tmp = Deletable_Sublists[--cur_deletable_pos];
          UPCxx_Workitem::Pool.freeLinkedList(tmp.first, tmp.second);
        }
 
        DeallocLock.unlock();

        Deletable_Sublists.clear(); //Needed because it is static!
      }
 
      Dones.clear(); //Needed because it is static!
    }
  
    // Implementation that only erases the trailing Deallocatable workitems
    void simple_clean_worklist(UPCxx_Workitem *lastkeep)
    {
      using Status_t = UPCxx_Workitem::Status_t;

      UPCxx_Workitem *last_workitem;
      volatile UPCxx_Workitem *p, *debug_p = lastkeep;

      task_ctr_t my_tentative_oldest_task = lastkeep->ctr_;
      bool my_tentative_oldest_task_local = lastkeep->where_ == MyRank;

      for(p = static_cast<volatile UPCxx_Workitem*>(lastkeep->next); p != nullptr; p = static_cast<volatile UPCxx_Workitem*>(p->next)) {
        
        if (p->status != Status_t::Deallocatable) {
          const upcxx::intrank_t workitem_place = p->where_;
          
          if ( p->ctr_ < Oldest_task[workitem_place] ) {
            assert(workitem_place != MyRank);
            const_cast<UPCxx_Workitem*>(p)->finish_execution();
          } else {
            if (workitem_place == MyRank) {
              assert(my_tentative_oldest_task > p->ctr_);
              // assert(p->status != Status_t::Deallocatable); can become Deallocatable since the if...
              my_tentative_oldest_task = p->ctr_;
              my_tentative_oldest_task_local = true;
              debug_p = p;
            }

            lastkeep = const_cast<UPCxx_Workitem*>(p);
          }
        }
        
        last_workitem = const_cast<UPCxx_Workitem*>(p);;
      }
      
      UPCxx_Workitem * const to_erase = static_cast<UPCxx_Workitem *>(lastkeep->next);

      p = common_clear(debug_p, my_tentative_oldest_task, my_tentative_oldest_task_local, lastkeep);

      if(to_erase != nullptr) {
        while(p->status == Status_t::Filling) { } // Waits until work p has its dependencies
        DeallocLock.lock();
        UPCxx_Workitem::Pool.freeLinkedList(to_erase, last_workitem);
        DeallocLock.unlock();
      }
    }

    void UPCxx_Workitem::Clean_worklist()
    {
      //Not necessarily true when coming from main_wait
      assert(EraserLock.is_locked()); // || (Upcxx_Waiting != Upcxx_Waiting_t::No));

      const ThreadState_t prev_thread_state = ThreadState;

      UPCxx_Workitem *lastkeep = upcxx_worklist;
      
      if (lastkeep != nullptr) { // for filtered implementations
        assert( prev_thread_state != ThreadState_t::Progress ); //Could be Idle, Service or TaskRun
        assert( prev_thread_state != ThreadState_t::CleanWorklist );
        // (prev_thread_state != ThreadState_t::TaskRun) only implies (enum_thr_spec_father == nullptr)
        //if users cannot try to run tasks from the pool from within tasks, i.e., if tasks behave as
        //atomic units where a task cannot begin in the middle of another task run in the same thread
        assert( ( (prev_thread_state == ThreadState_t::TaskRun) && (enum_thr_spec_father != nullptr) ) ||
                ( (prev_thread_state != ThreadState_t::TaskRun) /*&& (enum_thr_spec_father == nullptr)*/ ) );

        ThreadState = ThreadState_t::CleanWorklist; // Notice: never >1 threads can be simultaneously at CleanWorklist
        DEPSPAWN_PROFILEACTION(upcxx_profile_erases++);

        if (COMPLEX_CLEAN) {
          complex_clean_worklist(lastkeep);
        } else {
          simple_clean_worklist(lastkeep);
        }

      }

      if (!CleanWorklistPendingPosts.empty() && (prev_thread_state == ThreadState_t::Service)) {
        assert(LocalPendingPosts.empty());
        std::swap(LocalPendingPosts, CleanWorklistPendingPosts);
      }

      EraserLock.unlock();
      
      ThreadState = prev_thread_state;
      run_pending_posts();
    }

    void force_clean_worklist_notify() { //only for hack from SparseDistrBlockMatrix
      // This is so that common_clear will notify the most current local oldest task
      Upcxx_Waiting = Upcxx_Waiting_t::Final;
      EraserLock.lock();
      UPCxx_Workitem::Clean_worklist();
      Upcxx_Waiting = Upcxx_Waiting_t::No;
    }

    static void final_remove_useless_workitems()
    { UPCxx_Workitem *q, *qnext, *deallocate_list, *end_deallocate_list, *tentative_Bottom_task;
      deallocate_list = end_deallocate_list = nullptr;

      DeallocLock.lock();

      for(UPCxx_Workitem* p = upcxx_worklist; p != nullptr; p = q) {
        q = static_cast<UPCxx_Workitem*>(p->next);
        if(q != nullptr) {

          const bool remote = !q->is_local();

          if(remote && (q->deps == nullptr) && (q->status < UPCxx_Workitem::Status_t::Done)) {
            q->finish_execution();
          }

          if(q->status == UPCxx_Workitem::Status_t::Deallocatable) {
            qnext = static_cast<UPCxx_Workitem*>(q->next);
            p->next = qnext;
            q->next = deallocate_list;
            if(end_deallocate_list == nullptr) {
              end_deallocate_list = q;
            }
            deallocate_list = q;
            q = p;
          }
        } else {
          tentative_Bottom_task = p;
        }
      }
      
      if (deallocate_list != nullptr) {
        UPCxx_Workitem::Pool.freeLinkedList(deallocate_list, end_deallocate_list);
      }
      
      DeallocLock.unlock();
    }

    static void main_wait(const bool is_final)
    { bool continue_loop = true;
      ConsecutiveRepsCtrl<int> deallocatable_workitems_reps_ctrl;

      if (is_final && (NRanks > 1)) { // for filtered implementations
        upcxx_spawn([]{});
      }
    
      Upcxx_Waiting = is_final ? Upcxx_Waiting_t::Final : Upcxx_Waiting_t::PeriodicStop;
      
      // if in !is_final state, we leave if Upcxx_Done_workitems reaches this value
      const int limit_Done_workitems = UPCxx_Workitem::Ctr - SPAWN_STOP_RATE;

      // Wait for possible erases in curse to finish
      EraserLock.lock();

      if(is_final) {
        final_remove_useless_workitems();
      }
      
      // This below is no longer needed; we now just unlock for cleanness :
      // We erase it for the sake of upcxx_depspawn_notify_finish_task.
      // Notice that there is no need to regain EraserLock to avoid competition
      // with erases coming from finish_execution because Upcxx_Waiting disabled invocations
      // from finish_execution. However we should obtain it before calling Clean_worklist
      // just in case upcxx_depspawn_notify_finish_task is searching something while we erase.
      EraserLock.unlock();
      
      do {

        int nready = 0, deallocatable_workitems = 0;

        for(UPCxx_Workitem* p = upcxx_worklist; p != nullptr; p = static_cast<UPCxx_Workitem*>(p->next)) {
          const auto workitem_status = p->status;

          deallocatable_workitems += (workitem_status == UPCxx_Workitem::Status_t::Deallocatable);

          // p->is_local() is not checked because only local tasks can be Ready
          nready += (workitem_status == UPCxx_Workitem::Status_t::Ready);
        }

        try_upcxx_advance(); // Help progress just in case?

        if (nready) {
          do {
            nready--;
            DEPSPAWN_PROFILEACTION(if(!is_final) upcxx_profile_stop_steal_attempts++);
            if(TP->try_run()) {
              deallocatable_workitems++; // This is going to be deallocatable
              DEPSPAWN_PROFILEACTION(if(!is_final) upcxx_profile_stop_steals++);
              // No need to call try_upcxx_advance() here because the try_run calls finish_execution,
              //which now finishes with a try_upcxx_advance() for local tasks like this
              if(!is_final && ((Upcxx_local_live_tasks <= MIN_LOCAL_TASKS) || (Upcxx_Done_workitems >= limit_Done_workitems))) {
                continue_loop = false;
                break;
              }
            } else {
              break;
            }
          } while ( nready );
        }

        // Wait for possible upcxx_depspawn_notify_finish_task searches in curse to finish
        //no longer needed; they will be exclusive on DeallocLock
        // EraserLock.lock();

        if ((is_final || (deallocatable_workitems > SPAWN_CLEAR_RATE) || deallocatable_workitems_reps_ctrl.report(deallocatable_workitems)) && EraserLock.try_lock()) {
          // we enter even if deallocatable_workitems is 0 in case during the clear we detect finished remote tasks
          DEPSPAWN_PROFILEACTION(if(!is_final) upcxx_profile_stop_advance_erases++);
          UPCxx_Workitem::Clean_worklist();
        }

        if (!continue_loop) {
          Upcxx_Waiting = Upcxx_Waiting_t::No;
          return;
        }
  
        if (is_final) {
          //(upcxx_worklist->next != nullptr) ||
          const auto GLB_debug = get_global_oldest_task(); 
          continue_loop = ( GLB_debug < (UPCxx_Workitem::Ctr - 1) );
        } else {
          continue_loop = (Upcxx_Done_workitems < limit_Done_workitems) && (Upcxx_local_live_tasks > MIN_LOCAL_TASKS);
        }

      } while ( continue_loop );

      if (is_final) {

        // async_copy may have set the minimum Oldest_task to UPCxx_Workitem::Ctr but may be the associated
        //remote tasks have not been released because we exited the main loop before a final Clean_worklist()
        EraserLock.lock();
        UPCxx_Workitem::Clean_worklist();
        
        //Here the upcxx_worklist workitem may be still pending
        wait_for_all();

        if (UPCXX_DEPSPAWN_ACTIVE_WAIT) {
          TP->set_idle_function( nullptr ); //When not in use by runtime, do not interfere with user
        }

        // This allows to serve advances until we make sure everyone finished its working tasks
        upcxx::persona_scope scope(master_persona_mutex, upcxx::master_persona());
        upcxx::barrier();
      }

      Upcxx_Waiting = Upcxx_Waiting_t::No;

    }
   
    
  } //namespace internal

  void upcxx_depspawn_runtime_setup()
  {
    MyRank = upcxx::rank_me();
    NRanks = upcxx::rank_n();
    
    upcxx::liberate_master_persona();

    // This way if there is a single rank, upcxx::progress is never invoked
    ADVANCE_IN_PROGRESS.store(NRanks == 1);
    
    assert(NRanks != 0);

    // DEPSPAWN_PROFILEACTION(
                           if (signal(SIGTERM, catch_function) == SIG_ERR) {
                             fputs("An error occurred while setting a signal handler.\n", stderr);
                           }
                           if (signal(SIGUSR1, catch_function) == SIG_ERR) {
                             fputs("An error occurred while setting a signal handler.\n", stderr);
                           }
   //                      );
    
    if (!DISABLE_RUNTIME_SETUP) {
      
      // this is to set the default values in a single place
      disable_upcxx_depspawn_runtime_setup();
      DISABLE_RUNTIME_SETUP = false;
      
      char *env_var = getenv("UPCXX_DEPSPAWN_ADV_RATE");
      if(env_var != nullptr) set_UPCXX_DEPSPAWN_ADV_RATE(atoi(env_var));
      
      env_var = getenv("UPCXX_DEPSPAWN_CLR_RATE");
      if(env_var != nullptr) set_UPCXX_DEPSPAWN_CLR_RATE(atoi(env_var));
      
      env_var = getenv("UPCXX_DEPSPAWN_CLEAN_TYPE");
      if(env_var != nullptr) set_UPCXX_DEPSPAWN_CLEAN_TYPE(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_MIN_REPORT");
      if(env_var != nullptr) SPAWN_MIN_TASK_REPORT = atoi(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_STOP_RATE");
      if(env_var != nullptr) SPAWN_STOP_RATE = atoi(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_MIN_LOCAL");
      if(env_var != nullptr) MIN_LOCAL_TASKS = atoi(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_MAX_CACHE");
      if ((env_var != nullptr) && (*env_var == '-')){
        upcxx::GlobalRefCache.set_self_tune(true);
        env_var++;
      }
      const size_t max_cache = (env_var == nullptr) ? upcxx::GlobalRefCache.max_size() : atoi(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_SLACK_CACHE");
      const size_t slack_cache = (env_var == nullptr) ? upcxx::GlobalRefCache.slack() : atoi(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_EXACT_MATCH");
      if(env_var != nullptr) EXACT_MATCH_MODE = atoi(env_var);
      
      env_var = getenv("UPCXX_DEPSPAWN_YIELD");
      if(env_var != nullptr) UPCXX_DEPSPAWN_YIELD = atoi(env_var);
  
      env_var = getenv("UPCXX_DEPSPAWN_FWD_NOTICE");
      if(env_var != nullptr) UPCXX_DEPSPAWN_FWD_NOTICE = atoi(env_var);

      env_var = getenv("UPCXX_DEPSPAWN_PREFETCH");
      if(env_var != nullptr) UPCXX_DEPSPAWN_PREFETCH = atoi(env_var);

      env_var = getenv("UPCXX_DEPSPAWN_ACTIVE_WAIT");
      if(env_var != nullptr) UPCXX_DEPSPAWN_ACTIVE_WAIT = atoi(env_var);

      upcxx::GlobalRefCache.set_cache_limits(max_cache, slack_cache);
    }

    if(MIN_LOCAL_TASKS > SPAWN_STOP_RATE) {
      SPAWN_STOP_RATE = MIN_LOCAL_TASKS;
    }

    DEPSPAWN_PROFILEACTION(
                           if(!MyRank && !DISABLE_RUNTIME_SETUP) {
                             print_upcxx_depspawn_runtime_setup();
                           }
                           );
  }

  void disable_upcxx_depspawn_runtime_setup() noexcept
  {
    DISABLE_RUNTIME_SETUP = true;
    SPAWN_ADVANCE_RATE = 4; // (NRanks > 16) ? 1 : 4;
    SPAWN_CLEAR_RATE = 64;
    COMPLEX_CLEAN = false;
    SPAWN_MIN_TASK_REPORT = 64; // std::max<int>(64, NRanks * 4);
    MIN_LOCAL_TASKS = 2 * (Nthreads - 1);
    SPAWN_STOP_RATE = std::min<int>(2048, (MIN_LOCAL_TASKS + 2) * NRanks * 2);
    EXACT_MATCH_MODE = false;
    UPCXX_DEPSPAWN_YIELD = false;
    UPCXX_DEPSPAWN_FWD_NOTICE = true;
    UPCXX_DEPSPAWN_PREFETCH = false;
    UPCXX_DEPSPAWN_ACTIVE_WAIT = false;
  }

  int set_UPCXX_DEPSPAWN_ADV_RATE(int val)
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      setPowerOf2Var(SPAWN_ADVANCE_RATE, val, "ADV_RATE");
    }
    return SPAWN_ADVANCE_RATE;
  }
  
  int get_UPCXX_DEPSPAWN_ADV_RATE() noexcept { return SPAWN_ADVANCE_RATE; }
  
  int set_UPCXX_DEPSPAWN_CLR_RATE(int val)
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      setPowerOf2Var(SPAWN_CLEAR_RATE, val, "CLR_RATE");
    }
    return SPAWN_CLEAR_RATE;
  }
  
  int get_UPCXX_DEPSPAWN_CLR_RATE() noexcept { return SPAWN_CLEAR_RATE; }
  
  const char *set_UPCXX_DEPSPAWN_CLEAN_TYPE(const char *val)
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if (val != nullptr) {
      if (!strcmp(val, "simple") || !strcmp(val, "SIMPLE") || !strcmp(val, "Simple")) {
        COMPLEX_CLEAN = false;
      } else if(!strcmp(val, "complex") || !strcmp(val, "COMPLEX") || !strcmp(val, "Complex")) {
        COMPLEX_CLEAN = true;
      } else {
        fprintf(stderr, "Unsupported value UPCXX_DEPSPAWN_CLEAN_TYPE=%s\n", val);
        exit(EXIT_FAILURE);
      }
    }

    return (COMPLEX_CLEAN ? "COMPLEX" : "SIMPLE");
  }
  
  const char *get_UPCXX_DEPSPAWN_CLEAN_TYPE() noexcept { return (COMPLEX_CLEAN ? "COMPLEX" : "SIMPLE"); }
  
  int set_UPCXX_DEPSPAWN_MIN_REPORT(int val) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      SPAWN_MIN_TASK_REPORT = val;
    }
    return SPAWN_MIN_TASK_REPORT;
  }
  
  int get_UPCXX_DEPSPAWN_MIN_REPORT() noexcept { return SPAWN_MIN_TASK_REPORT; }
  
  int set_UPCXX_DEPSPAWN_STOP_RATE(int val) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      SPAWN_STOP_RATE = val;
    }
    return SPAWN_STOP_RATE;
  }
  
  int get_UPCXX_DEPSPAWN_STOP_RATE() noexcept { return SPAWN_STOP_RATE; }
  
  int set_UPCXX_DEPSPAWN_MIN_LOCAL(int val) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      MIN_LOCAL_TASKS = val;
    }
    return MIN_LOCAL_TASKS;
  }
  
  int get_UPCXX_DEPSPAWN_MIN_LOCAL() noexcept { return MIN_LOCAL_TASKS; }
  
  int set_UPCXX_DEPSPAWN_MAX_CACHE(int val) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      upcxx::GlobalRefCache.set_cache_limits(val, upcxx::GlobalRefCache.slack());
    }
    return upcxx::GlobalRefCache.max_size();
  }
  
  int get_UPCXX_DEPSPAWN_MAX_CACHE() noexcept { return upcxx::GlobalRefCache.max_size(); }
  
  int set_UPCXX_DEPSPAWN_SLACK_CACHE(int val) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    if(val >= 0) {
      upcxx::GlobalRefCache.set_cache_limits(upcxx::GlobalRefCache.max_size(), val);
    }
    return upcxx::GlobalRefCache.slack();
  }
  
  int get_UPCXX_DEPSPAWN_SLACK_CACHE() noexcept { return upcxx::GlobalRefCache.slack(); }
  
  bool set_UPCXX_DEPSPAWN_EXACT_MATCH(bool value) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    EXACT_MATCH_MODE = value;
    return EXACT_MATCH_MODE;
  }
  
  bool get_UPCXX_DEPSPAWN_EXACT_MATCH() noexcept { return EXACT_MATCH_MODE; }
  
  bool set_UPCXX_DEPSPAWN_YIELD(bool value) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    UPCXX_DEPSPAWN_YIELD = value;
    return UPCXX_DEPSPAWN_YIELD;
  }

  bool get_UPCXX_DEPSPAWN_YIELD() noexcept { return UPCXX_DEPSPAWN_YIELD; }
  
  bool set_UPCXX_DEPSPAWN_FWD_NOTICE(bool value) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    UPCXX_DEPSPAWN_FWD_NOTICE = value;
    return UPCXX_DEPSPAWN_FWD_NOTICE;
  }

  bool get_UPCXX_DEPSPAWN_FWD_NOTICE() noexcept { return UPCXX_DEPSPAWN_FWD_NOTICE; }
  
  bool set_UPCXX_DEPSPAWN_PREFETCH(bool value) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    UPCXX_DEPSPAWN_PREFETCH = value;
    return UPCXX_DEPSPAWN_PREFETCH;
  }

  bool get_UPCXX_DEPSPAWN_PREFETCH() noexcept { return UPCXX_DEPSPAWN_PREFETCH; }

  bool set_UPCXX_DEPSPAWN_ACTIVE_WAIT(bool value) noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
    }
    UPCXX_DEPSPAWN_ACTIVE_WAIT = value;
    return UPCXX_DEPSPAWN_ACTIVE_WAIT;
  }

  bool get_UPCXX_DEPSPAWN_ACTIVE_WAIT() noexcept { return UPCXX_DEPSPAWN_ACTIVE_WAIT; }

  void print_upcxx_depspawn_runtime_setup() noexcept
  {
    if(!NRanks) {
      upcxx_depspawn_runtime_setup();
      DEPSPAWN_PROFILEACTION(if(!MyRank && !DISABLE_RUNTIME_SETUP) return;);
    }

    printf("UPCXX_DEPSPAWN_ADV_RATE   =%u\n", SPAWN_ADVANCE_RATE);
    printf("UPCXX_DEPSPAWN_CLR_RATE   =%u\n", SPAWN_CLEAR_RATE);
    printf("UPCXX_DEPSPAWN_CLEAN_TYPE =%s\n", set_UPCXX_DEPSPAWN_CLEAN_TYPE(nullptr));
    printf("UPCXX_DEPSPAWN_MIN_REPORT =%u\n", SPAWN_MIN_TASK_REPORT);
    printf("UPCXX_DEPSPAWN_STOP_RATE  =%u\n", SPAWN_STOP_RATE);
    printf("UPCXX_DEPSPAWN_MIN_LOCAL  =%u\n", MIN_LOCAL_TASKS);
    printf("UPCXX_DEPSPAWN_EXACT_MATCH=%s\n", EXACT_MATCH_MODE ? "YES" : "NO");
    printf("UPCXX_DEPSPAWN_FWD_NOTICE =%s\n", UPCXX_DEPSPAWN_FWD_NOTICE ? "YES" : "NO");
    printf("UPCXX_DEPSPAWN_MAX_CACHE  =%zu\n", upcxx::GlobalRefCache.max_size());
    printf("UPCXX_DEPSPAWN_SLACK_CACHE=%zu\n", upcxx::GlobalRefCache.slack());
    printf("UPCXX_DEPSPAWN_PREFETCH   =%s\n", UPCXX_DEPSPAWN_PREFETCH ? "YES" : "NO");
    printf("UPCXX_DEPSPAWN_ACTIVE_WAIT=%s\n", UPCXX_DEPSPAWN_ACTIVE_WAIT ? "YES" : "NO");
    printf("UPCXX_DEPSPAWN_YIELD      =%s\n", UPCXX_DEPSPAWN_YIELD ? "YES" : "NO");

  }

  void print_upcxx_depspawn_profile_results(bool reset)
  {
    DEPSPAWN_PROFILEACTION(
      constexpr size_t LOG_SIZE = 1024;
      struct log_line_t {  char buffer[LOG_SIZE]; };

      const upcxx::global_ptr<log_line_t> Log_ptr = upcxx::new_<log_line_t>();
      log_line_t& Log = *Log_ptr.local();

      char * buff = Log.buffer;
      const size_t failed_steals = upcxx_profile_steal_attempts - upcxx_profile_steals;
      const size_t failed_stop_steals = upcxx_profile_stop_steal_attempts - upcxx_profile_stop_steals;
      const size_t msgs_sent = upcxx_profile_msgs_sent_ftask + upcxx_profile_msgs_sent_otask;
      const size_t msgs_recv = upcxx_profile_msgs_recv_ftask + upcxx_profile_msgs_recv_otask;
      
      buff += sprintf(buff, "Jobs: %zu Local Jobs: %zu Steals=%zu Failed Steals=%zu Total Erases=%zu Advance Erases=%zu Predead=%zu\n",
                      upcxx_profile_jobs, upcxx_profile_local_jobs, upcxx_profile_steals, failed_steals, upcxx_profile_erases, upcxx_profile_advance_erases, upcxx_profile_predead_tasks);
      buff += sprintf(buff, "Avg Wkitems: %.1f Avg ready Wkitems: %.1f\n", upcxx_profile_avg_workitems, upcxx_profile_avg_ready_workitems);
      buff += sprintf(buff, "Stops: %zu Steals=%zu Failed Steals=%zu Stop Erases=%zu\n", upcxx_profile_stops, upcxx_profile_stop_steals, failed_stop_steals, upcxx_profile_stop_advance_erases);
      buff += sprintf(buff, "Sent Msgs: %zu (f %zu + o %zu) Recv Msgs: %zu (f %zu + o %zu) Too early f: %zu\n",
                      msgs_sent, (size_t)upcxx_profile_msgs_sent_ftask, (size_t)upcxx_profile_msgs_sent_otask,
                      msgs_recv, (size_t)upcxx_profile_msgs_recv_ftask, (size_t)upcxx_profile_msgs_recv_otask,
                      (size_t)upcxx_profile_too_early_ftask);
      buff += sprintf(buff, "Too early checks=%zu cleans=%zu fixed=%zu fixed_other=%zu\n", upcxx_profile_too_early_checks, upcxx_profile_too_early_cleans, upcxx_profile_too_early_fixed, upcxx_profile_too_early_fixedo);
      buff += sprintf(buff, "FastInsert=%zu ExtlFastInsert=%zu\n", upcxx_profile_fast_insert, upcxx_profile_fast_extl_insert);
      
      if(depspawn::internal::ProfiledCache::Profiled_caches != nullptr) {
        for(const auto ptr : (*depspawn::internal::ProfiledCache::Profiled_caches)) {
          buff += ptr->dump_profiling(buff, upcxx::rank_me());
          if(reset) {
            ptr->clear_profiling();
          }
        }
      }
      
      *buff++ = 0;
      assert( (buff - Log.buffer) <= LOG_SIZE );

      upcxx::persona_scope scope(master_persona_mutex, upcxx::master_persona());

      static log_line_t buf_tmp;
      const auto pr_func = [](int i, const upcxx::global_ptr<log_line_t> ptr) {
        return upcxx::rget(ptr, &buf_tmp, 1).then([i] {
         printf("Process %d:\n%s=====================================================\n", i, buf_tmp.buffer);
        });
      };

      upcxx::barrier();

      for (int i = 0; i < upcxx::rank_n(); i++) {
        if(i == upcxx::rank_me()) {
          if(i) {
            upcxx::rpc(0, pr_func, i, Log_ptr).wait();
            
          } else {
            pr_func(0, Log_ptr).wait();
          }
        }
        upcxx::barrier();
      }
                  
      upcxx::delete_<log_line_t>(Log_ptr);

      if(reset) {
        upcxx_profile_stops = 0;
        upcxx_profile_jobs = 0;
        upcxx_profile_local_jobs = 0;
        upcxx_profile_steals = upcxx_profile_stop_steals = 0;
        upcxx_profile_steal_attempts = upcxx_profile_stop_steal_attempts = 0;
        upcxx_profile_avg_workitems = 0.f;
        upcxx_profile_avg_ready_workitems = 0.f;
        upcxx_profile_erases = 0;
        upcxx_profile_predead_tasks = 0;
        upcxx_profile_advance_erases = upcxx_profile_stop_advance_erases = 0;
        upcxx_profile_msgs_sent_ftask = upcxx_profile_msgs_sent_otask = 0;
        upcxx_profile_msgs_recv_ftask = upcxx_profile_msgs_recv_otask = 0;
        upcxx_profile_too_early_ftask = 0;
        upcxx_profile_too_early_checks = upcxx_profile_too_early_cleans = upcxx_profile_too_early_fixed = upcxx_profile_too_early_fixedo = 0;
        upcxx_profile_fast_extl_insert = upcxx_profile_fast_insert = 0;
      }
    ); // END DEPSPAWN_PROFILEACTION
  }
  
  std::mutex& get_master_persona_mutex() noexcept
  {
    return internal::master_persona_mutex;
  }

  void upcxx_wait_for_all()
  { //static auto DoNothing_on_UPCxx_Workitem_ptr = [](UPCxx_Workitem *i) { };

    if(upcxx_worklist != nullptr) { //always true unless no tasks were spawned
      internal::main_wait(true);
      UPCxx_Workitem::Pool.freeLinkedList(upcxx_worklist); //, DoNothing_on_UPCxx_Workitem_ptr);
      upcxx_worklist = nullptr;
      ExactMatchKeyMap.clear();
#ifdef ATOMIC_EXACT_MATCH
      ExactMatchDeleteQueue.clear();
#endif
      std::fill(Oldest_task, Oldest_task + NRanks, First_Task_Id);
      Bottom_task = nullptr;
    }
    Early_finish_notifications.clear(); // Just in case
    DEPSPAWN_DEBUGACTION(Oldest_local_task_found = First_Task_Id);
    UPCxx_Workitem::Ctr = First_Task_Id;
    Upcxx_Done_workitems = First_Task_Id;
    
    assert(Upcxx_local_live_tasks == 0);

    DEPSPAWN_PROFILEACTION(print_upcxx_depspawn_profile_results(true));

    upcxx::GlobalRefCache.clear();
  }
  
} //namespace depspawn
