/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

///
/// \file     upcxx_depspawn.h
/// \brief    Main header file, and only one that needs to be included in the client code
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>
///

#ifndef __UPCXX_DEPSPAWN_H
#define __UPCXX_DEPSPAWN_H

#include <unordered_map>
#include <set>
#include <upcxx/upcxx.hpp>

#ifdef UPCXX_DEPSPAWN_NO_CACHE
namespace upcxx {
  template <typename T>
  using cached_global_ptr = global_ptr<T>;
}
#else
#include "persistent_cached_global_ptr.h"
#endif

#include "depspawn/depspawn.h"

namespace depspawn {
  
  namespace internal {
    
    typedef int task_ctr_t; ///< Type of the counter that uniquely identifies UPCxx_Workitems

    /// This process rank
    extern upcxx::intrank_t MyRank;

    /// Trait used to detect upcxx::global_ptr
    template<typename T>
    struct is_upcxx_global_ptr : public std::false_type {
      typedef T pointed_type;
    };
    
    template<typename T>
    struct is_upcxx_global_ptr<upcxx::global_ptr<T>> : public std::true_type {
      typedef T pointed_type;
    };

#ifndef UPCXX_DEPSPAWN_NO_CACHE
    template<typename T>
    struct is_upcxx_global_ptr<upcxx::cached_global_ptr<T>> : public std::true_type {
      typedef T pointed_type;
    };
#endif

    void upcxx_depspawn_start_system();
    
    struct upcxx_args_support {
      
      typedef std::unordered_map<char, int> RankToWeight_t;
      
      RankToWeight_t prio_; //maps from ranks to weight in args list
      arg_info *pargs_;
      int nargs_;           //non-ignored global_ptrs arguments (i.e. args with dependencies)
      
      upcxx_args_support() noexcept;

      inline void add_rank_weight(const char rank, const bool wr) { prio_[rank] += (1 + ((int)wr) * 4); }

      void insert_arg(arg_info *n) noexcept;

      upcxx::intrank_t get_rank() const noexcept;

      void clear() noexcept;
    };

    extern upcxx_args_support Static_args_support;
    extern std::atomic<int> Upcxx_Done_workitems; // Only needed for upcxx_cond_spawn

    struct UPCxx_Workitem : public Workitem {
      
      upcxx::intrank_t where_;
      task_ctr_t ctr_;
      //TODO: Try with std::vector<bool> or std::bitset?
      std::set<upcxx::intrank_t> dependent_ranks_; ///< Ranks with tasks that depend on this one.

      static task_ctr_t Ctr;
      
      /// Default constructor, for pool purposes
      UPCxx_Workitem();
      
      /// Build UPCxx_Workitem associated to a list of information on arguments
      /// @internal the args_support is restored so it can be reused
      UPCxx_Workitem(upcxx_args_support& args_support);

      /// Destructor
      /// \internal Only required because of the virtual finish_execution
      virtual ~UPCxx_Workitem() {}
      
      /// Provide task with work for this UPCxx_Workitem and insert it in the worklist
      void insert_in_worklist(AbstractRunner* itask);

      /// Mark the UPCxx_Workitem as completed and clean up as appropriate
      virtual void finish_execution();
      
      void post();

      // TODO: Possible optimization?
      // Mark the execution of a UPCxx_Workitem run in another process
      // void remote_finish_execution();
      
      /// Communicate to needed processes that this task finished
      void remote_awake();
      
      bool is_local() const noexcept { return where_ == internal::MyRank; }

      void insert_depencency_on(UPCxx_Workitem *workitem);

      static void Clean_worklist();

      /// Class pool. false for not tbb scalabble malloc/free
      static Pool_t<UPCxx_Workitem, false> Pool;

    };

    /// Argument analysis base case
    template<bool ONLY_RANK, typename T_it>
    void upcxx_fill_args(upcxx_args_support& args_support) { }
    
    // Recursively analyze the list of types and arguments, filling in the argument list of the Workitem.
    ///Currently, by default, argument types are ignored
    /// \internal the std::enable_if should not be needed. It was included because of problems with gcc 4.9.2
    template<bool ONLY_RANK, typename T_it, typename Head, typename... Tail>
    typename std::enable_if< !is_upcxx_global_ptr< typename std::remove_const< typename std::remove_reference<Head>::type >::type >::value >::type
    upcxx_fill_args(upcxx_args_support& args_support, const Head& h, Tail&&... t) {
      upcxx_fill_args<ONLY_RANK, typename boost::mpl::next<T_it>::type>(args_support, std::forward<Tail>(t)...);
    }

    /// Recursively analyze the list of types and arguments, filling in the argument list of the Workitem.
    ///Case when the argument is a global_ptr/cached_global_ptr
    /// The formal parameter is allowed to NOT be a global_ptr/cached_global_ptr,
    ///but in that case it should be read-only, as non-global_ptr/cached_global_ptr variable modifications
    ///cannot be propagated to the origin of the data without explicit assigments
    template<bool ONLY_RANK, typename T_it, typename T, typename... Tail>
    //typename std::enable_if< is_upcxx_global_ptr< typename std::remove_const< typename std::remove_reference<Head>::type >::type >::value >::type
    void upcxx_fill_args(upcxx_args_support& args_support, const upcxx::global_ptr<T>& h, Tail&&... t) {
      typedef typename boost::mpl::deref<T_it>::type curr_t;            //Formal parameter type
      typedef typename std::remove_reference<curr_t>::type deref_t;
      typedef typename std::remove_const<deref_t>::type deref_const_t;

      // Notice that since the enable_if matched, is_ignored<Head&&>::value must be false
      constexpr bool process_argument = !is_ignored<curr_t&&>::value;

      if(process_argument) {

        typedef typename is_upcxx_global_ptr<deref_const_t>::pointed_type pointed_type;

        constexpr bool formal_param_is_global_ptr = is_upcxx_global_ptr<deref_const_t>::value;
        
        static_assert(!(formal_param_is_global_ptr && std::is_reference<pointed_type>::value), "cannot be global_ptr<T&>");
        
        constexpr bool is_writable =
          formal_param_is_global_ptr ? !std::is_const<pointed_type>::value
                                     : (std::is_reference<curr_t>::value && !std::is_const<deref_t>::value);
        
        static_assert(formal_param_is_global_ptr || !is_writable, "changes to T& from global_ptr<T> are not propagated");

        if (ONLY_RANK) {
          args_support.add_rank_weight(h.where(), is_writable);
        } else {
          arg_info* n = arg_info::Pool.malloc();
          n->size = sizeof(pointed_type);
          n->wr   = is_writable;
          n->addr = (value_t)const_cast<upcxx::global_ptr<T>&>(h).raw_internal(upcxx::detail::internal_only()); //internal hack
          n->rank = h.where();
          n->array_range = nullptr;
          
          // Insert new_arg (n) in order
          args_support.insert_arg(n);
        }
      }

      upcxx_fill_args<ONLY_RANK, typename boost::mpl::next<T_it>::type>(args_support, std::forward<Tail>(t)...);
    }

    /// Freeze global_ptrs instead of keeping a reference to them
    template<typename T>
    struct ref<upcxx::global_ptr<T>&&> {
      static inline upcxx::global_ptr<T> make(upcxx::global_ptr<T>& t) {
        return t;
      }
    };
    
    template<typename T>
    struct ref<upcxx::global_ptr<T>&> {
      static inline upcxx::global_ptr<T>& make(upcxx::global_ptr<T>& t) {
        return t;
      }
    };

#ifndef UPCXX_DEPSPAWN_NO_CACHE
    template<typename T>
    struct ref<upcxx::cached_global_ptr<T>&&> {
      static inline upcxx::cached_global_ptr<T> make(upcxx::cached_global_ptr<T>& t) {
        return t;
      }
    };
    
    template<typename T>
    struct ref<upcxx::cached_global_ptr<T>&> {
      static inline upcxx::cached_global_ptr<T> make(upcxx::cached_global_ptr<T>& t) {
        return t;
      }
    };
#endif

    /** \name Translation of global_ptr arguments of spawn into cached_global_ptr for binding
     */
    ///@{
    
    template<typename T>
    struct ref_cache {
      static inline T& make(T& t) {
        return t;
      }
    };
    
    template<typename T>
    struct ref_cache<upcxx::global_ptr<T>> { //Notice it is not &&
      static inline upcxx::cached_global_ptr<const T> make(upcxx::global_ptr<T>& t) {
        //std::cout << "TalkerDerived<int> ref_cache<Talker_tsp>::make(Talker_tsp& t)\n";
        return t;
      }
    };
    
    template<typename T>
    struct ref_cache<upcxx::global_ptr<T>&> {
      static inline upcxx::cached_global_ptr<const T> make(upcxx::global_ptr<T>& t) {
        //std::cout << "cached_global_ptr<const T> ref_cache<global_ptr<T>&>::make(global_ptr<T>& t)\n";
        return t;
      }
    };
    
    ///@}

  } //namespace internal

  /** @name Runtime library control
   * Modify or retrieve runtime control variables
   */
  ///@{
  
  /// perform runtime set up. This happens automatically on the first spawn.
  /// @internal UPC++ must have been previously initialized
  void upcxx_depspawn_runtime_setup();

  /// Ignore library setup from environment variables and use default setup
  void disable_upcxx_depspawn_runtime_setup() noexcept;
  
  /// Sets UPCXX_SPAWN_ADV_RATE to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_ADV_RATE(int val);
  
  /// Sets UPCXX_SPAWN_CLR_RATE to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_CLR_RATE(int val);
  
  /// Sets UPCXX_DEPSPAWN_CLEAN_TYPE to \c val, unless it is negative. Returns the resulting value of the variable
  const char *set_UPCXX_DEPSPAWN_CLEAN_TYPE(const char *val);

  /// Sets UPCXX_SPAWN_MIN_REPORT to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_MIN_REPORT(int val) noexcept;
  
  /// Sets UPCXX_DEPSPAWN_STOP_RATE to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_STOP_RATE(int val) noexcept;
  
  /// Sets UPCXX_DEPSPAWN_MIN_LOCAL to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_MIN_LOCAL(int val) noexcept;
  
  /// Sets UPCXX_DEPSPAWN_MAX_CACHE to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_MAX_CACHE(int val) noexcept;
  
  /// Sets UPCXX_DEPSPAWN_SLACK_CACHE to \c val, unless it is negative. Returns the resulting value of the variable
  int set_UPCXX_DEPSPAWN_SLACK_CACHE(int val) noexcept;
  
  /// Sets the ExactMatch mode. Returns the resulting value of the variable
  bool set_UPCXX_DEPSPAWN_EXACT_MATCH(bool value) noexcept;

  /// Sets the UPCXX_DEPSPAWN_YIELD variable that controls whether threads try to make progress during waits. Returns the resulting value of the variable
  bool set_UPCXX_DEPSPAWN_YIELD(bool value) noexcept;

  /// Sets the UPCXX_DEPSPAWN_FWD_NOTICE variable that controls whether task terminations are reported ASAP. Returns the resulting value of the variable
  bool set_UPCXX_DEPSPAWN_FWD_NOTICE(bool value) noexcept;

  /// Sets the UPCXX_DEPSPAWN_PREFETCH variable that controls whether prefetchs to the cache are made for read-only data of ready tasks
  bool set_UPCXX_DEPSPAWN_PREFETCH(bool value) noexcept;

  /// Sets the UPCXX_DEPSPAWN_ACTIVE_WAIT variable that controls whether idle threads try to make progress
  bool set_UPCXX_DEPSPAWN_ACTIVE_WAIT(bool value) noexcept;

  int get_UPCXX_DEPSPAWN_ADV_RATE() noexcept;
  
  int get_UPCXX_DEPSPAWN_CLR_RATE() noexcept;
  
  const char *get_UPCXX_DEPSPAWN_CLEAN_TYPE() noexcept;
  
  int get_UPCXX_DEPSPAWN_MIN_REPORT() noexcept;
  
  int get_UPCXX_DEPSPAWN_STOP_RATE() noexcept;
  
  int get_UPCXX_DEPSPAWN_MIN_LOCAL() noexcept;
  
  int get_UPCXX_DEPSPAWN_MAX_CACHE() noexcept;
  
  int get_UPCXX_DEPSPAWN_SLACK_CACHE() noexcept;
  
  bool get_UPCXX_DEPSPAWN_EXACT_MATCH() noexcept;
  
  bool get_UPCXX_DEPSPAWN_YIELD() noexcept;
  
  bool get_UPCXX_DEPSPAWN_FWD_NOTICE() noexcept;

  bool get_UPCXX_DEPSPAWN_PREFETCH() noexcept;
 
  bool get_UPCXX_DEPSPAWN_ACTIVE_WAIT() noexcept;

  /// Prints the variables that control the UPC++ DepSpawn runtime.
  /// Calls upcxx_depspawn_runtime_setup if not done yet.
  void print_upcxx_depspawn_runtime_setup() noexcept;

  ///@}

  /** @name Member function adaptors
   * They modify a member function pointer so that it can be spawned either on a upcxx::global_ptr (by default)
   * or on a upcxx::cached_global_ptr
   */
  ///@{
  
  // Because the method is non-const, there is actually no need to support cached_global_red as of now
  template <template <typename T, upcxx::memory_kind KindSet> class GRefT=upcxx::global_ptr, typename ClassT, typename R, typename... Args>
  auto adapt_member(R (ClassT::* method) (Args...)) -> std::function<R(GRefT<ClassT, upcxx::memory_kind::host>, Args...)>
  {
    auto f = [method](GRefT<ClassT, upcxx::memory_kind::host> gr, Args&&... args) -> R {
      if(gr.is_local()) {
        return (gr.local()->*method)(std::forward<Args>(args)...);
      } else {
        ClassT tmp = upcxx::rget(gr).wait();
        const R res = (tmp.*method)(std::forward<Args>(args)...);
        upcxx::rput(tmp, gr).wait();
        return res;
      }
    };
    return f;
  }

  // Because the method is non-const, there is actually no need to support cached_global_red as of now
  template <template <typename T, upcxx::memory_kind KindSet> class GRefT=upcxx::global_ptr, typename ClassT, typename... Args>
  auto adapt_member(void (ClassT::* method) (Args...)) -> std::function<void(GRefT<ClassT, upcxx::memory_kind::host>, Args...)>
  {
    auto f = [method](GRefT<ClassT, upcxx::memory_kind::host> gr, Args&&... args) -> void {
      if(gr.is_local()) {
        (gr.local()->*method)(std::forward<Args>(args)...);
      } else {
        ClassT tmp = upcxx::rget(gr).wait();
        (tmp.*method)(std::forward<Args>(args)...);
        upcxx::rput(tmp, gr).wait();
      }
    };
    return f;
  }

  template <template <typename T, upcxx::memory_kind KindSet> class GRefT=upcxx::global_ptr, typename ClassT, typename R, typename... Args>
  auto adapt_member(R (ClassT::* method) (Args...) const) -> std::function<R(GRefT<const ClassT, upcxx::memory_kind::host>, Args...)>
  {
    auto f = [method](GRefT<const ClassT, upcxx::memory_kind::host> gr, Args&&... args) -> R {
      if(gr.is_local()) {
        return (gr.local()->*method)(std::forward<Args>(args)...);
      } else {
        return (upcxx::rget(gr).wait().*method)(std::forward<Args>(args)...);
      }
    };
    return f;
  }
  ///@}
  
  
  template<typename Function, typename... Args>
  inline internal::spawn_ret_t upcxx_spawn(const Function& f, Args&&... args) {
    using parameter_types = typename internal::ParameterTypes<Function>::type;
    
    internal::upcxx_fill_args<false, typename boost::mpl::begin<parameter_types>::type>(internal::Static_args_support, std::forward<Args>(args)...);
    internal::upcxx_depspawn_start_system();
    internal::UPCxx_Workitem* w = internal::UPCxx_Workitem::Pool.malloc(internal::Static_args_support);
    depspawn::internal::TaskPool::Task *itask = nullptr;

    if(w->is_local()) {
#ifdef UPCXX_DEPSPAWN_AUTOMATIC_CACHE
      auto fs = typename std::decay<decltype(f)>::type(f);
      const auto caching_lambda = [fs](decltype(internal::ref<Args&&>::make(args))... args_lambda) -> void {
        fs(internal::ref_cache<decltype(internal::ref<Args&&>::make(args))>::make(args_lambda)...);
      };
      itask = internal::TP->build_task(w, caching_lambda, internal::ref<Args&&>::make(args)...);
#else // UPCXX_DEPSPAWN_AUTOMATIC_CACHE
      itask = internal::TP->build_task(w, f, internal::ref<Args&&>::make(args)...);
#endif // UPCXX_DEPSPAWN_AUTOMATIC_CACHE
    }
    
    w->insert_in_worklist(itask);
  }
  
  /// Spawns a functor, but only if there is a single operator()
  template<typename T, typename... Args>
  typename std::enable_if< std::is_reference<T>::value &&
                         ! std::is_member_function_pointer<typename std::remove_reference<T>::type>::value &&
                         ! std::is_function<typename std::remove_reference<T>::type>::value &&
                         ! internal::is_function_object<typename std::remove_reference<T>::type>::value,
                           internal::spawn_ret_t >::type
  upcxx_spawn(T&& functor, Args&&... args) {
    typedef typename std::remove_reference<T>::type base_type;
    return upcxx_spawn(& base_type::operator(), std::forward<T>(functor), std::forward<Args>(args)...);
  }
  
  /// Show profiling information (if compiled with profiling)
  void print_upcxx_depspawn_profile_results(bool reset = false);
  
  /// Get mutex to compite with UPC++ DepSpawn for the master persona
  std::mutex& get_master_persona_mutex() noexcept;

  /// Waits for all tasks to finish
  void upcxx_wait_for_all();

  /// when condition is true, performs a upcxx_spawn.
  /// Otherwise if accepts_cond_rank() is true it records the rank, if only increases Ctr and Done_workitems
  template<typename Function, typename... Args>
  inline internal::spawn_ret_t upcxx_cond_spawn(const bool cond, const Function& f, Args&&... args) {
    if (cond) {
      // if (imyrank) std::cerr << imyrank << ' ' << depspawn::internal::UPCxx_Workitem::Ctr <<std::endl;
      return upcxx_spawn(std::forward<Function>(f), std::forward<Args>(args)...);
    } else {
      internal::UPCxx_Workitem::Ctr++;
      internal::Upcxx_Done_workitems.fetch_add(1, std::memory_order_relaxed);
    }
  }

  /// Specialization of upcxx_cond_spawn for functors
  template<typename T, typename... Args>
  typename std::enable_if< std::is_reference<T>::value &&
                         ! std::is_member_function_pointer<typename std::remove_reference<T>::type>::value &&
                         ! std::is_function<typename std::remove_reference<T>::type>::value &&
                         ! internal::is_function_object<typename std::remove_reference<T>::type>::value,
                           internal::spawn_ret_t >::type
  upcxx_cond_spawn(const bool cond, T&& functor, Args&&... args) {
    typedef typename std::remove_reference<T>::type base_type;
    return upcxx_cond_spawn(cond, & base_type::operator(), std::forward<T>(functor), std::forward<Args>(args)...);
  }

} //namespace depspawn

#define MACRO_UPCXX_COND_SPAWN(cond, ...) {          \
  if(cond) {                                         \
  /*  if (imyrank) std::cerr << imyrank << ' ' << depspawn::internal::UPCxx_Workitem::Ctr << ' ' << #__VA_ARGS__<<std::endl; */                    \
    depspawn::upcxx_spawn(__VA_ARGS__);              \
  } else { internal::UPCxx_Workitem::Ctr++; internal::Upcxx_Done_workitems++; /*No OLDEST_TASK_TOKEN_PASS*/ } \
}

#endif // __UPCXX_DEPSPAWN_H
