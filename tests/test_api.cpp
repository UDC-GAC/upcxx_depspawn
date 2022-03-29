/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/
/// \file     test_api.cpp
/// \brief    Tests spawning of several kinds of functions
/// \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

#include <cstdio>
#include <thread>
#include <chrono>
#include "upcxx_depspawn/upcxx_depspawn.h"
#include "SharedArray.h"
#include "common_io.cpp"  // This is only for serializing parallel prints

using namespace depspawn;

struct Data {

  int value_;
  
  Data(int input = 0) :
  value_(input)
  { }
  
  operator int() const { return value_; }
  
  Data& inc() { value_++; return *this; }
  Data& dec() { value_--; return *this; }

  void inc_ret_void() { value_++; }

  //This overload confuses adapt_member and cannot be used directly because
  //there is no implicit conversion from upcxx::global_ptr<Data> argument to Data parameter
  //Data& add(const Data& other) { value_ += other.value_; return *this; }

  Data& add(upcxx::global_ptr<const Data> other) {
    const auto tmp = upcxx::rget(other).wait();
    value_ += tmp.value_;
    return *this;
  }

  void change_both(Data& other) { value_++; other.value_++; }
  
  void change_both_fixed(upcxx::global_ptr<Data> other) {
    value_++;
    upcxx::rput(Data(static_cast<int>(upcxx::rget(other).wait()) + 1), other).wait();
    
  }
  
  // and this can only be linked to adapt_member<upcxx::cached_global_ptr> because it is marked const!
  void add_me_to(upcxx::global_ptr<Data> other) const {
    upcxx::rput(Data(static_cast<int>(upcxx::rget(other).wait()) + value_), other).wait();
  }
  
  void print() const { std::cout << value_ << '\n'; }
};

  
struct Functor {
  void operator()(upcxx::global_ptr<Data> d) {
    upcxx::rput(upcxx::rget(d).wait().inc(), d).wait();
  }
};

struct Functor2 {
  void operator()(upcxx::global_ptr<Data> d, upcxx::global_ptr<const Data> a)
  {
    //upcxx::rput(upcxx::rget(d).wait().add(upcxx::rget(a).wait()), d).wait();
    upcxx::rput(upcxx::rget(d).wait().add(a), d).wait();
  };
};

SharedArray<Data> Table;
std::chrono::high_resolution_clock::time_point t0, t1;

void f()
{
  LOG("Node " << upcxx::rank_me() << " runs f on " << std::this_thread::get_id());
}

void print_table(const char *str = nullptr)
{
  upcxx::barrier();
  if (!upcxx::rank_me()) {
    if (str != nullptr) {
      fprintf(stderr, "%s\n", str);
    }
    for (int i = 0; i < upcxx::rank_n(); ++i) {
      fprintf(stderr, "Table[%d]=%d\n", i, (int)(upcxx::rget(Table[i]).wait()));
    }
    fprintf(stderr, "=========\n");
  }
  upcxx::barrier();
}

/*
template <typename ClassT, typename R, typename... Args>
auto adapt_member(R (ClassT::* method) (Args...)) -> std::function<R(upcxx::global_ptr<ClassT>, Args...)>
{
  auto f = [method](upcxx::global_ptr<ClassT> gr, Args&&... args) -> R {
    if(gr.where() == upcxx::myrank()) { // == depspawn::internal::MyRank //after initialization
      return (gr.raw_ptr()->*method)(std::forward<Args>(args)...);
    } else {
      ClassT tmp = gr.get();
      R res = (tmp.*method)(std::forward<Args>(args)...);
      gr = tmp;
      return res;
    }
  };
  return f;
}

template <typename ClassT, typename... Args>
auto adapt_member(void (ClassT::* method) (Args...)) -> std::function<void(upcxx::global_ptr<ClassT>, Args...)>
{
  auto f = [method](upcxx::global_ptr<ClassT> gr, Args&&... args) -> void {
    if(gr.where() == upcxx::myrank()) { // == depspawn::internal::MyRank //after initialization
      (gr.raw_ptr()->*method)(std::forward<Args>(args)...);
    } else {
      ClassT tmp = gr.get();
      (tmp.*method)(std::forward<Args>(args)...);
      gr = tmp;
    }
  };
  return f;
}

template <typename ClassT, typename R, typename... Args>
auto adapt_member(R (ClassT::* method) (Args...) const) -> std::function<R(upcxx::global_ptr<const ClassT>, Args...)>
{
  auto f = [method](upcxx::global_ptr<const ClassT> gr, Args&&... args) -> R {
    if(gr.where() == upcxx::myrank()) { // == depspawn::internal::MyRank //after initialization
      return (gr.raw_ptr()->*method)(std::forward<Args>(args)...);
    } else {
      return (gr.get().*method)(std::forward<Args>(args)...);
    }
  };
  return f;
}
*/

/*
// Because the method is non-const, there is actually no need to support cached_global_red as of now
template <template <typename T, class place_t = upcxx::rank_t > class GRefT =upcxx::global_ptr, typename ClassT, typename R, typename... Args>
auto adapt_member(R (ClassT::* method) (Args...)) -> std::function<R(GRefT<ClassT>, Args...)>
{
  auto f = [method](GRefT<ClassT> gr, Args&&... args) -> R {
    if(gr.where() == upcxx::myrank()) { // == depspawn::internal::MyRank //after initialization
      return (gr.raw_ptr()->*method)(std::forward<Args>(args)...);
    } else {
      ClassT tmp = gr.get();
      R res = (tmp.*method)(std::forward<Args>(args)...);
      gr = tmp;
      return res;
    }
  };
  return f;
}

// Because the method is non-const, there is actually no need to support cached_global_red as of now
template <template <typename T, class place_t = upcxx::rank_t > class GRefT =upcxx::global_ptr, typename ClassT,typename... Args>
auto adapt_member(void (ClassT::* method) (Args...)) -> std::function<void(GRefT<ClassT>, Args...)>
{
  auto f = [method](GRefT<ClassT> gr, Args&&... args) -> void {
    if(gr.where() == upcxx::myrank()) { // == depspawn::internal::MyRank //after initialization
      (gr.raw_ptr()->*method)(std::forward<Args>(args)...);
    } else {
      ClassT tmp = gr.get();
      (tmp.*method)(std::forward<Args>(args)...);
      gr = tmp;
    }
  };
  return f;
}

template <template <typename T, class place_t = upcxx::rank_t > class GRefT =upcxx::global_ptr, typename ClassT, typename R, typename... Args>
auto adapt_member(R (ClassT::* method) (Args...) const) -> std::function<R(GRefT<const ClassT>, Args...)>
{
  auto f = [method](GRefT<const ClassT> gr, Args&&... args) -> R {
    if(gr.where() == upcxx::myrank()) { // == depspawn::internal::MyRank //after initialization
      return (gr.raw_ptr()->*method)(std::forward<Args>(args)...);
    } else {
      return (gr.get().*method)(std::forward<Args>(args)...);
    }
  };
  return f;
}
*/

void free_inc(upcxx::global_ptr<Data> value)
{
  upcxx::rput(upcxx::rget(value).wait().inc(), value).wait();
}

void free_add(upcxx::global_ptr<Data> v, upcxx::global_ptr<const Data> a)
{
  //upcxx::rput(upcxx::rget(v).wait().add(upcxx::rget(a).wait()), v).wait();
  upcxx::rput(upcxx::rget(v).wait().add(a), v).wait();
}

void sequential_tests(int myrank)
{
  upcxx::global_ptr<Data> const gr = upcxx::new_<Data>();
  upcxx::global_ptr<const Data> const c_gr = gr;

  //without std::ref it would be binded by value, not updating the original 'd'
  //auto kk = std::bind(&Data::inc, std::ref(d));
  
  //auto kk = std::bind(&Data::inc, gr); // Does not compile because &data != upcxx::global_ptr<Data>
  
  auto kk = std::bind(adapt_member(&Data::inc), gr);
  kk();
  
  auto kkv = std::bind(adapt_member(&Data::inc_ret_void), gr);
  kkv();
  
  auto kk_c = std::bind(adapt_member(&Data::inc), c_gr); //can build
//  Although it can be built, the invocation
//   kk_c();
//   correctly fails because:
//   bind<std::::function<Data &(upcxx::global_ptr<Data, unsigned int>)>, upcxx::global_ptr<const Data, unsigned int> &>

  auto my_print = std::bind(adapt_member(&Data::print), gr);
  my_print();

  auto my_c_print = std::bind(adapt_member(&Data::print), c_gr);
  my_c_print();

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; ++i) {
    kk();
  }
  t1 = std::chrono::high_resolution_clock::now();
  std::cerr << (int)*gr.local() << " in " << std::chrono::duration <double>(t1 - t0).count() << '\n';

  auto method = &Data::inc;
  auto kk2 = std::bind([method](upcxx::global_ptr<Data> gr) -> Data& {
    assert(gr.where() == upcxx::rank_me()); // == depspawn::internal::MyRank //after initialization
    return (gr.local()->*method)();
  }, gr);
  for (int i = 0; i < 1000000; ++i) {
    kk2();
  }
  t1 = std::chrono::high_resolution_clock::now();
  std::cerr << (int)*gr.local() << " in " << std::chrono::duration <double>(t1 - t0).count() << '\n';

  upcxx::delete_<Data>(gr);
}

  
int main(int argc, char **argv)
{ bool test_ok = true;

  const int nthreads = (argc > 1) ? strtoul(argv[1], 0, 0) : 2;

  set_threads(nthreads);

  upcxx::init();
  
  const int myrank = upcxx::rank_me();
  const int NRanks  = upcxx::rank_n();

  if (!myrank) {
    std::cerr << "Using " << NRanks << " Procs x " << nthreads << " threads Cache: " << upcxx::cached_global_ptr<const Data>::cacheTypeName() << std::endl;
    sequential_tests(myrank);
  }

  Table.init(NRanks);
  int * const correct_values = new int [NRanks];

  upcxx::barrier();
  
  upcxx::rput(Data(myrank), Table[myrank]).wait();

  print_table();
  
  upcxx_spawn(f);

  auto lamba1 = [](upcxx::global_ptr<Data> d) {
    upcxx::rput(upcxx::rget(d).wait().inc(), d).wait();
  };
  std::function<void(upcxx::global_ptr<Data>)> std_fn = lamba1;

  Functor functor_object;
  Functor2 functor_object2;

  constexpr int NTestsLoop1 = 5;

  for (int i = 0; i < NRanks; ++i) {
    upcxx_spawn(free_inc, Table[i]);
    upcxx_spawn(lamba1, Table[i]);
    upcxx_spawn(std_fn, Table[i]);
    upcxx_spawn(functor_object, Table[i]);
    upcxx_spawn(adapt_member(&Data::inc), Table[i]);
  }
  
  upcxx_wait_for_all();
  
  {
    upcxx::persona_scope scope(upcxx::master_persona());
  
    if (!myrank) {
      std::cerr << "completed upcxx_wait_for_all1()\n";
    }
    
    print_table();
    
    for (int i = 0; i < NRanks; ++i) {
      correct_values[i] = i + NTestsLoop1;
      int tmp = upcxx::rget(Table[i]).wait();
      test_ok = test_ok && (tmp == correct_values[i]);
      if (tmp != correct_values[i]) {
        std::cerr << "Table[" << i << "]=" << tmp << " != " << correct_values[i] << std::endl;
      }
    }
    
    upcxx::barrier();
  }
  
  auto lamba2 = [](upcxx::global_ptr<Data> v, upcxx::global_ptr<const Data> a) {
    //upcxx::rput(upcxx::rget(v).wait().add(upcxx::rget(a).wait()), v).wait();
    upcxx::rput(upcxx::rget(v).wait().add(a), v).wait();
  };
  std::function<void(upcxx::global_ptr<Data> v, upcxx::global_ptr<const Data> a)> std_fn2 = lamba2;

  constexpr int NTestsLoop2 = 4;

  for (int i = 0; i < NRanks; ++i) {
    upcxx_spawn(free_add, Table[i], Table[(i+1) % NRanks]);
    upcxx_spawn(lamba2, Table[i], Table[(i+1) % NRanks]);
    upcxx_spawn(std_fn2, Table[i], Table[(i+1) % NRanks]);
    upcxx_spawn(adapt_member(&Data::add), Table[i], Table[(i+1) % NRanks]);
  }
  
  upcxx_wait_for_all();

  {
    upcxx::persona_scope scope(upcxx::master_persona());

    if (!myrank) {
      std::cerr << "completed upcxx_wait_for_all2()\n";
    }
    
    print_table();

    for (int i = 0; i < NRanks; ++i) {
      correct_values[i] += NTestsLoop2 * correct_values[(i+1) % NRanks];
      /*
       int should_be = (i + NTestsLoop1) +
                    NTestsLoop2 * ( (i != (NRanks-1))
                                    ? (i + NTestsLoop1 + 1)
                                    : NTestsLoop1 + NTestsLoop2 * (NTestsLoop1 + 1));
       */
      int tmp = upcxx::rget(Table[i]).wait();
      test_ok = test_ok && (tmp == correct_values[i]);
      if (tmp != correct_values[i]) {
        std::cerr << "Table[" << i << "]=" << tmp << " != " << correct_values[i] << std::endl;
      }
    }

    upcxx::barrier();
  }
  
  Data local_data(0);
  for (int i = 0; i < NRanks; ++i) {

    //Does not compile, which is good for safeness
    //upcxx_spawn(adapt_member(&Data::change_both), Table[i], Table[(i+1) % NRanks]);
  
    upcxx_spawn(adapt_member(&Data::change_both), Table[i], local_data);
    upcxx_spawn(adapt_member(&Data::change_both_fixed), Table[i], Table[(i+1) % NRanks]);
  }
  upcxx_wait_for_all();
  
  {
    upcxx::persona_scope scope(upcxx::master_persona());

    if (!myrank) {
      std::cerr << "completed upcxx_wait_for_all3()\n";
      std::cerr << "local_data=" << static_cast<int>(local_data) << '\n';
    }
  
    print_table();
  
    for (int i = 0; i < NRanks; ++i) {
      correct_values[i] += 3;
      int tmp = upcxx::rget(Table[i]).wait();
      test_ok = test_ok && (tmp == correct_values[i]);
      if (tmp != correct_values[i]) {
        std::cerr << "Table[" << i << "]=" << tmp << " != " << correct_values[i] << std::endl;
      }
    }

    // it must be 1 because only ONE upcxx_spawn was run locally
    if (static_cast<int>(local_data) != 1) {
      test_ok = false;
      std::cerr << "local_data=" << static_cast<int>(local_data) << " != " << 1 << std::endl;
    }
  }
  
  for (int i = 0; i < NRanks; ++i) {
    /* This must/cannot be done because upcxx::cached_global_ptr should only store read-only references
    upcxx_spawn(adapt_member<upcxx::cached_global_ptr>(&Data::change_both), Table[i], local_data);
    upcxx_spawn(adapt_member<upcxx::cached_global_ptr>(&Data::change_both_fixed), Table[i], Table[(i+1) % NRanks]);
     */
    upcxx_spawn(adapt_member<upcxx::cached_global_ptr>(&Data::add_me_to), Table[(i+1) % NRanks], Table[i]);
  }

  upcxx_wait_for_all();
  
  {
    
    upcxx::persona_scope scope(upcxx::master_persona());
    
    if (!myrank) {
      std::cerr << "completed upcxx_wait_for_all4()\n";
    }

    print_table();
    
    for (int i = 0; i < NRanks; ++i) {
      correct_values[i] += correct_values[(i+1) % NRanks];
      int tmp = upcxx::rget(Table[i]).wait();
      test_ok = test_ok && (tmp == correct_values[i]);
      if (tmp != correct_values[i]) {
        std::cerr << "Table[" << i << "]=" << tmp << " != " << correct_values[i] << std::endl;
      }
    }
    
    upcxx::finalize();
  }

  if (!myrank) {
    std::cout << "TEST " << (test_ok ? "SUCCESSFUL" : "UNSUCCESSFUL") << std::endl;
  }
  
  delete [] correct_values;

  return !test_ok;
}
