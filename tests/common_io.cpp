/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include <mutex>

static std::mutex  My_io_mutex; // This is only for serializing parallel prints

#define LOG(...)   do{ const std::lock_guard<std::mutex> lock(My_io_mutex); std::cerr << 'P' << upcxx::rank_me() << ' ' << __VA_ARGS__ << std::endl; } while(0)
