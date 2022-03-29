#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)


# \file     CMakeLists.txt
# \brief    Builds the LU tests
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

cmake_minimum_required( VERSION 2.8.7 )

add_executable( lu_upcdsp lu.cpp dgetrf.cpp dgetrf_filtered.cpp dgetrf_seq.cpp )
