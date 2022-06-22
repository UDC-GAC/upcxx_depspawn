#UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
#Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
#
# Distributed under the MIT License. (See accompanying file LICENSE)

# \file     CMakeLists.txt
# \brief    Builds the cholesky tests
# \author   Basilio B. Fraguela <basilio.fraguela@udc.es>

set( CHOLESLY_SOURCES choleskyupc.cpp init.cpp cholesky_upcdsp_barriered.cpp cholesky_upcdsp.cpp cholesky_upcdsp_filtered.cpp cholesky_upcdsp_seq.cpp save_matrix.cpp )

add_executable( cholesky_upcdsp ${CHOLESLY_SOURCES} )
target_link_libraries( cholesky_upcdsp ${BLAS_LIBRARIES} )
add_dependencies( benchmarks cholesky_upcdsp )

