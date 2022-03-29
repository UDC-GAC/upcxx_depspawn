/*
 UPCxx_DepSpawn: UPC++ Data Dependent Spawn library
 Copyright (C) 2021-2022 Basilio B. Fraguela. Universidade da Coruna
 
  Distributed under the MIT License. (See accompanying file LICENSE)
*/

#include <cmath>
#include <iostream>
#include <atomic>
//#include "mkl_cblas.h"
//#include "mkl_lapack.h"

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

//
// A tile is a square matrix, used to tile a larger array
//
#ifndef TILESIZE
#define TILESIZE 200
#endif

#define MINPIVOT 1E-12

class tile
{

  public:

    double m_tile[TILESIZE * TILESIZE];
    
    tile()
    {
        memset(m_tile, 0, sizeof(m_tile));
    }

    ~tile() = default;

    double *data() { return (double *)m_tile; }
    const double *data() const { return (const double *)m_tile; }
  
    inline void set(const int i, const int j, double d) { m_tile[i * TILESIZE + j] = d; }
    inline double get(const int i, const int j) const { return m_tile[i * TILESIZE + j]; }

    void dump(double epsilon = MINPIVOT) const
    {
        for (int i = 0; i < TILESIZE; i++)
        {
            for (int j = 0; j < TILESIZE; j++)
            {
                std::cout.width(10);
                double t = get(i, j);
                if (fabs(t) < MINPIVOT)
                    t = 0.0;
                std::cout << t << " ";
            }
            std::cout << std::endl;
        }
    }

    int identity_check(double epsilon = MINPIVOT) const
    {
        int ecount = 0;
        for (int i = 0; i < TILESIZE; i++)
        {
            for (int j = 0; j < TILESIZE; j++)
            {
                double t = get(i, j);
                if (i == j && (fabs(t - 1.0) < epsilon))
                    continue;
                if (fabs(t) < epsilon)
                    continue;

                std::cout << "(" << i << "," << j << "):" << t << std::endl;
                ecount++;
            }
        }
        return ecount;
    }

    int zero_check(double epsilon = MINPIVOT) const
    {
        int ecount = 0;
        for (int i = 0; i < TILESIZE; i++)
        {
            for (int j = 0; j < TILESIZE; j++)
            {
                double t = get(i, j);
                if (fabs(t) < epsilon)
                    continue;
                std::cout << "(" << i << "," << j << "):" << t << std::endl;
                ecount++;
            }
        }
        return ecount;
    }

    bool equal(const tile &t) const
    {
        for (int i = 0; i < TILESIZE; i++)
        {
            for (int j = 0; j < TILESIZE; j++)
            {
                if (get(i, j) != t.get(i, j))
                    return false;
            }
        }
        return true;
    }

    bool similar(const tile &t) const
    {
      for (int i = 0; i < TILESIZE; i++) {
        for (int j = 0; j < TILESIZE; j++) {
          double r_err = get(i, j) - t.get(i, j);
          if ( r_err == 0.0 ) continue;
          
          if (r_err < 0.0 ) r_err = -r_err;
          
          if ( get(i, j) == 0.0) {
            std::cerr << "i=" << i << " j=" << j << " get(i, j) == 0.0\n";
            return false;
          }
          r_err = r_err / get(i, j);
          if(r_err > 1e-5) { // some epsilon
            std::cerr << "i=" << i << " j=" << j << " get(i, j)=" << get(i, j) << " t.get(i, j)=" << t.get(i, j) << '\n';
            return false;
          }
        }
      }
      return true;
    }
  
    //b = inverse(*this)
    void inverse_(tile &b) const
    {
        b = *this;

        for (int n = 0; n < TILESIZE; n++)
        {
            double pivot = b.get(n, n);
            if (fabs(pivot) < MINPIVOT)
            {
                std::cout << "Pivot too small! Pivot( " << pivot << ")" << std::endl;
                b.dump();
                exit(0);
            }

            double pivot_inverse = 1 / pivot;
            double row[TILESIZE];

            row[n] = pivot_inverse;
            b.set(n, n, pivot_inverse);
            for (int j = 0; j < TILESIZE; j++)
            {
                if (j == n)
                    continue;
                row[j] = b.get(n, j) * pivot_inverse;
                b.set(n, j, row[j]);
            }

            for (int i = 0; i < TILESIZE; i++)
            {
                if (i == n)
                    continue;
                double tin = b.get(i, n);
                b.set(i, n, -tin * row[n]);
                for (int j = 0; j < TILESIZE; j++)
                {
                    if (j == n)
                        continue;
                    b.set(i, j, b.get(i, j) - tin * row[j]);
                }
            }
        }
    }

    //b = inverse(*this)
    tile inverse() const
    {
        tile b;      //std::cout << "input to inverse:\n"; dump(0.);
        inverse_(b); //std::cout << "output to inverse:\n"; b.dump(0.);
        return b;
    }

    void S1_compute(tile &A_block)
    {

        //char uplo = 'l';
        //int info;
        //dpotf2(&uplo, (const int *) TILESIZE, const_cast< double * >( (double *)A_block.m_tile ), (const int *) TILESIZE, &info);

        for (int k_b = 0; k_b < TILESIZE; k_b++)
        {
            if (A_block.get(k_b, k_b) <= 0)
            {
                fprintf(stderr, "Not a symmetric positive definite (SPD) matrix\n");
                exit(0);
            }
            A_block.set(k_b, k_b, sqrt(A_block.get(k_b, k_b)));

            for (int j_b = k_b + 1; j_b < TILESIZE; j_b++)
            {
                A_block.set(j_b, k_b, A_block.get(j_b, k_b) / A_block.get(k_b, k_b));
            }
            for (int i_b = k_b + 1; i_b < TILESIZE; i_b++)
            {
                double tmp = A_block.get(i_b, k_b);
                for (int j_bb = k_b + 1; j_bb <= i_b; j_bb++)
                {
                    A_block.set(i_b, j_bb, A_block.get(i_b, j_bb) - (tmp * A_block.get(j_bb, k_b)));
                }
            }
        }
    }

    // Perform triangular system solve on the input tile.
    // Input to this step are the input tile and the output tile of the previous step.
    void S2_compute(const tile &Li_block)
    {
        for (int k_b = 0; k_b < TILESIZE; k_b++)
        {
            for (int i_b = 0; i_b < TILESIZE; i_b++)
            {
                this->set(i_b, k_b, this->get(i_b, k_b) / Li_block.get(k_b, k_b));
            }
            for (int i_b = 0; i_b < TILESIZE; i_b++)
            {
                double tmp = this->get(i_b, k_b);
                for (int j_b = k_b + 1; j_b < TILESIZE; j_b++)
                {
                    this->set(i_b, j_b, this->get(i_b, j_b) - (Li_block.get(j_b, k_b) * tmp));
                }
            }
        }
    }

    void S_dsyrk(const tile &L2_block, tile &A_block)
    {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, TILESIZE, TILESIZE, -1., L2_block.m_tile, TILESIZE, 1., (double *)A_block.m_tile, TILESIZE);
    }

    void S_dgemm(const tile &b, tile &c) const
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, TILESIZE, TILESIZE, TILESIZE,
                    -1., (double *)m_tile, TILESIZE, (double *)b.m_tile, TILESIZE, 1., c.m_tile, TILESIZE);
    }

    // c = this * b
    void multiply_(const tile &b, tile &c) const
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    TILESIZE, TILESIZE, TILESIZE,
                    1., (double *)m_tile, TILESIZE, (double *)b.m_tile, TILESIZE, 0., c.m_tile, TILESIZE);
    }

    tile multiply(const tile &b) const
    {
        tile c;
        multiply_(b, c);
        return c;
    }

    // c = -(this * b)
    void multiply_negate_(const tile &b, tile &c) const
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    TILESIZE, TILESIZE, TILESIZE,
                    -1., (double *)m_tile, TILESIZE, (double *)b.m_tile, TILESIZE, 0., c.m_tile, TILESIZE);
    }

    tile multiply_negate(const tile &b) const
    {
        tile c;
        multiply_negate_(b, c);
        return c;
    }

    // d = this - (b * c)
    tile multiply_subtract(const tile &b, const tile &c) const
    {
        tile d = *this;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    TILESIZE, TILESIZE, TILESIZE,
                    -1., (double *)b.m_tile, TILESIZE, (double *)c.m_tile, TILESIZE, 1., d.m_tile, TILESIZE);
        return d;
    }

    // this =  this - (b * c)
    void multiply_subtract_in_place(const tile &b, const tile &c)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    TILESIZE, TILESIZE, TILESIZE,
                    -1., (double *)b.m_tile, TILESIZE, (double *)c.m_tile, TILESIZE, 1., m_tile, TILESIZE);
    }

    // d = this + (b * c)
    tile multiply_add(const tile &b, const tile &c) const
    {
        tile d = *this;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    TILESIZE, TILESIZE, TILESIZE,
                    1., (double *)b.m_tile, TILESIZE, (double *)c.m_tile, TILESIZE, 1., d.m_tile, TILESIZE);
        return d;
    }

    // this = this + (b * c)
    void multiply_add_in_place(const tile &b, const tile &c)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    TILESIZE, TILESIZE, TILESIZE,
                    1., (double *)b.m_tile, TILESIZE, (double *)c.m_tile, TILESIZE, 1., m_tile, TILESIZE);
    }

    // this = 0.0;
    void zero()
    {
        for (int i = 0; i < TILESIZE; i++)
            for (int j = 0; j < TILESIZE; j++)
                set(i, j, 0.0);
    }

    double hash() const
    {
        double r = m_tile[0];
        int *const pr = (int *)&r;
        const int sz = sizeof(double) / sizeof(int);

        for (int i = 0; i < TILESIZE; i++)
            for (int j = 0; j < TILESIZE; j++)
            {
                const int *const p = (const int *)(m_tile + (i * TILESIZE + j));
                for (int k = 0; k < sz; k++)
                {
                    pr[k] = pr[k] ^ p[k];
                }
            }

        return r;
    }
};
