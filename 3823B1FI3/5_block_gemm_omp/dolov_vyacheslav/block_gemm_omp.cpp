#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")

#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    const float* __restrict__ pA = a.data();
    const float* __restrict__ pB = b.data();
    float* __restrict__ pC = c.data();

    const int block_size = (n < 64) ? n : 64;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                
                for (int i = ii; i < ii + block_size; ++i) {
                    float* __restrict__ c_row = pC + i * n;
                    const float* __restrict__ a_row = pA + i * n;
                    
                    for (int k = kk; k < kk + block_size; ++k) {
                        float a_val = a_row[k];
                        const float* __restrict__ b_row = pB + k * n;
                        
                        #pragma omp simd
                        for (int j = jj; j < jj + block_size; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }

            }
        }
    }

    return c;
}