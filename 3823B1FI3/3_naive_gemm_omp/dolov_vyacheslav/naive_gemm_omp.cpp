#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")

#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    const float* __restrict__ pA = a.data();
    const float* __restrict__ pB = b.data();
    float* __restrict__ pC = c.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float* c_row = pC + i * n;
        const float* a_row = pA + i * n;
        
        for (int k = 0; k < n; ++k) {
            float val_a = a_row[k];
            const float* b_row = pB + k * n;
            
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                c_row[j] += val_a * b_row[j];
            }
        }
    }

    return c;
}