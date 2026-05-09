#ifdef __GNUC__
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#endif

#include "block_gemm_omp.h"
#include <algorithm>
#include <vector>
#include <omp.h>

/* Optimizations
1. Simple block version
2. SIMD, Flags
3. Pointers
4. Tried to find optimal block_size, which is hard without test system
*/

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);

    int block_size = 64;

#pragma omp parallel for schedule(static)
    for (int block_i = 0; block_i < n / block_size; block_i++) {
        for (int block_j = 0; block_j < n / block_size; block_j++) {
            for (int block_k = 0; block_k < n / block_size; block_k++) {
                // Calculating borders
                int i_left = block_i * block_size;
                int i_right = i_left + block_size;
                int j_left = block_j * block_size;
                int j_right = j_left + block_size;
                int k_left = block_k * block_size;
                int k_right = k_left + block_size;

                for (int i = i_left; i < i_right; i++) {
                    float* c_i = &c[i * n];         
                    const float* a_row = &a[i * n];
                    for (int k = k_left; k < k_right; k++) {
                        float a_ik = a_row[k];
                        const float* b_k = &b[k * n];
#ifdef __GNUC__
#pragma omp simd
#endif
                        for (int j = j_left; j < j_right; j++) {
                            c_i[j] += a_ik * b_k[j];  
                        }
                    }
                }
            }
        }
    }
    return c;
}