#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")

#include "block_gemm_omp.h"
#include <algorithm>
#include <cstddef>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> res(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);
    
    constexpr size_t block_size = 128;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < static_cast<size_t>(n); i += block_size) {
        const size_t end_i = std::min(i + block_size, static_cast<size_t>(n));
        
        for (size_t j = 0; j < static_cast<size_t>(n); j += block_size) {
            const size_t end_j = std::min(j + block_size, static_cast<size_t>(n));
            
            for (size_t k = 0; k < static_cast<size_t>(n); k += block_size) {
                const size_t end_k = std::min(k + block_size, static_cast<size_t>(n));
                
                for (size_t I = i; I < end_i; ++I) {
                    float* res_row = res.data() + I * static_cast<size_t>(n);
                    const float* a_row = a.data() + I * static_cast<size_t>(n);
                    
                    for (size_t K = k; K < end_k; ++K) {
                        const float a_val = a_row[K];
                        const float* b_row = b.data() + K * static_cast<size_t>(n);
                        
                        #pragma omp simd
                        for (size_t J = j; J < end_j; ++J) {
                            res_row[J] += a_val * b_row[J];
                        }
                    }
                }
            }
        }
    }
    
    return res;
}
