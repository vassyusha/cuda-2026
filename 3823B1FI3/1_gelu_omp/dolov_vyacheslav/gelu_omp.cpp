#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx2,fma")

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    const float* __restrict__ in_ptr = input.data();
    float* __restrict__ out_ptr = output.data();
    const float alpha = -1.59576912f; 
    const float beta = -0.07135482f;

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        float x = in_ptr[i];
        float x2 = x * x;

        float exp_arg = x * (alpha + beta * x2);
        
        out_ptr[i] = x / (1.0f + std::exp(exp_arg));
    }

    return output;
}