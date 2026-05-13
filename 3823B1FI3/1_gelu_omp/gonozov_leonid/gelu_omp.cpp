// Compiler flags used: -O3 -mavx2 -ffast-math -fopenmp -flto

#pragma GCC optimize("Ofast")

#include "gelu_omp.h"

#include <cmath>

#pragma GCC target("avx2,fma")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    const float coef1 = 0.044715f;
    const float coef2 = 0.79788456f;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        float x = input[i];
        float x1 = 0.5f * x;
        float x2 = x * x * x;
        float x3 = x + coef1 * x2;
        float x4 = expf(coef2 * x3);
        output[i] = x1 * (1 + x4);
    }
    
    return output;
}
