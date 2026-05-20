#ifdef __GNUC__
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#endif

#include <vector>
#include <cmath>
#include <omp.h>

#include "gelu_omp.h"

constexpr float coeff_1 = 1.59576912f;
constexpr float coeff_2 = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int sz = static_cast<int>(input.size());
    std::vector<float> res(sz);


#ifdef __GNUC__
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < sz; i++)
    {
        float x = input[i];

        float x2 = x * x;
        float exp_arg = coeff_1 * x * (1.0f + coeff_2 * x2);
        float e = expf(exp_arg);
        res[i] = x * e / (e + 1.0f);
    }
    return res;
}