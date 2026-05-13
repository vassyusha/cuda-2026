#include <vector>
#include <cmath>
#include <omp.h>

#include "gelu_omp.h"

const float pi_c = 0.797884F;
const float c = 0.044715F;

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int size = static_cast<int>(input.size());
    std::vector<float> result(size);

    const float* __restrict in = input.data();
    float* __restrict out = result.data();

    const float scale = 2.0F * pi_c;

    #pragma omp parallel for simd schedule(static) num_threads(omp_get_max_threads())
    for (int i = 0; i < size; ++i)
    {
        float x = in[i];
        float x3 = x * x * x;
        float z = scale * (x + c * x3);
        float exp_val = expf(z);
        out[i] = x * exp_val / (exp_val + 1.0F);
    }
    return result;
}