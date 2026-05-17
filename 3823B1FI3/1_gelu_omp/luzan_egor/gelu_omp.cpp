#include <cmath>
// #include <cstdio>
#include <vector>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2 / pi)
    constexpr float kCoeff = 0.044715f;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float z  = kSqrt2OverPi * (x + kCoeff * x3);
        const float e = std::exp(-2.0f * z);

        output[i] = 0.5f * x * (2.0f / (e + 1.0f));
    }

    return output;
}