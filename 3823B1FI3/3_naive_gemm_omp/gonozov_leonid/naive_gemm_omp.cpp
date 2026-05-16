#include "naive_gemm_omp.h"

#pragma GCC optimize("Ofast")

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> res(n * n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

    #pragma omp simd
            for (int k = 0; k < n; k++) {
                res[i * n + k] += a[i * n + j] * b[j * n + k];
            }
        }
    }
    return res;
}
