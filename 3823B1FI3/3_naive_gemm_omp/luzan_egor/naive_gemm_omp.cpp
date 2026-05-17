#include <vector>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float acum = 0.0f;
            for (int k = 0; k < n; k++)
                acum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = acum;
        }

    return c;
}