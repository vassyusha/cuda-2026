#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                const int n) {
    std::vector<float> c(n * n, 0.0f);
    const int kBlock = 64;

    #pragma omp parallel for 
    for (int I = 0; I < n; I += kBlock)
        for (int J = 0; J < n; J += kBlock)
            for (int K = 0; K < n; K += kBlock) {
                const int i_end = (I + kBlock < n) ? I + kBlock : n; 
                const int j_end = (J + kBlock < n) ? J + kBlock : n;
                const int k_end = (K + kBlock < n) ? K + kBlock : n;

                for (int i = I; i < i_end; ++i)
                    for (int k = K; k < k_end; ++k) {
                        const float a_ik = a[i * n + k];
                        for (int j = J; j < j_end; ++j)
                            c[i * n + j] += a_ik * b[k * n + j];
                    }
            }
    return c;
}