#include "block_gemm_omp.h"
#include <algorithm>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    std::vector<float> c(n * n);

    const float* __restrict A = a.data();
    const float* __restrict B = b.data();
    float* __restrict C = c.data();

    const int BI = 64;
    const int BJ = 64;
    const int BK = 64;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < n; ii += BI)
    {
        for (int jj = 0; jj < n; jj += BJ)
        {
            const int i_end = std::min(ii + BI, n);
            const int j_end = std::min(jj + BJ, n);

            for (int kk = 0; kk < n; kk += BK)
            {
                const int k_end = std::min(kk + BK, n);

                for (int i = ii; i < i_end; ++i)
                    for (int k = kk; k < k_end; ++k)
                    {
                        float aik = A[i * n + k];

                        #pragma omp simd
                        for (int j = jj; j < j_end; ++j)
                            C[i * n + j] += aik * B[k * n + j];
                    }
            }
        }
    }
    return c;
}