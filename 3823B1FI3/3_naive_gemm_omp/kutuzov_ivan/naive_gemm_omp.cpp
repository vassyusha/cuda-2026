#include <vector>
#include <algorithm>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    std::vector<float> c(n * n);

    const float* __restrict A = a.data();
    const float* __restrict B = b.data();
    float*       __restrict C = c.data();

    const int BS = 64;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BS)
    {
        const int i_end = std::min(ii + BS, n);

        for (int kk = 0; kk < n; kk += BS)
        {
            const int k_end = std::min(kk + BS, n);

            for (int jj = 0; jj < n; jj += BS)
            {
                const int j_end = std::min(jj + BS, n);

                for (int i = ii; i < i_end; ++i)
                {
                    for (int k = kk; k < k_end; ++k)
                    {
                        const float aik = A[i * n + k];

                        #pragma omp simd
                        for (int j = jj; j < j_end; ++j)
                        {
                            C[i * n + j] += aik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}