#ifdef __GNUC__
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#endif

/* Optimizations
1. Parralelism, SIMD, Flags
2. Better cache usage (IJK -> IKJ)
3. Calculating a[i][k] out of j-loop
4. SIMD in inner loop
5. Using pointers to (MAYBE) simplify data access
6. Unrolled loop 
Did it by my hands because it help when I was testing in on my laptop
*/

#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a_ik = a[i * n + k];
            float* c_i = &c[i * n];
            const float* b_k = &b[k * n];

#ifdef __GNUC__
#pragma omp simd
#endif
            for (int j = 0; j < n; j += 4) {
                c_i[j] += a_ik * b_k[j];
                c_i[j + 1] += a_ik * b_k[j + 1];
                c_i[j + 2] += a_ik * b_k[j + 2];
                c_i[j + 3] += a_ik * b_k[j + 3];
            }
        }
    }

    return c;
}