#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float s_A[32][32];
    __shared__ float s_B[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int r_base = blockIdx.y * 32 + ty * 4;
    int c_base = blockIdx.x * 32 + tx;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int m = 0; m < (n + 31) / 32; ++m) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = ty + i * 8;
            int g_row_A = blockIdx.y * 32 + r;
            int g_col_A = m * 32 + tx;
            s_A[r][tx] = (g_row_A < n && g_col_A < n) ? A[g_row_A * n + g_col_A] : 0.0f;

            int g_row_B = m * 32 + r;
            int g_col_B = blockIdx.x * 32 + tx;
            s_B[r][tx] = (g_row_B < n && g_col_B < n) ? B[g_row_B * n + g_col_B] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            float b_val = s_B[k][tx];
            sum0 = fmaf(s_A[ty * 4 + 0][k], b_val, sum0);
            sum1 = fmaf(s_A[ty * 4 + 1][k], b_val, sum1);
            sum2 = fmaf(s_A[ty * 4 + 2][k], b_val, sum2);
            sum3 = fmaf(s_A[ty * 4 + 3][k], b_val, sum3);
        }

        __syncthreads();
    }

    if (c_base < n) {
        if (r_base + 0 < n) C[(r_base + 0) * n + c_base] = sum0;
        if (r_base + 1 < n) C[(r_base + 1) * n + c_base] = sum1;
        if (r_base + 2 < n) C[(r_base + 2) * n + c_base] = sum2;
        if (r_base + 3 < n) C[(r_base + 3) * n + c_base] = sum3;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) return {};

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static int allocated_n = 0;
    static cudaStream_t stream = nullptr;

    if (n > allocated_n) {
        if (d_A) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
        size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        allocated_n = n;
    }

    if (!stream) {
        cudaStreamCreate(&stream);
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaMemcpyAsync(d_A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(32, 8);
    dim3 grid((n + 31) / 32, (n + 31) / 32);

    block_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, n);

    std::vector<float> c(n * n);
    cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    return c;
}