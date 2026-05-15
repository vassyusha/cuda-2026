#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void naive_gemm_float4_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_vec * 4;

    if (row >= n || col >= n) return;

    float4 c_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll 8
    for (int k = 0; k < n; ++k) {
        float a_val = __ldg(&A[row * n + k]);
        float4 b_vec = reinterpret_cast<const float4*>(&B[k * n + col])[0];
        
        c_vec.x = fmaf(a_val, b_vec.x, c_vec.x);
        c_vec.y = fmaf(a_val, b_vec.y, c_vec.y);
        c_vec.z = fmaf(a_val, b_vec.z, c_vec.z);
        c_vec.w = fmaf(a_val, b_vec.w, c_vec.w);
    }

    reinterpret_cast<float4*>(&C[row * n + col])[0] = c_vec;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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
    dim3 grid((n / 4 + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    naive_gemm_float4_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, n);

    std::vector<float> c(n * n);
    cudaMemcpyAsync(c.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);

    return c;
}