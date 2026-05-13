#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__device__ inline float4 operator*(const float4& vec, float scalar) {
    return make_float4(vec.x * scalar, vec.y * scalar, vec.z * scalar, vec.w * scalar);
}

__device__ inline float4& operator+=(float4& dest, const float4& addi) {
    dest.x += addi.x;
    dest.y += addi.y;
    dest.z += addi.z;
    dest.w += addi.w;
    return dest;
}

__global__ void NaiveGemm_cu(const float* A, const float* B, float* C, int n) {
    const int n_4 = n / 4;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n_4) return;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int k = 0; k < n; k++) {
        float elem_a = A[row * n + k];
        float4 elem_b = ((float4*)B)[k * n_4 + col];

        sum += elem_b * elem_a;
    }
    
    ((float4*)C)[row * n_4 + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = n * n * sizeof(float);                                 
    static float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    static cudaStream_t stream = nullptr;
    static int mem_sz = 0;

    if (mem_sz != size) {
        if (stream == nullptr) {
            cudaStreamCreate(&stream);
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        mem_sz = size;
    }
    
    cudaMemcpyAsync(d_A, a.data(), size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b.data(), size, cudaMemcpyHostToDevice, stream);
                                     
    dim3 threads(16, 16);
    dim3 grid(
        (n + threads.x - 1) / threads.x,
        ((n / 4) + threads.y - 1) / threads.y
    );
    
    NaiveGemm_cu<<<grid, threads, 0, stream>>>(d_A, d_B, d_C, n);
    
    std::vector<float> c(n * n, 0.0f);
    cudaMemcpyAsync(c.data(), d_C, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return c;
}