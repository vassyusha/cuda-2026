#include "gelu_cuda.h"
#include <cuda_runtime.h>

#include <cmath>

__device__ __forceinline__ float gelu(float x) {
    float _2y = x * 1.595769122f * (1.0f + 0.044715f * x * x);
    float e2y = __expf(_2y);
    return x * (1.0f - 1.0f / (e2y + 1.0f));
}

__global__ void kernel_vec4(int n_vec, float4* d_memory) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 v = d_memory[idx];
        v.x = gelu(v.x);
        v.y = gelu(v.y);
        v.z = gelu(v.z);
        v.w = gelu(v.w);
        d_memory[idx] = v;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();

    int n_padded = n + (4 - n % 4) % 4;
    int n_vec = n_padded / 4;

    static float* d_memory = nullptr;
    static int capacity = 0;
    if (n_padded > capacity) {
        if (d_memory) cudaFree(d_memory);
        cudaMalloc(&d_memory, n_padded * sizeof(float));
        capacity = n_padded;
    }

    std::vector<float> output(n_padded);

    cudaMemcpy(d_memory, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int BLOCK = 512;
    kernel_vec4 <<< (n_vec + BLOCK - 1) / BLOCK, BLOCK >>> (n_vec, reinterpret_cast<float4*>(d_memory));

    cudaMemcpy(output.data(), d_memory, n_padded * sizeof(float), cudaMemcpyDeviceToHost);

    output.resize(n);
    return output;
}