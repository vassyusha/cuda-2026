#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

__global__ void gelu_cuda_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const float alpha = -1.59576912f;
    const float beta = -0.07135482f;
    
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + __expf(x * (alpha + beta * (x * x))));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static int allocated_capacity = 0;
    static cudaStream_t stream = nullptr;

    if (n > allocated_capacity) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        allocated_capacity = n;
    }

    if (!stream) {
        cudaStreamCreate(&stream);
    }

    cudaMemcpyAsync(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads_per_block = 256;
    int blocks_per_grid = std::min(4096, (n + threads_per_block - 1) / threads_per_block);

    gelu_cuda_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_in, d_out, n);

    std::vector<float> output(n);

    cudaMemcpyAsync(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}