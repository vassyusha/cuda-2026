#include "gelu_cuda.h"

__global__ void gelu_kernel_fast(float* __restrict__ input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const float coef1 = 0.044715f;
    const float coef2 = 1.5957691f; 

    if (idx < n) {
        float x = input[idx];
        float x1 = x * x * x;
        float x2 = coef2 * (x + coef1 * x1);
        input[idx] = x / (1.0f + std::exp(-x2));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int size = input.size();

    std::vector<float> output(size);
    
    float *input_cuda;
    cudaMalloc(&input_cuda, size * sizeof(float));
    cudaMemcpy(input_cuda, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel_fast<<<blocks, threads>>>(input_cuda, size);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), input_cuda, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(input_cuda);
    return output;
}
