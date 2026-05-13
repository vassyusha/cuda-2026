#include "gelu_cuda.h"

__global__ void gelu_kernel_fast(float* __restrict__ input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const float coef1 = 0.044715f;
    const float coef2 = 0.79788456f;

    if (idx < n) {
        float x = input[i];
        float x1 = 0.5f * x;
        float x2 = x * x * x;
        float x3 = x + coef1 * x2;
        float x4 = expf(coef2 * x3);
        input[i] = x1 * (1 + x4);
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
    
    std::vector<float> output(size);
    cudaMemcpy(output.data(), input_cuda, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(input_cuda);
    return output;
}
