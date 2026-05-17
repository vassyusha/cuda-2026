#include <cuda_runtime.h>
#include "gelu_cuda.h"

// #include <cmath>

__global__ void GeluKernel(const float* __restrict__ input, float* __restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) 
        return;

    const float x = input[idx];

    const float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2 / pi)
    const float kCoeff = 0.044715f;

    float x3 = x * x * x;
    float z  = kSqrt2OverPi * (x + kCoeff * x3);
    float e = expf(-2.0f * z);

    output[idx] = 0.5f * x * (2.0f / (e + 1.0f));
}


std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    float* d_input  = nullptr;
    float* d_output = nullptr;
    
    const size_t bytes = n * sizeof(float);

    cudaMalloc(&d_input,  bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int gridSize = (n + block_size - 1) / block_size;
    GeluKernel<<<gridSize, block_size>>>(d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
