#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void kernel(float* input, float* output, int n) 
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id < n) 
    {
        float x = input[thread_id];
        float argument = 1.595769f * (x + 0.044715f * x * x * x);
        output[thread_id] = x - x / (expf(argument) + 1.0f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) 
{
    float* gpu_input;
    float* gpu_output;
    int n = input.size();
    cudaMalloc(&gpu_input, n * sizeof(float));
    cudaMalloc(&gpu_output, n * sizeof(float));

    cudaMemcpy(gpu_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    kernel<<<grid_size, block_size>>> (gpu_input, gpu_output, n);

    std::vector<float> output(n);

    cudaMemcpy(output.data(), gpu_output, n  * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_output);

    return output;
}
