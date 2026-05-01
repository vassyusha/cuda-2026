#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalizeKernel(float* data, int complex_count, float norm_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < complex_count) {
        data[2*idx] *= norm_factor;
        data[2*idx+1] *= norm_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int size = input.size();
    int complex_count = size / 2;
    int n = complex_count / batch;
    
    size_t bytes = size * sizeof(float);
    
    float* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    
    cudaStreamSynchronize(stream);

    cufftSetStream(plan, stream);

    cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_data), 
                 reinterpret_cast<cufftComplex*>(d_data), CUFFT_FORWARD);
    cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_data), 
                 reinterpret_cast<cufftComplex*>(d_data), CUFFT_INVERSE);
    
    int block_size = 256;
    int grid_size = (complex_count + block_size - 1) / block_size;
    float scale = 1.0f / static_cast<float>(n);
    
    normalizeKernel<<<grid_size, block_size, 0, stream>>>(d_data, complex_count, scale);
    
    std::vector<float> output(size);
    cudaMemcpyAsync(output.data(), d_data, bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    
    return output;
}