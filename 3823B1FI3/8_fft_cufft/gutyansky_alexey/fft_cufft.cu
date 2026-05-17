#include <cufft.h>
#include <cuda_runtime.h>
#include <cassert>
#include <complex>

#include "fft_cufft.h"

__global__ void Normalize_cu(cufftComplex *mem, int N, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * batch) {
        float scale = 1.0f / N;
        mem[i].x *= scale;
        mem[i].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int N = input.size() / (2 * batch);
    
    size_t data_size = batch * N * sizeof(cufftComplex);
    static cufftComplex *d_data = nullptr;
    static cudaStream_t stream = nullptr;
    static int mem_sz = 0;

    static cufftHandle plan_forward = 0, plan_inverse = 0;
    
    if (mem_sz != data_size) {
        cudaStreamDestroy(stream);
        cudaStreamCreate(&stream);
    
        cufftDestroy(plan_forward);
        cufftPlan1d(&plan_forward, N, CUFFT_C2C, batch);
        cufftSetStream(plan_forward, stream);

        cufftDestroy(plan_inverse);
        cufftPlan1d(&plan_inverse, N, CUFFT_C2C, batch);
        cufftSetStream(plan_inverse, stream);

        cudaFree(d_data);
        cudaMalloc(&d_data, data_size);
        mem_sz = data_size;
    }
    
    cudaMemcpyAsync(d_data, input.data(), data_size, cudaMemcpyHostToDevice, stream);
    
    cufftExecC2C(plan_forward, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan_inverse, d_data, d_data, CUFFT_INVERSE);

    const int BlockSize = 256;
    int num_blocks = (N + BlockSize - 1) / BlockSize;
    Normalize_cu<<<num_blocks,BlockSize, 0, stream>>>(d_data, N, batch);

    std::vector<float> output(input.size());
    
    cudaMemcpyAsync(output.data(), d_data, data_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return output;
}