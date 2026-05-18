#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void NormalizeKernel(cufftComplex* data, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) 
        return;
    data[idx].x *= scale;
    data[idx].y *= scale;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int n = static_cast<int>(input.size()) / 2 / batch;
    const size_t bytes = input.size() * sizeof(float);

    std::vector<float> output(input.size());

    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    constexpr int kBlock = 256;
    const int totalComplex = n * batch;
    const float scale = 1.0f / static_cast<float>(n);
    NormalizeKernel<<<(totalComplex + kBlock - 1) / kBlock, kBlock>>>(d_data, scale, totalComplex);

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
