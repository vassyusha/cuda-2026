#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalizeKernel(cufftComplex* data, int total, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int total_complex_numbers = static_cast<int>(input.size()) / 2;
    const int n = total_complex_numbers / batch;
    const size_t bytes = input.size() * sizeof(float);

    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    const int total = n * batch;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    const float scale = 1.0f / static_cast<float>(n);
    normalizeKernel<<<grid_size, block_size>>>(d_data, total, scale);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}