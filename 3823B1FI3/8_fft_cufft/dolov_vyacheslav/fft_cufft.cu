#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void scale_elements_kernel(cufftComplex* __restrict__ data, int count, float factor) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count) {
        data[idx].x *= factor;
        data[idx].y *= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) return {};

    int total_size = input.size();
    int complex_count = total_size / 2;
    int n = complex_count / batch;
    size_t bytes = total_size * sizeof(float);

    static cufftComplex* d_buffer = nullptr;
    static int allocated_size = 0;
    static cufftHandle plan = 0;
    static cudaStream_t stream = nullptr;

    if (total_size > allocated_size) {
        if (d_buffer) cudaFree(d_buffer);
        if (plan) cufftDestroy(plan);

        cudaMalloc(&d_buffer, bytes);
        cufftPlan1d(&plan, n, CUFFT_C2C, batch);
        allocated_size = total_size;
    }

    if (!stream) {
        cudaStreamCreate(&stream);
        cufftSetStream(plan, stream);
    }

    cudaMemcpyAsync(d_buffer, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    cufftExecC2C(plan, d_buffer, d_buffer, CUFFT_FORWARD);
    cufftExecC2C(plan, d_buffer, d_buffer, CUFFT_INVERSE);

    int block = 256;
    int grid = (complex_count + block - 1) / block;
    float factor = 1.0f / static_cast<float>(n);

    scale_elements_kernel<<<grid, block, 0, stream>>>(d_buffer, complex_count, factor);

    std::vector<float> result(total_size);

    cudaMemcpyAsync(result.data(), d_buffer, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return result;
}