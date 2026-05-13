#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void normalize_kernel(float* data, int total, float mult) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        data[i] *= mult;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) 
{
    int total = static_cast<int>(input.size());
    if (total == 0) return {};

    int n = total / (2 * batch);
    size_t bytes = total * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* h_pinned_in = nullptr;
    float* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, bytes);
    cudaMallocHost(&h_pinned_out, bytes);

    std::copy(input.begin(), input.end(), h_pinned_in);

    cufftComplex* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    cudaMemcpyAsync(d_data, h_pinned_in, bytes, cudaMemcpyHostToDevice, stream);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftSetStream(plan, stream);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    const int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    normalize_kernel<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<float*>(d_data), total, 1.0f / n);

    cudaMemcpyAsync(h_pinned_out, d_data, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::vector<float> result(h_pinned_out, h_pinned_out + total);

    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);
    cudaFree(d_data);

    return result;
}