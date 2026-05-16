#include "fft_cufft.h"
#include <cufft.h>

/* Optimizations
1. Static device memory reuse
2. Static cuFFT plan with caching
3. Pinned host memory 
4. In-place FFT 
5. Normalization on device
6. Coalesced memory access in normalize kernel
*/

__global__ void normalize_kernel(float* data, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = static_cast<int>(input.size()) / (2 * batch);
    size_t bytes = input.size() * sizeof(float);
    int total = static_cast<int>(input.size());

    static cufftComplex* d_data     = nullptr;
    static size_t device_capacity   = 0;

    if (bytes > device_capacity) {
        if (device_capacity > 0) cudaFree(d_data);
        cudaMalloc(&d_data, bytes);
        device_capacity = bytes;
    }

    static float* h_out = nullptr;
    static size_t pinned_capacity = 0;

    if (bytes > pinned_capacity) {
        if (h_out) cudaFreeHost(h_out);
        cudaMallocHost(&h_out, bytes);
        pinned_capacity = bytes;
    }

    static cufftHandle plan    = 0;
    static int cached_n        = 0;
    static int cached_batch    = 0;

    if (cached_n != n || cached_batch != batch) {
        if (plan) cufftDestroy(plan);
        cufftPlan1d(&plan, n, CUFFT_C2C, batch);
        cached_n     = n;
        cached_batch = batch;
    }

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;
    normalize_kernel<<<grid_size, block_size>>>(reinterpret_cast<float*>(d_data), total, 1.0f / n);

    cudaMemcpy(h_out, d_data, bytes, cudaMemcpyDeviceToHost);

    return std::vector<float>(h_out, h_out + total);
}