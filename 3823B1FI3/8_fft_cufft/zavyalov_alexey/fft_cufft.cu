#include "fft_cufft.h"
#include "cufft.h"

__global__ void normalize_kernel(float* data, int n, float mult) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] *= mult;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    size_t bytes = input.size() * sizeof(float);
    int total = static_cast<int>(input.size());

    static cufftComplex* gpu_data = nullptr;
    static size_t device_capacity = 0;

    if (device_capacity != bytes) {
        if (device_capacity > 0) cudaFree(gpu_data);
        cudaMalloc(&gpu_data, bytes);
        device_capacity = bytes;
    }

    static cufftHandle plan = 0;
    static int cached_n = 0;
    static int cached_batch = 0;

    if (cached_n != n || cached_batch != batch) {
        if (plan) cufftDestroy(plan);
        cufftPlan1d(&plan, n, CUFFT_C2C, batch);
        cached_n = n;
        cached_batch = batch;
    }

    cudaMemcpy(gpu_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
    cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE);

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    normalize_kernel<<<num_blocks, block_size>>>(reinterpret_cast<float*>(gpu_data), total, 1.0f / n);

    std::vector<float> result(total);
    cudaMemcpy(result.data(), gpu_data, bytes, cudaMemcpyDeviceToHost);

    return result;
}