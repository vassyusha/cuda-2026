#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void kernel_transform(cufftComplex* data, int n, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int count_complex_kernel = n * batch;
    float coef = 1.0f / (float)n;
    
    if (idx < count_complex_kernel) {
        data[idx].x *= coef;
        data[idx].y *= coef;
    }
}

static cufftComplex* d_data = nullptr;
static size_t allocated = 0;
static cufftHandle plan_fwd = 0, plan_inv = 0;
static cudaStream_t stream = nullptr;

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int count_complex = n * batch;
    const size_t bytes = count_complex * sizeof(cufftComplex);

    if (!stream) {
        cudaStreamCreate(&stream);
    }

    if (plan_fwd == 0) {
        cufftPlan1d(&plan_fwd, n, CUFFT_C2C, batch);
        cufftPlan1d(&plan_inv, n, CUFFT_C2C, batch);
        cufftSetStream(plan_fwd, stream);
        cufftSetStream(plan_inv, stream);
    }

    if (allocated < bytes) {
        if (d_data) cudaFree(d_data);
        cudaMalloc(&d_data, bytes);
        allocated = bytes;
    }

    cudaMemcpyAsync(d_data, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    cufftExecC2C(plan_fwd, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan_inv, d_data, d_data, CUFFT_INVERSE);

    int block = 256;
    int grid = (count_complex + block - 1) / block;
    kernel_transform<<<grid, block, 0, stream>>>(d_data, n, batch);

    std::vector<float> output(input.size());
    cudaMemcpyAsync(output.data(), d_data, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}
