#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

static constexpr float sqrt_2_div_pi     = 0.7978845608028653558f;
static constexpr float two_sqrt_2_div_pi = 1.5957691216057307116f;
static constexpr float coeff_cubic       = 0.044715f;

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float x   = __ldg(input + i);
        float x3  = x * x * x;
        float arg = -two_sqrt_2_div_pi * (x + coeff_cubic * x3);
        output[i] = x / (1.0f + __expf(arg));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input,  n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    float *h_pinned = nullptr;
    cudaMallocHost(&h_pinned, n * sizeof(float));

    const int block_size = 256;
    const int grid_size  = std::min(128 * 56, (n + block_size - 1) / block_size);

    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gelu_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
    cudaMemcpy(h_pinned, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> result(h_pinned, h_pinned + n);

    cudaFreeHost(h_pinned);
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}