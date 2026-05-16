#include "gelu_cuda.h"

/* Optimizations
1. Static device memory - cudaMalloc/cudaFree only when size grows
2. Pinned host memory 
3. Async copy of second half (cudaMemcpyAsync) overlapped with kernel execution on first half
4. Static stream - created once, not on every call
5. __expf() instead of expf()
6. __restrict__ pointers
7. Coalesced global memory access
*/

const float two_mult_sqrt_2_div_pi = 1.5957691216057307116f;

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float arg = two_mult_sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        float e = __expf(arg);
        result[i] = x * e / (e + 1.0f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    const int block_size = 256;

    int half1 = n / 2;
    int half2 = (n + 1) / 2;
    int grid1 = (half1 + block_size - 1) / block_size;
    int grid2 = (half2 + block_size - 1) / block_size;

    static float* d_in1  = nullptr;
    static float* d_in2  = nullptr;
    static float* d_out1 = nullptr;
    static float* d_out2 = nullptr;
    static int device_capacity = 0;

    if (n > device_capacity) {
        if (device_capacity > 0) {
            cudaFree(d_in1);  cudaFree(d_out1);
            cudaFree(d_in2);  cudaFree(d_out2);
        }
        cudaMalloc(&d_in1,  half1 * sizeof(float));
        cudaMalloc(&d_out1, half1 * sizeof(float));
        cudaMalloc(&d_in2,  half2 * sizeof(float));
        cudaMalloc(&d_out2, half2 * sizeof(float));
        device_capacity = n;
    }

    static float* h_out = nullptr;
    static int pinned_capacity = 0;

    if (n > pinned_capacity) {
        if (h_out) cudaFreeHost(h_out);
        cudaMallocHost(&h_out, n * sizeof(float));
        pinned_capacity = n;
    }

    static cudaStream_t stream = nullptr;
    if (!stream) cudaStreamCreate(&stream);

    cudaMemcpy(d_in1, input.data(), half1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_in2, input.data() + half1, half2 * sizeof(float), cudaMemcpyHostToDevice, stream);

    gelu_kernel<<<grid1, block_size>>>(d_in1, d_out1, half1);

    cudaMemcpy(h_out, d_out1, half1 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);
    gelu_kernel<<<grid2, block_size>>>(d_in2, d_out2, half2);

    cudaMemcpy(h_out + half1, d_out2, half2 * sizeof(float), cudaMemcpyDeviceToHost);

    return std::vector<float>(h_out, h_out + n);
}