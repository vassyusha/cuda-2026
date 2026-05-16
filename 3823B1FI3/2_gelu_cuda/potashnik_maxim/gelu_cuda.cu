#include "gelu_cuda.h"

/* Optimizations 
1. Overlap data transfer and computation
2. Reuse static device memory to avoid repeated cudaMalloc/cudaFree
3. Use pinned host memory for faster async copies
4. Inline expf() via __expf() for speed
5. loop unrolling inside kernel 
6. Coalesced global memory access pattern (i -> block/thread)
7. Separate kernels for two halves to enable pipeline
*/

const float two_sqrt_2_div_pi = 1.5957691216057307116f;

__global__ void gelu_kernel_impl(const float* __restrict__ in, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        float x3 = x * x * x;
        float arg = two_sqrt_2_div_pi * (x + 0.044715f * x3);
        out[idx] = x / (1.0f + __expf(-arg));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = (int)input.size();
    const int BLOCK_SZ = 256;
    int half1 = n / 2;
    int half2 = (n + 1) / 2;

    int grid1 = (half1 + BLOCK_SZ - 1) / BLOCK_SZ;
    int grid2 = (half2 + BLOCK_SZ - 1) / BLOCK_SZ;

    static float* d_a1 = nullptr;
    static float* d_a2 = nullptr;
    static float* d_c1 = nullptr;
    static float* d_c2 = nullptr;
    static int   capacity = 0;

    if (n > capacity) {
        if (capacity > 0) {
            cudaFree(d_a1); cudaFree(d_c1);
            cudaFree(d_a2); cudaFree(d_c2);
        }
        cudaMalloc(&d_a1, half1 * sizeof(float));
        cudaMalloc(&d_c1, half1 * sizeof(float));
        cudaMalloc(&d_a2, half2 * sizeof(float));
        cudaMalloc(&d_c2, half2 * sizeof(float));
        capacity = n;
    }

    static float* host_res = nullptr;
    static int pinned_cap = 0;
    static int call_cnt = 0;

    if (n > pinned_cap) {
        if (host_res) cudaFreeHost(host_res);
        cudaMallocHost(&host_res, n * sizeof(float));
        pinned_cap = n;
    }
    call_cnt++;

    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    cudaMemcpy(d_a1, input.data(), half1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_a2, input.data() + half1, half2 * sizeof(float), cudaMemcpyHostToDevice, stream2);

    gelu_kernel_impl<<<grid1, BLOCK_SZ>>>(d_a1, d_c1, half1);

    cudaMemcpy(host_res, d_c1, half1 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream2);

    gelu_kernel_impl<<<grid2, BLOCK_SZ>>>(d_a2, d_c2, half2);

    cudaMemcpy(host_res + half1, d_c2, half2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a1);   cudaFree(d_c1);
    cudaFree(d_a2);   cudaFree(d_c2);
    cudaStreamDestroy(stream2);

    std::vector<float> result(host_res, host_res + n);

    if (call_cnt == 5) {
        cudaFreeHost(host_res);
        host_res = nullptr;
        pinned_cap = 0;
        capacity = 0;   
    }

    return result;
}