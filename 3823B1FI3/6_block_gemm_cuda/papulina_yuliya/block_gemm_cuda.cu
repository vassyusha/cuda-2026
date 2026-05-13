#include "block_gemm_cuda.h"

#define block 32

__global__ void BlockGemmCUDAkernel(const float* __restrict__ a, const float* __restrict__ b,float* __restrict__ res, int n) {
    __shared__ float a_block[block][block];
    __shared__ float b_block[block][block];
    int i = blockIdx.x * block + threadIdx.x;
    int j = blockIdx.y * block + threadIdx.y;
    float tmp = 0.0f;
    for (int step = 0; step < n / block; step++) {
        a_block[threadIdx.y][threadIdx.x] = a[j * n + (step * block + threadIdx.x)];
        b_block[threadIdx.y][threadIdx.x] = b[(step * block + threadIdx.y) * n + i];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < block; k++) {
            tmp += a_block[threadIdx.y][k] * b_block[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (i < n && j < n) {
        res[j * n + i] = tmp;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> result(static_cast<size_t>(n) * n);
    float *a_in = nullptr, *b_in = nullptr, *res = nullptr;
    size_t bytes = static_cast<size_t>(n) *n*sizeof(float);

    cudaMalloc(&a_in, bytes);
    cudaMalloc(&b_in, bytes);
    cudaMalloc(&res, bytes);
    cudaMemcpy(a_in, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_in, b.data(), bytes, cudaMemcpyHostToDevice);
    dim3 threadsForBlock(block, block); 
    dim3 numBlocks(n/block, n/block);
    BlockGemmCUDAkernel<<<numBlocks, threadsForBlock>>>(a_in, b_in, res, n);
    cudaMemcpy(result.data(), res, bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_in);
    cudaFree(b_in);
    cudaFree(res);

    return result;
}