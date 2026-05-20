#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void BlockGemm_cu(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    __shared__ float A_shared_1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared_1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float A_shared_2[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared_2[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    const int BLOCK_CNT = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int block_idx = 0; block_idx < BLOCK_CNT; block_idx += 2) {
        
        {
            int block_start = block_idx * BLOCK_SIZE;

            float a_val = (row < n && block_start + tx < n) ? A[row * n + block_start + tx] : 0.0f;
            float b_val = (block_start + ty < n && col < n) ? B[(block_start + ty) * n + col] : 0.0f;

            A_shared_1[ty][tx] = a_val;
            B_shared_1[ty][tx] = b_val;
            
            __syncthreads();
            
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += A_shared_1[ty][k] * B_shared_1[k][tx];
            }
        }

        if (block_idx + 1 < BLOCK_CNT) {
            int block_start = (block_idx + 1) * BLOCK_SIZE;

            float a_val = (row < n && block_start + tx < n) ? A[row * n + block_start + tx] : 0.0f;
            float b_val = (block_start + ty < n && col < n) ? B[(block_start + ty) * n + col] : 0.0f;

            A_shared_2[ty][tx] = a_val;
            B_shared_2[ty][tx] = b_val;
            
            __syncthreads();
            
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += A_shared_2[ty][k] * B_shared_2[k][tx];
            }
        }
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = n * n * sizeof(float);
    static float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    static cudaStream_t stream = nullptr;
    static int mem_sz = 0;

    if (mem_sz != size) {
        if (stream == nullptr) {
            cudaStreamCreate(&stream);
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        mem_sz = size;
    }
    
    cudaMemcpyAsync(d_A, a.data(), size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b.data(), size, cudaMemcpyHostToDevice, stream);
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    BlockGemm_cu<<<grid, threads, 0, stream>>>(d_A, d_B, d_C, n);
    
    std::vector<float> c(n * n, 0.0f);
    cudaMemcpyAsync(c.data(), d_C, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return c;
}