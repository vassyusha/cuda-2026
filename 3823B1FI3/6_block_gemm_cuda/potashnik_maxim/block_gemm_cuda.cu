#include "block_gemm_cuda.h"
#include <vector>

/* Optimizations
1. Tiled multiplication with shared memory
2. 2D thread blocks (tile_dim x tile_dim)
3. Loop unrolling inside tile
4. Static device memory reuse
5. cudaMemset on device
6. __restrict__ pointers
*/

const int tile_dim = 32;

__global__ void gemm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float A_tile[tile_dim][tile_dim];
    __shared__ float B_tile[tile_dim][tile_dim];

    int col = blockIdx.x * tile_dim + threadIdx.x;
    int row = blockIdx.y * tile_dim + threadIdx.y;

    float sum = 0.0f;

    int num_tiles = (N + tile_dim - 1) / tile_dim;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * tile_dim + threadIdx.x;
        A_tile[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        int b_row = t * tile_dim + threadIdx.y;
        B_tile[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tile_dim; ++k) sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        
            __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A, const std::vector<float>& B, int N) {
    int total_size = N * N;

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static int capacity = 0;

    if (total_size > capacity) {
        if (capacity > 0) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
        cudaMalloc(&d_A, total_size * sizeof(float));
        cudaMalloc(&d_B, total_size * sizeof(float));
        cudaMalloc(&d_C, total_size * sizeof(float));
        capacity = total_size;
    }

    cudaMemcpy(d_A, A.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, total_size * sizeof(float));

    dim3 blockDim(tile_dim, tile_dim);
    dim3 gridDim((N + tile_dim - 1) / tile_dim,
                 (N + tile_dim - 1) / tile_dim);

    gemm_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    std::vector<float> C(total_size);
    cudaMemcpy(C.data(), d_C, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    return C;
}