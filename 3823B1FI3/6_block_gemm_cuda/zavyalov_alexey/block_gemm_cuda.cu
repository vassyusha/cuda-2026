#include "block_gemm_cuda.h"

#define TILE_SIZE 32

__global__ void gemm_kernel(const float* a, const float* b, float* result, int n) {

    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float local_res = 0.0f;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {

        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        a_shared[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;

        b_shared[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            local_res += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        result[row * n + col] = local_res;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> res(n*n);
    
    static float *a_gpu = nullptr, *b_gpu = nullptr, *result_gpu=nullptr;

    static int call_number = 0;
    static int device_capacity = 0;
    
    if (device_capacity != n * n) {
        if (device_capacity > 0) {
            cudaFree(a_gpu);
            cudaFree(b_gpu);
            cudaFree(result_gpu);
        }
        cudaMalloc(&b_gpu, n * n * sizeof(float));
        cudaMalloc(&a_gpu, n * n * sizeof(float));
        cudaMalloc(&result_gpu, n * n  * sizeof(float));
        device_capacity = n * n;
    }

    cudaMemcpy(b_gpu, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_gpu, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(result_gpu, 0, n * n  * sizeof(float));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<grid, block>>>(a_gpu, b_gpu, result_gpu, n);


    cudaMemcpy(res.data(), result_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);


    ++call_number;
    if (call_number == 5) {
        cudaFree(b_gpu);
    }
    return res;
}