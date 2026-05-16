#include "block_gemm_cuda.h"
#include <algorithm>

#define BLOCK_SIZE 16

__global__ void BlockGemmKernel(const float* __restrict__ A, 
                                 const float* __restrict__ B, 
                                 float* __restrict__ C, 
                                 int n) {

    __shared__ float shareinput_a_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shareinput_b_cuda[BLOCK_SIZE][BLOCK_SIZE];
    

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    

    float sum = 0.0f;
    int num_blocks = n / BLOCK_SIZE;
    
    for (int block_k = 0; block_k < num_blocks; ++block_k) {
        if (row < n && (block_k * BLOCK_SIZE + threadIdx.x) < n) {
            shareinput_a_cuda[threadIdx.y][threadIdx.x] = A[row * n + block_k * BLOCK_SIZE + threadIdx.x];
        } else {
            shareinput_a_cuda[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((block_k * BLOCK_SIZE + threadIdx.y) < n && col < n) {
            shareinput_b_cuda[threadIdx.y][threadIdx.x] = B[(block_k * BLOCK_SIZE + threadIdx.y) * n + col];
        } else {
            shareinput_b_cuda[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shareinput_a_cuda[threadIdx.y][k] * shareinput_b_cuda[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = static_cast<size_t>(n) * n;
    size_t bytes = size * sizeof(float);

    static float *input_a_cuda = nullptr, *input_b_cuda = nullptr, *input_c_cuda = nullptr;
    static size_t prev_size = 0;
    static cudaStream_t stream = nullptr;

    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    if (prev_size != size) {
        if (input_a_cuda != nullptr) cudaFree(input_a_cuda);
        if (input_b_cuda != nullptr) cudaFree(input_b_cuda);
        if (input_c_cuda != nullptr) cudaFree(input_c_cuda);
        
        cudaMalloc(&input_a_cuda, bytes);
        cudaMalloc(&input_b_cuda, bytes);
        cudaMalloc(&input_c_cuda, bytes);
        
        prev_size = size;
    }

    cudaMemcpyAsync(input_a_cuda, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(input_b_cuda, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(n / BLOCK_SIZE, n / BLOCK_SIZE);

    BlockGemmKernel<<<grid, threads, 0, stream>>>(input_a_cuda, input_b_cuda, input_c_cuda, n);

    std::vector<float> output_c(size, 0.0f);

    cudaMemcpyAsync(output_c.data(), input_c_cuda, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    return output_c;
}
