#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void NaiveGemmCUDAkernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ res, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*4;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll 8
        for(int k=0; k<n; k++){
            float a_elem = a[j*n +k];
            float b_elem = reinterpret_cast<const float4*>(b + k * n + i)[0];
            tmp.x += a_elem * b_elem.x;
            tmp.y += a_elem * b_elem.y;
            tmp.z += a_elem * b_elem.z;
            tmp.w += a_elem * b_elem.w;
        }
        reinterpret_cast<float4*>(res + j * n + i)[0] = tmp;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> result(n*n);
    float * a_in = nullptr, *b_in=nullptr, *res = nullptr;
    size_t bytes = n * sizeof(float) * n;
    const int block_size = 256;

    cudaMalloc(&a_in, bytes);
    cudaMalloc(&b_in, bytes);
    cudaMalloc(&res, bytes);

    cudaMemcpy(a_in, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_in, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsForBlock(16, 16); 
    dim3 numBlocks((n/4 + threadsForBlock.x - 1)/threadsForBlock.x,(n + threadsForBlock.y - 1)/threadsForBlock.y);

    NaiveGemmCUDAkernel<<<numBlocks, threadsForBlock>>>(a_in, b_in, res, n);

    cudaMemcpy(result.data(), res, bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_in);
    cudaFree(b_in);
    cudaFree(res);

    return result;
}
