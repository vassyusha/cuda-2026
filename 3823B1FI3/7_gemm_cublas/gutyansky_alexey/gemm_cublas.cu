#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = n * n * sizeof(float);
    static float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    static cudaStream_t stream = nullptr;
    static int mem_sz = 0;
    static cublasHandle_t handle = nullptr;
    
    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }

    if (mem_sz < size) {

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

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_B, n,
        d_A, n,
        &beta,
        d_C, n);
    
    std::vector<float> c(n * n, 0.0f);
    cudaMemcpyAsync(c.data(), d_C, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return c;
}