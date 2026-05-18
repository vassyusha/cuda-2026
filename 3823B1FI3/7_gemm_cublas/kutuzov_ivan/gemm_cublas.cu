#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    float* gpu_A;
    float* gpu_B;
    float* gpu_C;
    int bytes = n * n * sizeof(float);
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaMalloc(&gpu_A, bytes);
    cudaMalloc(&gpu_B, bytes);
    cudaMalloc(&gpu_C, bytes);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, gpu_A, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, gpu_B, n);

    cublasHandle_t handle;
    cublasCreate(&handle);


    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, gpu_B, n,
                gpu_A, n, &beta, gpu_C, n);

        
    std::vector<float> C(n * n);

    cublasGetMatrix(n, n, sizeof(float), gpu_C, n, C.data(), n);

    cublasDestroy(handle);
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    return C;
}