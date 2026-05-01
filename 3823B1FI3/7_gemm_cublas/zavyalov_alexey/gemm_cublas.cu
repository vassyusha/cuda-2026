#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> res(n * n);
    static float *a_gpu = nullptr, *b_gpu = nullptr, *result_gpu=nullptr;

    static int device_capacity = 0;

    static cublasHandle_t handle = nullptr;
    
    if (handle == nullptr) {
        cublasCreate(&handle);
    }

    if (device_capacity != n * n) {
        if (device_capacity > 0) {
            cudaFree(a_gpu);
            cudaFree(b_gpu);
            cudaFree(result_gpu);
        }
        cudaMalloc(&a_gpu,      n * n * sizeof(float));
        cudaMalloc(&b_gpu,      n * n * sizeof(float));
        cudaMalloc(&result_gpu, n * n * sizeof(float));
        device_capacity = n * n;
    }

    cudaMemcpy(a_gpu, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        b_gpu, n,
        a_gpu, n,
        &beta,
        result_gpu, n
    );
    
    cudaMemcpy(res.data(), result_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    return res;
}