#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};

    static float* dev_mem_a = nullptr;
    static float* dev_mem_b = nullptr;
    static float* dev_mem_c = nullptr;
    static int current_allocated_n = 0;
    static cublasHandle_t blas_engine_handle = nullptr;
    static cudaStream_t execution_stream = nullptr;

    size_t total_elements = static_cast<size_t>(n) * n;
    size_t total_bytes = total_elements * sizeof(float);

    if (n > current_allocated_n) {
        if (dev_mem_a) {
            cudaFree(dev_mem_a);
            cudaFree(dev_mem_b);
            cudaFree(dev_mem_c);
        }
        cudaMalloc(&dev_mem_a, total_bytes);
        cudaMalloc(&dev_mem_b, total_bytes);
        cudaMalloc(&dev_mem_c, total_bytes);
        current_allocated_n = n;
    }

    if (!execution_stream) {
        cudaStreamCreate(&execution_stream);
        cublasCreate(&blas_engine_handle);
        cublasSetStream(blas_engine_handle, execution_stream);
    }

    cudaMemcpyAsync(dev_mem_a, a.data(), total_bytes, cudaMemcpyHostToDevice, execution_stream);
    cudaMemcpyAsync(dev_mem_b, b.data(), total_bytes, cudaMemcpyHostToDevice, execution_stream);

    const float scaling_alpha = 1.0f;
    const float scaling_beta = 0.0f;

    cublasSgemm(blas_engine_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &scaling_alpha,
                dev_mem_b, n,
                dev_mem_a, n,
                &scaling_beta,
                dev_mem_c, n);

    std::vector<float> host_result(total_elements);
    cudaMemcpyAsync(host_result.data(), dev_mem_c, total_bytes, cudaMemcpyDeviceToHost, execution_stream);

    cudaStreamSynchronize(execution_stream);

    return host_result;
}