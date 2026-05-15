#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const int size = n * n;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,   
                d_a, n,  
                &beta,
                d_c, n);

    std::vector<float> result(size);

    cudaMemcpyAsync(result.data(), d_c, size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}