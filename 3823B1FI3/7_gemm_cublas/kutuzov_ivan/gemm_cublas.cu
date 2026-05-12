#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    const int total = n * n;
    if (total == 0) return {};

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *h_pinned_inA = nullptr, *h_pinned_inB = nullptr, *h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_inA, total * sizeof(float));
    cudaMallocHost(&h_pinned_inB, total * sizeof(float));
    cudaMallocHost(&h_pinned_out, total * sizeof(float));

    std::copy(a.begin(), a.end(), h_pinned_inA);
    std::copy(b.begin(), b.end(), h_pinned_inB);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    cudaMemcpyAsync(d_A, h_pinned_inA, total * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_pinned_inB, total * sizeof(float), cudaMemcpyHostToDevice, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);

    cudaMemcpyAsync(h_pinned_out, d_C, total * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::vector<float> result(h_pinned_out, h_pinned_out + total);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    cudaFreeHost(h_pinned_inA);
    cudaFreeHost(h_pinned_inB);
    cudaFreeHost(h_pinned_out);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}