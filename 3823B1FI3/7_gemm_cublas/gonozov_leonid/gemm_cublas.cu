#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {

    size_t bytes = n * n * sizeof(float);
    float *input_a_cuda, *input_b_cuda, *output_c_cuda;

    cudaMalloc(&input_a_cuda, bytes);
    cudaMalloc(&input_b_cuda, bytes);
    cudaMalloc(&output_c_cuda, bytes);

    cudaMemcpy(input_a_cuda, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(input_b_cuda, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, input_b_cuda, n, input_a_cuda, n, &beta, output_c_cuda, n);

    std::vector<float> output(n * n);
    cudaMemcpy(output.data(), output_c_cuda, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(input_a_cuda);
    cudaFree(input_b_cuda);
    cudaFree(output_c_cuda);

    return output;
}
