#include "naive_gemm_cuda.h"

__global__ void naive_gemm_kernel(float* __restrict__ input_a, float* __restrict__ input_b, float* __restrict__ output_c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += input_a[idy * n + k] * input_b[k * n + idx];
        }
        output_c[idy * n + idx] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {

    std::vector<float> output(n * n);
    
    float *input_cuda_a;
    float *input_cuda_b;
    float *outut_cuda_c;

    cudaMalloc(&input_cuda_a, n * n * sizeof(float));
    cudaMemcpy(input_cuda_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&input_cuda_b, n * n * sizeof(float));
    cudaMemcpy(input_cuda_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&outut_cuda_c, n * n * sizeof(float));
    
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

    naive_gemm_kernel<<<blocks, threads>>>(input_cuda_a, input_cuda_b, outut_cuda_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), outut_cuda_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_cuda_a);
    cudaFree(input_cuda_b);
    cudaFree(outut_cuda_c);    
    return output;
}
