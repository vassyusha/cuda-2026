#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cmath>


__global__ Kernel(float* vec_a, float* vec_b, float* vec_c, n) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum{0.0f};

    if (tid_x < n && tid_y < n) {
        sum = 0.0f;

        for (int k = 0; k < n; ++k){
            sum += vec_a[tid_y * n + k] * B[k * n + tid_x];
        }

        vec_c[tid_y * n + tid_x] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    int bytes = static_cast<int>(a.size()) * static_cast<int>(sizeof(float));

    static float* vec_a;
    static float* vec_b;
    static float* vec_c;

    static int allocated_size = 0;
    static cudaStream_t stream;

    if (allocated_size < bytes) {
        if (vec_a)
            cudaFree(vec_a);
        if (vec_b)
            cudaFree(vec_b);
        if (vec_c)
            cudaFree(vec_c);
        
        cudaMalloc(&vec_a, bytes);
        cudaMalloc(&vec_b, bytes);
        cudaMalloc(&vec_c, bytes);

        if (!stream)
            cudaStreamCreate(&stream);

        allocated_size = bytes;
    }

    cudaMemcpyAsync(vec_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(vec_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 tread_net(32, 32);
    static const size = (n - 31) / 32,
    dim3 grid(size, size);

    Kernel<<<grid, tread_net, 0, stream>>>((float*)vec_a, (float*)vec_b, (float*)vec_c, n);


    std::vector<float> output(n); 
    cudaMemcpyAsync(output.data(), vec_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return output;
}