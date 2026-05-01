#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];

    const float c1 = 0.044715f;
    const float c2 = 0.7978845608f;  // sqrt(2/pi)

    float x2 = x * x;
    float inner = c2 * (x + c1 * x * x2);
    float z = 2.0f * inner;

    float neg_z = -fminf(z, 20.0f);
    float sigmoid = __fdividef(1.0f, 1.0f + __expf(neg_z));

    output[idx] = x * sigmoid;
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t N = input.size();
    std::vector<float> output(N);
    if (N == 0) return output;

    cudaHostRegister((void*)input.data(), N * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(output.data(), N * sizeof(float), cudaHostRegisterDefault);

    float *d_input = nullptr, *d_output = nullptr;
    const int num_streams = 2;
    cudaStream_t streams[num_streams];

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    for (int i = 0; i < num_streams; ++i)
        cudaStreamCreate(&streams[i]);   

    size_t chunk_size = (N + num_streams - 1) / num_streams;
    for (int i = 0; i < num_streams; ++i) {
        size_t offset = i * chunk_size;
        size_t size = std::min(chunk_size, N - offset);
        if (size == 0) continue;

        cudaMemcpyAsync(d_input + offset, input.data() + offset,
                        size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[i]);  

        int block = 256;
        int grid = (size + block - 1) / block;
        gelu_kernel<<<grid, block, 0, streams[i]>>>(d_input + offset,
                                                    d_output + offset, size);

        cudaMemcpyAsync(output.data() + offset, d_output + offset,
                        size * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[i]);  
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < num_streams; ++i) cudaStreamDestroy(streams[i]);  
    cudaFree(d_input);
    cudaFree(d_output);
    cudaHostUnregister((void*)input.data());
    cudaHostUnregister(output.data());

    return output;
}