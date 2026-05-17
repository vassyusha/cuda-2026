#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory>

#define block 128

__global__ void normalizationCUDAkernel(cufftComplex* __restrict__ data, float coeff, int size) {
    int i = blockIdx.x * block + threadIdx.x;
    if(i < size){
        data[i].x *= coeff;
        data[i].y *= coeff;
    }
}

struct Deleter {
    void operator()(cufftComplex * ptr) const {
        if (ptr) cudaFree(ptr);
    }
};
using my_pointer = std::unique_ptr<cufftComplex[], Deleter>;

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n_total = input.size()/2;
    int n = n_total/batch;
    float coeff = 1.0f/n;
    std::vector<float> result(input.size());
    size_t bytes = static_cast<size_t>(n_total * sizeof(cufftComplex));
    
    cufftComplex * data_tmp;
    cudaMalloc(&data_tmp,bytes);
    my_pointer d_data(data_tmp);
    cudaMemcpy(data_tmp, input.data(),bytes, cudaMemcpyHostToDevice);
    
    cufftHandle handle;
    cufftPlan1d(&handle,n,CUFFT_C2C,batch);
    cufftExecC2C(handle,data_tmp,data_tmp,CUFFT_FORWARD);
    cufftExecC2C(handle,data_tmp,data_tmp,CUFFT_INVERSE);

    int threadsForBlock=block; 
    int numBlocks = (n_total+block-1)/block;
    normalizationCUDAkernel<<<numBlocks, threadsForBlock>>>(data_tmp, coeff,n_total);
    cudaMemcpy(result.data(), data_tmp, bytes, cudaMemcpyDeviceToHost);
    cufftDestroy(handle);
    return result;
}