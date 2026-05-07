#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

// ядро для нормализации данных на каждом ядре GPU
__global__ void NormalizeKernel(cufftComplex* data, int total_elements, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

// основная функция (выполняется на CPU)
std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int total_complex_elements = static_cast<int>(input.size() / 2); // общее количество комплексных чисел
    int n = total_complex_elements / batch; // длина одного сигнала n

    // статическая указатели, чтобы не делать аллокацию и деаллокацию памяти на каждом вызове
    static cufftComplex *d_data = nullptr; 
    static int last_size = 0;
    static cufftHandle plan = 0;

    // поток для асинхронности
    static cudaStream_t stream = nullptr; 

    if (last_size < total_complex_elements) {
        if (d_data)
            cudaFree(d_data);
        if (plan)
            cufftDestroy(plan);
        
        cudaMalloc(&d_data, total_complex_elements * sizeof(cufftComplex));

        cufftPlan1d(&plan, n, CUFFT_C2C, batch);

        if (!stream) 
            cudaStreamCreate(&stream);

        cufftSetStream(plan, stream);

        last_size = total_complex_elements;
    }

    // асинхронное копирование данных с CPU на GPU в потоке stream
    cudaMemcpyAsync(d_data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD); // прямое преобразование
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE); // обратное преобразование

    int threads = 256;
    int blocks = (total_complex_elements + threads - 1) / threads; 

    NormalizeKernel<<<blocks, threads, 0, stream>>>(d_data, total_complex_elements, 1.0f / n); // запуск ядра на GPU с конфигурацией запуска асинхронно в потоке stream

    std::vector<float> output(input.size()); // пока GPU выполняет вычисления, выделяем память для результата на CPU

    // асинхронное копирование результата с GPU на CPU в потоке stream
    cudaMemcpyAsync(output.data(), d_data, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream); // синхронизация всех операций в потоке stream

    return output;
}