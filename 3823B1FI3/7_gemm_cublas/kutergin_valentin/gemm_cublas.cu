#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// основная функция (выполняется на CPU)
std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t size = (size_t)n * n;

    // статические указатели, чтобы не делать аллокацию и деаллокацию памяти на каждом вызове 
    static float *d_a = nullptr;
    static float *d_b = nullptr;
    static float *d_c = nullptr;
    static int allocated_size = 0;

    // поток для асинхронности
    static cudaStream_t stream = nullptr;

    // "Контекст" библиотеки cuBLAS
    static cublasHandle_t handle = nullptr;

    // выделение памяти на GPU
    if (allocated_size < n) {
        if (d_a)
            cudaFree(d_a);
        if (d_b)
            cudaFree(d_b);
        if (d_c)
            cudaFree(d_c);
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));\
        if (!stream) {
            cudaStreamCreate(&stream); // создание потока для асинхронных операций
        }
        if (!handle) {
            cublasCreate(&handle); // создание контекста cuBLAS
        }
        cublasSetStream(handle, stream); // привязка контекста к потоку
        allocated_size = n;
    }

     // асинхронное копирование входных матриц с CPU на GPU в потоке stream
    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n); // выполнение умножения матриц на GPU

    std::vector<float> c(size); // пока GPU выполняет вычисления, выделяем память для результата на CPU

    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, stream); // асинхронное копирование результата с GPU на CPU в потоке stream

    cudaStreamSynchronize(stream); // // синхронизация всех операций в потоке stream

    return c;
}