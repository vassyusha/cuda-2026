#define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <chrono>

const char* source = 
"__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {"
"    int i = get_global_id(0);"
"    if (i < n) {"
"        float x = input[i];"
"        float x3 = x * x * x;"
"        float arg = 0.7978845608f * (x + 0.044715f * x3);"
"        output[i] = x / (1.0f + exp(-2.0f * arg));"
"    }"
"}";

static cl_device_id g_device;
static cl_context g_context;
static cl_command_queue g_queue;
static cl_program g_program;
static cl_kernel g_kernel;
static cl_mem g_gpu_input = nullptr;
static cl_mem g_gpu_output = nullptr;
static size_t g_current_size = 0;
static bool g_is_init = false;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (!g_is_init) {
        cl_platform_id pid = platforms[platform];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &g_device, nullptr);

        g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, nullptr);

        cl_queue_properties props[] = {0};
        g_queue = clCreateCommandQueueWithProperties(g_context, g_device, props, nullptr);

        g_program = clCreateProgramWithSource(g_context, 1, &source, nullptr, nullptr);
        clBuildProgram(g_program, 1, &g_device, nullptr, nullptr, nullptr);
        g_kernel = clCreateKernel(g_program, "gelu_kernel", nullptr);

        g_is_init = true;
    }

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    // Переиспользуем буферы если размер не изменился
    if (g_gpu_input != nullptr && g_current_size != n) {
        clReleaseMemObject(g_gpu_input);
        clReleaseMemObject(g_gpu_output);
        g_gpu_input = nullptr;
        g_gpu_output = nullptr;
    }

    if (g_gpu_input == nullptr) {
        g_gpu_input = clCreateBuffer(g_context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        g_gpu_output = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        g_current_size = n;
    }

    int n_for_kernel = static_cast<int>(n);
    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_gpu_input);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_gpu_output);
    clSetKernelArg(g_kernel, 2, sizeof(int), &n_for_kernel);

    // Асинхронное копирование на GPU
    cl_event write_event;
    clEnqueueWriteBuffer(g_queue, g_gpu_input, CL_FALSE, 0, bytes, 
                         input.data(), 0, nullptr, &write_event);

    // Асинхронный запуск ядра
    cl_event kernel_event;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &n, nullptr, 
                           1, &write_event, &kernel_event);

    // Асинхронное чтение результата
    std::vector<float> result(n);
    cl_event read_event;
    clEnqueueReadBuffer(g_queue, g_gpu_output, CL_FALSE, 0, bytes, 
                        result.data(), 1, &kernel_event, &read_event);

    // Ждём только чтение
    clWaitForEvents(1, &read_event);

    // Освобождаем события
    clReleaseEvent(write_event);
    clReleaseEvent(kernel_event);
    clReleaseEvent(read_event);

    return result;
}