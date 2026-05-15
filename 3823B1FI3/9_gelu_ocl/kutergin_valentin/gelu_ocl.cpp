#include "gelu_ocl.h"
#include <CL/cl.h>

// код ядра в виде строки для вычисления GELU активации на каждом ядре GPU
const char* kernelSource = R"(
__kernel void GeluKernel(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        const float double_sqrt_2_over_pi = 1.59576912f;
        const float coeff = 0.044715f;
        float arg = double_sqrt_2_over_pi * (x + coeff * x * x * x);
        output[i] = x / (1.0f + exp(-arg)); // формула, аппроксимирующая GELU
    }
}
)";

// основная функция (выполняется на CPU)
std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int n = static_cast<int>(input.size());

    // статические переменные для хранения ресурсов OpenCL, чтобы инициализировать их только один раз
    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_kernel kernel = nullptr;
    static cl_program program = nullptr;
    static cl_mem d_in = nullptr;
    static cl_mem d_out = nullptr;
    static int allocated_size = 0;

    // Инициализация OpenCL ресурсов при первом вызове
    if (!context) {
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms); // получение количества платформ
        std::vector<cl_platform_id> selected_platforms(num_platforms);
        clGetPlatformIDs(num_platforms, selected_platforms.data(), nullptr); // получение идентификаторов платформ
        
        cl_platform_id platform_id = selected_platforms[platform]; 

        cl_device_id device;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr); // получение идентификатора GPU устройства 

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr); // создание контекста
        queue = clCreateCommandQueue(context, device, 0, nullptr); // создание очереди команд;

        // компиляция кода ядра из строки
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "GeluKernel", nullptr);
    }

    // выделение буферов на GPU
    if (allocated_size < n) {
        if (d_in)
            clReleaseMemObject(d_in);
        if (d_out)
            clReleaseMemObject(d_out);
        d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
        d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);
        allocated_size = n;
    }

    clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, n * sizeof(float), input.data(), 0, nullptr, nullptr); // копируем данные с CPU на GPU асинхронно

    // Установка аргументов ядра
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    // Запуск вычислений на GPU
    size_t global_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr); // постановка запуска ядра в очередь команд

    std::vector<float> output(n); // пока GPU выполняет вычисления, выделяем память для результата на CPU

    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr); // блокирующее чтение результата с GPU на CPU

    return output;
}