#define CL_TARGET_OPENCL_VERSION 300

#include "gelu_ocl.h"
#include <CL/cl.h>

static const char* gelu_src = R"(
__kernel void gelu_ocl(__global const float* in, __global float* out, int n) {

    const float _2SQRT2PI = 2.0f * sqrt(2.0f / 3.141592653589793238462643f);
    const float C1 = 0.044715f;
    
    int i = get_global_id(0);
    if (i < n) {
        float x = in[i];

        float x3 = x * x * x;
        float arg = _2SQRT2PI * (x + C1 * x3);
        float ex = exp(-arg);

        out[i] = x / (1.0f + ex);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int N = input.size();
    const size_t bytes_sz = N * sizeof(float);

    static int initialized_platform = -1;

    static cl_context context;
    static cl_command_queue queue;
    static cl_program program;
    static cl_kernel kernel;

    static size_t mem_sz = 0;

    if (initialized_platform != platform) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_device_id device;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);
    
        program = clCreateProgramWithSource(context, 1, &gelu_src, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_ocl", nullptr);

        initialized_platform = platform;
        mem_sz = 0;
    }

    static cl_mem d_in = nullptr;
    static cl_mem d_out = nullptr;

    if (mem_sz != bytes_sz) {
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);

        d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_sz, nullptr, nullptr);
        d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_sz, nullptr, nullptr);

        mem_sz = bytes_sz;
    }

    clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, bytes_sz, input.data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &N);

    const size_t block_size = 256;
    const size_t full_size = ((N + block_size - 1) / block_size) * block_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &full_size, &block_size, 0, nullptr, nullptr);

    clFinish(queue);
    
    std::vector<float> output(N);
    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes_sz, output.data(), 0, nullptr, nullptr);

    return output;
}