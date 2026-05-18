#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <vector>
#include <cstring>

#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("fma,avx2")

static const char* kernel_src = R"(
__kernel void gelu(__global const float* input, __global float* res, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    const float sqrt_2_div_pi = 0.7978845608028653558798921198687637369517172623298693153318516593f;
    float x = input[i];
    res[i] = x * (1.0f - 1.0f / (native_exp(2.0f * sqrt_2_div_pi * (x + 0.044715f * x * x * x)) + 1.0f));
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int n = input.size();
    std::vector<float> output(n);

    static cl_device_id device = nullptr;
    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_kernel kernel = nullptr;
    static cl_mem buf_in = nullptr, buf_out = nullptr;
    static int capacity = 0;
    static size_t local_size = 128;

    if (!context) {
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueueWithProperties(context, device, 0, nullptr);

        cl_program prog = clCreateProgramWithSource(context, 1, &kernel_src, nullptr, nullptr);
        clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(prog, "gelu", nullptr);
        clReleaseProgram(prog);

        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, nullptr);
    }

    if (n > capacity) {
        if (capacity > 0) {
            clReleaseMemObject(buf_in);

            clReleaseMemObject(buf_out);
        }
        buf_in  = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * n, nullptr, nullptr);
        buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * n, nullptr, nullptr);

        capacity = n;
    }

    float* mapped = (float*)clEnqueueMapBuffer(queue, buf_in, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(float) * n, 0, nullptr, nullptr, nullptr);
    memcpy(mapped, input.data(), sizeof(float) * n);
    cl_event write_event;
    clEnqueueUnmapMemObject(queue, buf_in, mapped, 0, nullptr, &write_event);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 2, sizeof(int),    &n);

    size_t global = ((n + local_size - 1) / local_size) * local_size;
    cl_event kernel_event;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local_size, 1, &write_event, &kernel_event);

    clEnqueueReadBuffer(queue, buf_out, CL_FALSE, 0, sizeof(float) * n,output.data(), 1, &kernel_event, nullptr);

    clFinish(queue);

    clReleaseEvent(write_event);
    clReleaseEvent(kernel_event);

    return output;
}