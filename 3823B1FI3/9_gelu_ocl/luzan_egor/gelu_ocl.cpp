// #define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>

static const char* kSrc =
"__kernel void gelu(__global const float* in, __global float* out, int n) {"
"    int i = get_global_id(0);"
"    if (i >= n) return;"
"    float x = in[i];"
"    float z = 0.7978845608f * (x + 0.044715f * x * x * x);"
"    float e = native_exp(-2.0f * z);"
"    out[i] = 0.5f * x * (2.0f / (e + 1.0f));"
"}";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    static cl_context context = nullptr;
    static cl_command_queue que = nullptr;
    static cl_kernel kernel = nullptr;

    if (!context) {
        cl_uint np;
        clGetPlatformIDs(0, nullptr, &np);
        std::vector<cl_platform_id> platforms(np);
        clGetPlatformIDs(np, platforms.data(), nullptr);

        cl_device_id device;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        que = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);

        cl_program program = clCreateProgramWithSource(context, 1, &kSrc, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu", nullptr);
        clReleaseProgram(program);
    }

    const int n = static_cast<int>(input.size());
    const size_t bytes = n * sizeof(float);

    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, const_cast<float*>(input.data()), nullptr);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global = n;
    clEnqueueNDRangeKernel(que, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    std::vector<float> output(n);
    clEnqueueReadBuffer(que, d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);

    return output;
}
