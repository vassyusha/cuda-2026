#include "gelu_ocl.h"
#include <CL/cl.h>

// GELU(x) = x / (1 + exp(-c * (x + 0.044715 * x^3))),  c = 2*sqrt(2/pi)
static const char* kKernelSrc = R"(
__kernel void gelu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float x = in[i];
    float t = 1.5957691216f * (x + 0.044715f * x * x * x);
    out[i] = x / (1.0f + native_exp(-t));
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = static_cast<int>(input.size());
    const size_t bytes = n * sizeof(float);

    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_device_id device;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSrc, nullptr, nullptr);
    clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "gelu", nullptr);

    cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    const size_t local = 256;
    const size_t global = ((n + local - 1) / local) * local;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);

    std::vector<float> output(n);

    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}