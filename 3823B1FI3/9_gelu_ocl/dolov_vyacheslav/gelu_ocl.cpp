#define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>

const char* ocl_gelu_code = R"(
__kernel void run_gelu(__global const float* src, __global float* dst, int total) {
    int idx = get_global_id(0);
    if (idx < total) {
        float val = src[idx];
        float cube = val * val * val;
        float continuous_arg = 1.59576912f * (val + 0.044715f * cube);
        dst[idx] = val / (1.0f + native_exp(-continuous_arg));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    int total_count = static_cast<int>(input.size());
    size_t memory_size = total_count * sizeof(float);

    static cl_context ocl_env_context = nullptr;
    static cl_command_queue ocl_cmd_queue = nullptr;
    static cl_kernel ocl_compute_kernel = nullptr;
    static cl_mem dev_input_vectors = nullptr;
    static cl_mem dev_output_vectors = nullptr;
    static size_t cached_buffer_bytes = 0;
    static bool is_env_ready = false;

    if (!is_env_ready) {
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms_list(platform_count);
        clGetPlatformIDs(platform_count, platforms_list.data(), nullptr);

        cl_platform_id target_platform = platforms_list[platform];

        cl_device_id gpu_device_id;
        clGetDeviceIDs(target_platform, CL_DEVICE_TYPE_GPU, 1, &gpu_device_id, nullptr);

        ocl_env_context = clCreateContext(nullptr, 1, &gpu_device_id, nullptr, nullptr, nullptr);
        ocl_cmd_queue = clCreateCommandQueue(ocl_env_context, gpu_device_id, 0, nullptr);

        cl_program ocl_prog = clCreateProgramWithSource(ocl_env_context, 1, &ocl_gelu_code, nullptr, nullptr);
        clBuildProgram(ocl_prog, 1, &gpu_device_id, "-cl-fast-relaxed-math", nullptr, nullptr);
        
        ocl_compute_kernel = clCreateKernel(ocl_prog, "run_gelu", nullptr);
        clReleaseProgram(ocl_prog);
        
        is_env_ready = true;
    }

    if (memory_size > cached_buffer_bytes) {
        if (dev_input_vectors) clReleaseMemObject(dev_input_vectors);
        if (dev_output_vectors) clReleaseMemObject(dev_output_vectors);

        dev_input_vectors = clCreateBuffer(ocl_env_context, CL_MEM_READ_ONLY, memory_size, nullptr, nullptr);
        dev_output_vectors = clCreateBuffer(ocl_env_context, CL_MEM_WRITE_ONLY, memory_size, nullptr, nullptr);
        cached_buffer_bytes = memory_size;
    }

    clEnqueueWriteBuffer(ocl_cmd_queue, dev_input_vectors, CL_FALSE, 0, memory_size, input.data(), 0, nullptr, nullptr);

    clSetKernelArg(ocl_compute_kernel, 0, sizeof(cl_mem), &dev_input_vectors);
    clSetKernelArg(ocl_compute_kernel, 1, sizeof(cl_mem), &dev_output_vectors);
    clSetKernelArg(ocl_compute_kernel, 2, sizeof(int), &total_count);

    size_t local_exec_size = 256;
    size_t global_exec_size = ((total_count + local_exec_size - 1) / local_exec_size) * local_exec_size;

    clEnqueueNDRangeKernel(ocl_cmd_queue, ocl_compute_kernel, 1, nullptr, &global_exec_size, &local_exec_size, 0, nullptr, nullptr);

    std::vector<float> host_output_buffer(total_count);
    clEnqueueReadBuffer(ocl_cmd_queue, dev_output_vectors, CL_TRUE, 0, memory_size, host_output_buffer.data(), 0, nullptr, nullptr);

    return host_output_buffer;
}