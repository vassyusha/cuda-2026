#define CL_TARGET_OPENCL_VERSION 200

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>

static const char* ocl_gelu_src = R"CLC(
#define COEFF 1.5957691216057308f

__kernel void gelu_fast_kernel(__global const float* d_in, __global float* d_out, const int size) {
    int i = get_global_id(0);
    
    if (i < size) {
        float v = d_in[i];
        float v3 = v * v * v;
        
        float p = COEFF * (v + 0.044715f * v3);
        
        d_out[i] = v * (1.0f - (1.0f / (1.0f + exp(p))));
    }
}
)CLC";

struct OclState {
    int plat_id = -1;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel kern = nullptr;
    cl_mem buf_in = nullptr;
    cl_mem buf_out = nullptr;
    size_t mem_cap = 0;
};

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int len = input.size();
    if (len == 0) return std::vector<float>();

    size_t byte_len = len * sizeof(float);
    static OclState state;

    if (state.plat_id != platform) {
        if (state.kern) clReleaseKernel(state.kern);
        if (state.prog) clReleaseProgram(state.prog);
        if (state.q) clReleaseCommandQueue(state.q);
        if (state.ctx) clReleaseContext(state.ctx);

        cl_uint p_count;
        clGetPlatformIDs(0, nullptr, &p_count);
        std::vector<cl_platform_id> p_list(p_count);
        clGetPlatformIDs(p_count, p_list.data(), nullptr);

        cl_device_id dev_id;
        clGetDeviceIDs(p_list[platform], CL_DEVICE_TYPE_GPU, 1, &dev_id, nullptr);

        state.ctx = clCreateContext(nullptr, 1, &dev_id, nullptr, nullptr, nullptr);
        state.q = clCreateCommandQueueWithProperties(state.ctx, dev_id, nullptr, nullptr);

        state.prog = clCreateProgramWithSource(state.ctx, 1, &ocl_gelu_src, nullptr, nullptr);
        clBuildProgram(state.prog, 1, &dev_id, "-cl-fast-relaxed-math", nullptr, nullptr);
        
        state.kern = clCreateKernel(state.prog, "gelu_fast_kernel", nullptr);
        
        state.plat_id = platform;
        state.mem_cap = 0; 
    }

    if (state.mem_cap != byte_len) {
        if (state.buf_in) clReleaseMemObject(state.buf_in);
        if (state.buf_out) clReleaseMemObject(state.buf_out);

        state.buf_in = clCreateBuffer(state.ctx, CL_MEM_READ_ONLY, byte_len, nullptr, nullptr);
        state.buf_out = clCreateBuffer(state.ctx, CL_MEM_WRITE_ONLY, byte_len, nullptr, nullptr);
        state.mem_cap = byte_len;
    }

    clEnqueueWriteBuffer(state.q, state.buf_in, CL_FALSE, 0, byte_len, input.data(), 0, nullptr, nullptr);

    clSetKernelArg(state.kern, 0, sizeof(cl_mem), &state.buf_in);
    clSetKernelArg(state.kern, 1, sizeof(cl_mem), &state.buf_out);
    clSetKernelArg(state.kern, 2, sizeof(int), &len);

    size_t local_w = 128;
    size_t global_w = ((len + local_w - 1) / local_w) * local_w;

    clEnqueueNDRangeKernel(state.q, state.kern, 1, nullptr, &global_w, &local_w, 0, nullptr, nullptr);

    std::vector<float> res(len);

    clEnqueueReadBuffer(state.q, state.buf_out, CL_TRUE, 0, byte_len, res.data(), 0, nullptr, nullptr);

    return res;
}