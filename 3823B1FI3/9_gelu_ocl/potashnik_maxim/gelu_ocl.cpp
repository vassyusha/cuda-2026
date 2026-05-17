#define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>
#include <vector>

/* Optimizations
1. OpenCL boilerplate init once
2. Static buffer reuse
3. CL_MEM_ALLOC_HOST_PTR
4. clEnqueueMapBuffer
5. Async pipeline with events
6. native_exp
7. CL_KERNEL_WORK_GROUP_SIZE
8. clReleaseProgram after kernel creation 
*/

static const char* kernel_src =
    "__kernel void gelu_kernel("
    "    __global const float* in, __global float* out, int n) {"
    "    int i = get_global_id(0);"
    "    if (i >= n) return;"
    "    float x = in[i];"
    "    float x2 = x * x;"
    "    float arg = 1.5957691216f * x * (1.0f + 0.044715f * x2);"
    "    float e = native_exp(arg);"
    "    out[i] = x * e / (e + 1.0f);"
    "}";

static cl_device_id g_device = nullptr;
static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_kernel g_kernel = nullptr;
static cl_mem g_buf_in = nullptr;
static cl_mem g_buf_out  = nullptr;
static size_t g_local_sz = 128;
static int g_capacity = 0;
static bool g_init = false;

static void init_ocl(int platform) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &g_device, nullptr);

    g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, nullptr);

    cl_queue_properties props[] = {0};
    g_queue = clCreateCommandQueueWithProperties(g_context, g_device, props, nullptr);

    cl_program program = clCreateProgramWithSource(g_context, 1, &kernel_src, nullptr, nullptr);
    clBuildProgram(program, 1, &g_device, nullptr, nullptr, nullptr);
    g_kernel = clCreateKernel(program, "gelu_kernel", nullptr);
    clReleaseProgram(program);

    clGetKernelWorkGroupInfo(g_kernel, g_device, CL_KERNEL_WORK_GROUP_SIZE,  sizeof(g_local_sz), &g_local_sz, nullptr);
}

static void ensure_buffers(int n) {
    size_t bytes = n * sizeof(float);
    if (n > g_capacity) {
        if (g_capacity > 0) {
            clReleaseMemObject(g_buf_in);
            clReleaseMemObject(g_buf_out);
        }
        g_buf_in = clCreateBuffer(g_context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, nullptr);
        g_buf_out = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, nullptr);
        g_capacity = n;
    }
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int n = static_cast<int>(input.size());
    size_t bytes = n * sizeof(float);

    if (!g_init) {
        init_ocl(platform);
        g_init = true;
    }

    ensure_buffers(n);

    float* mapped = (float*)clEnqueueMapBuffer(g_queue, g_buf_in, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, bytes, 0, nullptr, nullptr, nullptr);
    memcpy(mapped, input.data(), bytes);
    cl_event ev_write;
    clEnqueueUnmapMemObject(g_queue, g_buf_in, mapped, 0, nullptr, &ev_write);

    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_buf_in);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_buf_out);
    clSetKernelArg(g_kernel, 2, sizeof(int),    &n);

    size_t global_sz = ((n + g_local_sz - 1) / g_local_sz) * g_local_sz;
    cl_event ev_kernel;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global_sz, &g_local_sz,
                           1, &ev_write, &ev_kernel);

    std::vector<float> result(n);
    cl_event ev_read;
    clEnqueueReadBuffer(g_queue, g_buf_out, CL_FALSE, 0, bytes, result.data(), 1, &ev_kernel, &ev_read);

    clWaitForEvents(1, &ev_read);

    clReleaseEvent(ev_write);
    clReleaseEvent(ev_kernel);
    clReleaseEvent(ev_read);

    return result;
}