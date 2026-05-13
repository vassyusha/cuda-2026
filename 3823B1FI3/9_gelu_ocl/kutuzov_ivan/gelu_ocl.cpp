#include <CL/cl.h>
#include <vector>
#include <cstring>
#include <cstdlib>

static const char* kernel_src = R"(
__kernel void gelu(__global const float* input, __global float* res, int n) {
    const int gid = get_global_id(0);
    const int i   = gid << 2;            // i = gid*4

    // Early exit not needed if global size is correctly rounded
    if (i >= n) return;

    const float coeff = 0.044715f;
    const float scale = 1.595769f;       // 2 * sqrt(2/pi)

    // Process 4 elements at a time when possible
    if (i + 3 < n) {
        float4 x = vload4(0, input + i);

        float4 x2 = x * x;
        float4 x3 = x2 * x;
        float4 arg = scale * (x + coeff * x3);

        float4 exp_val = native_exp(arg);
        float4 out = x * exp_val / (exp_val + 1.0f);

        vstore4(out, 0, res + i);
    } else {
        // Scalar tail for the last 1–3 elements
        for (int j = i; j < n; ++j) {
            float x   = input[j];
            float x2  = x * x;
            float x3  = x2 * x;
            float arg = scale * (x + coeff * x3);
            float exp_val = native_exp(arg);
            res[j] = x * exp_val / (exp_val + 1.0f);
        }
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    static cl_device_id device = nullptr;
    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_kernel kernel = nullptr;
    static cl_mem buf_in = nullptr;
    static cl_mem buf_out = nullptr;
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
        const char* build_opts = "-cl-fast-relaxed-math -cl-mad-enable";
        clBuildProgram(prog, 1, &device, build_opts, nullptr, nullptr);
        kernel = clCreateKernel(prog, "gelu", nullptr);
        clReleaseProgram(prog);

        clGetKernelWorkGroupInfo(kernel, device,
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(local_size), &local_size, nullptr);
    }

    if (n > capacity) {
        if (capacity > 0) {
            clReleaseMemObject(buf_in);
            clReleaseMemObject(buf_out);
        }
        buf_in  = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(float) * n, nullptr, nullptr);
        buf_out = clCreateBuffer(context,
                                 CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(float) * n, nullptr, nullptr);
        capacity = n;
    }

    cl_event write_event;
    clEnqueueWriteBuffer(queue, buf_in, CL_FALSE, 0,
                         sizeof(float) * n, input.data(),
                         0, nullptr, &write_event);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 2, sizeof(int),    &n);

    size_t global = (n + 3) / 4;
    global = ((global + local_size - 1) / local_size) * local_size;

    cl_event kernel_event;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                            &global, &local_size,
                            1, &write_event, &kernel_event);

    clEnqueueReadBuffer(queue, buf_out, CL_FALSE, 0,
                        sizeof(float) * n, output.data(),
                        1, &kernel_event, nullptr);

    clFinish(queue);

    clReleaseEvent(write_event);
    clReleaseEvent(kernel_event);

    return output;
}