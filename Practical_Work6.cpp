#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>

#define CL_CHECK(call)                                                       \
    do {                                                                     \
        cl_int err = (call);                                                 \
        if (err != CL_SUCCESS) {                                             \
            std::cerr << "OpenCL error: " << err                             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// Утилита: чтение текста файла (kernel.cl)

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Failed to open file: " << path << "\n";
        std::exit(1);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}


// Печать имени платформы/устройства

static std::string get_platform_str(cl_platform_id p, cl_platform_info param) {
    size_t sz = 0;
    CL_CHECK(clGetPlatformInfo(p, param, 0, nullptr, &sz));
    std::string out(sz, '\0');
    CL_CHECK(clGetPlatformInfo(p, param, sz, out.data(), nullptr));
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

static std::string get_device_str(cl_device_id d, cl_device_info param) {
    size_t sz = 0;
    CL_CHECK(clGetDeviceInfo(d, param, 0, nullptr, &sz));
    std::string out(sz, '\0');
    CL_CHECK(clGetDeviceInfo(d, param, sz, out.data(), nullptr));
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}


// Выбор устройства: CPU или GPU 


static bool pick_device(cl_device_type type,
                        cl_platform_id& out_platform,
                        cl_device_id& out_device)
{
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    for (cl_platform_id p : platforms) {
        cl_uint num_devices = 0;
        cl_int err = clGetDeviceIDs(p, type, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        std::vector<cl_device_id> devs(num_devices);
        CL_CHECK(clGetDeviceIDs(p, type, num_devices, devs.data(), nullptr));

        // Берём первое устройство этого типа
        out_platform = p;
        out_device = devs[0];
        return true;
    }

    return false;
}


// ASCII-столбик для графика в консоли

static std::string bar(double v, double vmax, int width = 30) {
    if (vmax <= 0.0) vmax = 1.0;
    double ratio = v / vmax;
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    int n = (int)std::round(ratio * width);
    n = std::max(0, std::min(width, n));
    return std::string(n, '#') + std::string(width - n, '.');
}


// TASK 1: Vector Add (OpenCL) на выбранном устройстве
// Возвращаем время ядра (ms) по профилированию события.

static double run_vector_add_opencl(cl_device_type dtype,
                                   int n,
                                   const std::string& kernel_path,
                                   float& out_max_abs_err,
                                   bool& ok_device)
{
    ok_device = false;
    out_max_abs_err = 0.0f;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    if (!pick_device(dtype, platform, device)) {
        return 0.0;
    }
    ok_device = true;

    // Создаём контекст
    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    // Очередь команд с профилированием (чтобы измерить kernel time)
    cl_command_queue queue = nullptr;
#if defined(CL_VERSION_2_0)
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(ctx, device, props, &err);
    CL_CHECK(err);
#else
    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);
#endif

    // Читаем kernel.cl и собираем программу
    std::string src = read_text_file(kernel_path);
    const char* csrc = src.c_str();
    size_t srclen = src.size();

    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
    CL_CHECK(err);

    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Печатаем лог сборки, чтобы было понятно, что не так
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        std::exit(1);
    }

    cl_kernel kernel = clCreateKernel(prog, "vector_add", &err);
    CL_CHECK(err);

    // Подготовка данных (host)
    std::vector<float> A(n), B(n), C(n), Ref(n);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
        Ref[i] = A[i] + B[i];
    }

    // Буферы (device)
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n * sizeof(float), A.data(), &err);
    CL_CHECK(err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, n * sizeof(float), B.data(), &err);
    CL_CHECK(err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // Аргументы ядра
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n));

    // Глобальный размер — по числу элементов
    size_t global = (size_t)n;

    // Запуск ядра + профилирование времени
    cl_event ev{};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &ev));
    CL_CHECK(clWaitForEvents(1, &ev));

    // Считываем результат
    CL_CHECK(clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, n * sizeof(float), C.data(), 0, nullptr, nullptr));

    // Проверка корректности
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(C[i] - Ref[i]));
    }
    out_max_abs_err = max_err;

    // Время ядра по событию (наносекунды -> миллисекунды)
    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr));
    double kernel_ms = (double)(t1 - t0) * 1e-6;

    // cleanup
    clReleaseEvent(ev);
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return kernel_ms;
}


// CPU baseline для vector add (последовательно)

static double cpu_vector_add(int n) {
    std::vector<float> A(n), B(n), C(n);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count();
}


// TASK 2: Matrix multiply (OpenCL) на выбранном устройстве
// Проверка корректности сравнением с CPU.

static double run_matmul_opencl(cl_device_type dtype,
                                int N, int M, int K,
                                const std::string& kernel_path,
                                float& out_max_abs_err,
                                bool& ok_device)
{
    ok_device = false;
    out_max_abs_err = 0.0f;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    if (!pick_device(dtype, platform, device)) {
        return 0.0;
    }
    ok_device = true;

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    cl_command_queue queue = nullptr;
#if defined(CL_VERSION_2_0)
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(ctx, device, props, &err);
    CL_CHECK(err);
#else
    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);
#endif

    std::string src = read_text_file(kernel_path);
    const char* csrc = src.c_str();
    size_t srclen = src.size();

    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
    CL_CHECK(err);

    err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        std::exit(1);
    }

    cl_kernel kernel = clCreateKernel(prog, "matmul", &err);
    CL_CHECK(err);

    // Подготовка матриц
    std::vector<float> A((size_t)N * M), B((size_t)M * K), C((size_t)N * K), Ref((size_t)N * K);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // CPU reference (для корректности)
    {
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < K; ++c) {
                float s = 0.0f;
                for (int i = 0; i < M; ++i) s += A[r * M + i] * B[i * K + c];
                Ref[r * K + c] = s;
            }
        }
    }

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(float), A.data(), &err);
    CL_CHECK(err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(float), B.data(), &err);
    CL_CHECK(err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, C.size() * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &M));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &K));

    // Глобальная рабочая группа по размеру C: (N, K)
    size_t global[2] = { (size_t)N, (size_t)K };

    cl_event ev{};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, &ev));
    CL_CHECK(clWaitForEvents(1, &ev));

    CL_CHECK(clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, C.size() * sizeof(float), C.data(), 0, nullptr, nullptr));

    float max_err = 0.0f;
    for (size_t i = 0; i < C.size(); ++i) {
        max_err = std::max(max_err, std::fabs(C[i] - Ref[i]));
    }
    out_max_abs_err = max_err;

    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr));
    double kernel_ms = (double)(t1 - t0) * 1e-6;

    clReleaseEvent(ev);
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return kernel_ms;
}

int main() {
    std::cout << "Practical_Work6\n\n";

    
    // TASK 1: Vector Add CPU vs GPU
    
    std::cout << "TASK 1\n";
    std::cout << "OpenCL Vector Add (A+B=C), comparing CPU device vs GPU device\n\n";

    std::string kernel_file = "kernel.cl";

    std::vector<int> sizes = {10'000, 100'000, 1'000'000};

    std::vector<double> cpu_ms(sizes.size(), 0.0);
    std::vector<double> ocl_cpu_ms(sizes.size(), 0.0);
    std::vector<double> ocl_gpu_ms(sizes.size(), 0.0);

    bool has_ocl_cpu = false;
    bool has_ocl_gpu = false;

    for (size_t i = 0; i < sizes.size(); ++i) {
        int n = sizes[i];

        // CPU baseline (обычный C++ цикл)
        cpu_ms[i] = cpu_vector_add(n);

        // OpenCL on CPU device
        float err_cpu = 0.0f;
        bool ok_cpu = false;
        double t_cpu = run_vector_add_opencl(CL_DEVICE_TYPE_CPU, n, kernel_file, err_cpu, ok_cpu);
        if (ok_cpu) {
            has_ocl_cpu = true;
            ocl_cpu_ms[i] = t_cpu;
        }

        // OpenCL on GPU device
        float err_gpu = 0.0f;
        bool ok_gpu = false;
        double t_gpu = run_vector_add_opencl(CL_DEVICE_TYPE_GPU, n, kernel_file, err_gpu, ok_gpu);
        if (ok_gpu) {
            has_ocl_gpu = true;
            ocl_gpu_ms[i] = t_gpu;
        }
    }

    // Таблица результатов
    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(18) << "CPU(ms)"
              << std::setw(18) << "OpenCL CPU(ms)"
              << std::setw(18) << "OpenCL GPU(ms)"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << std::left
                  << std::setw(12) << sizes[i]
                  << std::setw(18) << std::fixed << std::setprecision(3) << cpu_ms[i]
                  << std::setw(18) << std::fixed << std::setprecision(3) << (has_ocl_cpu ? ocl_cpu_ms[i] : 0.0)
                  << std::setw(18) << std::fixed << std::setprecision(3) << (has_ocl_gpu ? ocl_gpu_ms[i] : 0.0)
                  << "\n";
    }

    // “График” сравнения: берём максимум среди всех значений (для масштаба)
    double vmax = 0.0;
    for (double v : cpu_ms) vmax = std::max(vmax, v);
    if (has_ocl_cpu) for (double v : ocl_cpu_ms) vmax = std::max(vmax, v);
    if (has_ocl_gpu) for (double v : ocl_gpu_ms) vmax = std::max(vmax, v);

    std::cout << "\nVector Add Graph (ASCII)\n";
    std::cout << "Legend: # = more time, . = less time\n\n";

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << "N=" << sizes[i] << "\n";
        std::cout << "  CPU        : " << std::setw(8) << std::fixed << std::setprecision(3) << cpu_ms[i]
                  << " ms | " << bar(cpu_ms[i], vmax) << "\n";
        std::cout << "  OpenCL CPU : " << std::setw(8) << std::fixed << std::setprecision(3) << (has_ocl_cpu ? ocl_cpu_ms[i] : 0.0)
                  << " ms | " << bar((has_ocl_cpu ? ocl_cpu_ms[i] : 0.0), vmax) << "\n";
        std::cout << "  OpenCL GPU : " << std::setw(8) << std::fixed << std::setprecision(3) << (has_ocl_gpu ? ocl_gpu_ms[i] : 0.0)
                  << " ms | " << bar((has_ocl_gpu ? ocl_gpu_ms[i] : 0.0), vmax) << "\n\n";
    }

    
    // TASK 2: Matrix Multiply (OpenCL) + correctness vs CPU
    
    std::cout << "TASK 2\n";
    std::cout << "OpenCL Matrix Multiply (C = A x B), checking correctness vs CPU\n\n";

    // Размеры матриц можно менять, но для учебного теста лучше не слишком большие,
    // иначе CPU reference будет считать долго.
    const int N = 256;
    const int M = 256;
    const int K = 256;

    std::cout << "Matrix sizes: A(" << N << "x" << M << "), B(" << M << "x" << K << "), C(" << N << "x" << K << ")\n\n";

    float err_cpu_dev = 0.0f, err_gpu_dev = 0.0f;
    bool ok_cpu_dev = false, ok_gpu_dev = false;

    double mm_cpu_ms = run_matmul_opencl(CL_DEVICE_TYPE_CPU, N, M, K, kernel_file, err_cpu_dev, ok_cpu_dev);
    double mm_gpu_ms = run_matmul_opencl(CL_DEVICE_TYPE_GPU, N, M, K, kernel_file, err_gpu_dev, ok_gpu_dev);

    std::cout << "OpenCL CPU device: " << (ok_cpu_dev ? "OK" : "NOT AVAILABLE") << "\n";
    if (ok_cpu_dev) {
        std::cout << "  Kernel time: " << std::fixed << std::setprecision(3) << mm_cpu_ms << " ms\n";
        std::cout << "  Max abs error vs CPU ref: " << std::scientific << err_cpu_dev << "\n";
    }

    std::cout << "\nOpenCL GPU device: " << (ok_gpu_dev ? "OK" : "NOT AVAILABLE") << "\n";
    if (ok_gpu_dev) {
        std::cout << "  Kernel time: " << std::fixed << std::setprecision(3) << mm_gpu_ms << " ms\n";
        std::cout << "  Max abs error vs CPU ref: " << std::scientific << err_gpu_dev << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
