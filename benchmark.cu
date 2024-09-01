#include <functional>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void clear_l2() {
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2; // just to be extra safe (cache is not necessarily strict LRU)
        CHECK_CUDA_ERROR(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    CHECK_CUDA_ERROR(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}


#define TIMINGS 15
#define START 10

float gt[TIMINGS];
float ct[TIMINGS];

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    float time;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    float latency = 0.f;

    for (unsigned int i{0}; i < num_repeats; ++i)
    {
      clear_l2();
      CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
      bound_function(stream);
      CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
      CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
      CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
      CHECK_LAST_CUDA_ERROR();
      latency += time / num_repeats;
    }
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return latency;
}

// Use all the constant memory.
constexpr unsigned int N{64U * 1024U / sizeof(float)};
__constant__ float const_values[N];

// Magic number for generating the pseudo-random access pattern.
constexpr unsigned int magic_number{1357U};

enum struct AccessPattern
{
    OneAccessPerBlock,
    OneAccessPerWarp,
    OneAccessPerThread,
    PseudoRandom
};

void add_constant_cpu(float* sums, float const* inputs, float const* values,
                      unsigned int num_sums, unsigned int num_values,
                      unsigned int block_size, AccessPattern access_pattern)
{
    for (unsigned int i{0U}; i < num_sums; ++i)
    {
        unsigned int const block_id{i / block_size};
        unsigned int const thread_id{i % block_size};
        unsigned int const warp_id{thread_id / 32U};
        unsigned int index{0U};

        switch (access_pattern)
        {
            case AccessPattern::OneAccessPerBlock:
                index = block_id % num_values;
                break;
            case AccessPattern::OneAccessPerWarp:
                index = warp_id % num_values;
                break;
            case AccessPattern::OneAccessPerThread:
                index = thread_id % num_values;
                break;
            case AccessPattern::PseudoRandom:
                index = (thread_id * magic_number) % num_values;
                break;
        }

        sums[i] = inputs[i] + values[index];
    }
}

__global__ void add_constant_global_memory(
    float* sums, float const* inputs, float const* values, unsigned int num_sums,
    unsigned int num_values,
    AccessPattern access_pattern = AccessPattern::OneAccessPerBlock)
{
    unsigned int const i{blockIdx.x * blockDim.x + threadIdx.x};
    unsigned int const block_id{blockIdx.x};
    unsigned int const thread_id{threadIdx.x};
    unsigned int const warp_id{threadIdx.x / warpSize};
    unsigned int index{0U};

    switch (access_pattern)
    {
        case AccessPattern::OneAccessPerBlock:
            index = block_id % num_values;
            break;
        case AccessPattern::OneAccessPerWarp:
            index = warp_id % num_values;
            break;
        case AccessPattern::OneAccessPerThread:
            index = thread_id % num_values;
            break;
        case AccessPattern::PseudoRandom:
            index = (thread_id * magic_number) % num_values;
            break;
    }

    if (i < num_sums)
    {
        sums[i] = inputs[i] + values[index];
    }
}

void launch_add_constant_global_memory(float* sums, float const* inputs,
                                       float const* values, unsigned int num_sums,
                                       unsigned int num_values,
                                       unsigned int block_size,
                                       AccessPattern access_pattern,
                                       cudaStream_t stream)
{
    add_constant_global_memory<<<(num_sums + block_size - 1) / block_size,
                                 block_size, 0, stream>>>(
        sums, inputs, values, num_sums, num_values, access_pattern);
    CHECK_LAST_CUDA_ERROR();
}

__global__ void add_constant_constant_memory(float* sums, float const* inputs,
                                             unsigned int num_sums,
                                             AccessPattern access_pattern)
{
    unsigned int const i{blockIdx.x * blockDim.x + threadIdx.x};
    unsigned int const block_id{blockIdx.x};
    unsigned int const thread_id{threadIdx.x};
    unsigned int const warp_id{threadIdx.x / warpSize};
    unsigned int index{0U};

    switch (access_pattern)
    {
        case AccessPattern::OneAccessPerBlock:
            index = block_id % N;
            break;
        case AccessPattern::OneAccessPerWarp:
            index = warp_id % N;
            break;
        case AccessPattern::OneAccessPerThread:
            index = thread_id % N;
            break;
        case AccessPattern::PseudoRandom:
            index = (thread_id * magic_number) % N;
            break;
    }

    if (i < num_sums)
    {
        sums[i] = inputs[i] + const_values[index];
    }
}

void launch_add_constant_constant_memory(float* sums, float const* inputs,
                                         unsigned int num_sums,
                                         unsigned int block_size,
                                         AccessPattern access_pattern,
                                         cudaStream_t stream)
{
    add_constant_constant_memory<<<(num_sums + block_size - 1) / block_size,
                                   block_size, 0, stream>>>(
        sums, inputs, num_sums, access_pattern);
    CHECK_LAST_CUDA_ERROR();
}

void parse_args(int argc, char** argv, AccessPattern& access_pattern,
                unsigned int& block_size)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <access pattern> <block size>"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string const access_pattern_str{argv[1]};
    if (access_pattern_str == "one_access_per_block")
    {
        access_pattern = AccessPattern::OneAccessPerBlock;
    }
    else if (access_pattern_str == "one_access_per_warp")
    {
        access_pattern = AccessPattern::OneAccessPerWarp;
    }
    else if (access_pattern_str == "one_access_per_thread")
    {
        access_pattern = AccessPattern::OneAccessPerThread;
    }
    else if (access_pattern_str == "pseudo_random")
    {
        access_pattern = AccessPattern::PseudoRandom;
    }
    else
    {
        std::cerr << "Invalid access pattern: " << access_pattern_str
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    block_size = std::stoi(argv[2]);
}

int main(int argc, char** argv)
{
    constexpr unsigned int num_warmups{1U};
    constexpr unsigned int num_repeats{100U};

    AccessPattern access_pattern{AccessPattern::OneAccessPerBlock};
    unsigned int block_size{1024U};
    unsigned long num_sums{12800000U};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    // Modify access pattern, block size and number of sums from command line.
    parse_args(argc, argv, access_pattern, block_size);

    for (int p = START; p<START+TIMINGS; p++)
    {
      num_sums = std::pow<long, long>(2, p);

      float h_values[N];
      // Initialize values on host memory.
      for (unsigned int i{0U}; i < N; ++i)
      {
        h_values[i] = i;
      }
      // Initialize values on global memory.
      float* d_values;
      CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, N * sizeof(float), stream));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_values, h_values, N * sizeof(float),
            cudaMemcpyHostToDevice, stream));
      // Initialize values on constant memory.
      CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(const_values, h_values,
            N * sizeof(float), 0,
            cudaMemcpyHostToDevice, stream));

      std::vector<float> inputs(num_sums, 0);
      float* h_inputs{inputs.data()};
      float* d_inputs_for_constant;
      float* d_inputs_for_global;
      CHECK_CUDA_ERROR(cudaMallocAsync(&d_inputs_for_constant,
            num_sums * sizeof(float), stream));
      CHECK_CUDA_ERROR(
          cudaMallocAsync(&d_inputs_for_global, num_sums * sizeof(float), stream));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_inputs_for_constant, h_inputs,
            num_sums * sizeof(float),
            cudaMemcpyHostToDevice, stream));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_inputs_for_global, h_inputs,
            num_sums * sizeof(float),
            cudaMemcpyHostToDevice, stream));

      std::vector<float> reference_sums(num_sums, 0);
      std::vector<float> sums_from_constant(num_sums, 1);
      std::vector<float> sums_from_global(num_sums, 2);

      float* h_reference_sums{reference_sums.data()};
      float* h_sums_from_constant{sums_from_constant.data()};
      float* h_sums_from_global{sums_from_global.data()};

      float* d_sums_from_constant;
      float* d_sums_from_global;
      CHECK_CUDA_ERROR(
          cudaMallocAsync(&d_sums_from_constant, num_sums * sizeof(float), stream));
      CHECK_CUDA_ERROR(
          cudaMallocAsync(&d_sums_from_global, num_sums * sizeof(float), stream));

      // Synchronize.
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

      // Compute reference sums on CPU.
      add_constant_cpu(h_reference_sums, h_inputs, h_values, num_sums, N,
          block_size, access_pattern);
      // Compute reference sums on GPU using global memory.
      launch_add_constant_global_memory(d_sums_from_global, d_inputs_for_global,
          d_values, num_sums, N, block_size,
          access_pattern, stream);
      // Compute reference sums on GPU using constant memory.
      launch_add_constant_constant_memory(d_sums_from_constant,
          d_inputs_for_constant, num_sums,
          block_size, access_pattern, stream);

      // Copy results from device to host.
      CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sums_from_constant, d_sums_from_constant,
            num_sums * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sums_from_global, d_sums_from_global,
            num_sums * sizeof(int),
            cudaMemcpyDeviceToHost, stream));

      // Synchronize.
      CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

      // Verify results.
      for (unsigned int i{0U}; i < num_sums; ++i)
      {
        if (h_reference_sums[i] != h_sums_from_constant[i])
        {
          std::cerr << "Error at index " << i << " for constant memory."
            << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (h_reference_sums[i] != h_sums_from_global[i])
        {
          std::cerr << "Error at index " << i << " for global memory."
            << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }

      // Measure performance.
      std::function<void(cudaStream_t)> bound_function_constant_memory{
        std::bind(launch_add_constant_constant_memory, d_sums_from_constant,
            d_inputs_for_constant, num_sums, block_size, access_pattern,
            std::placeholders::_1)};
      std::function<void(cudaStream_t)> bound_function_global_memory{
        std::bind(launch_add_constant_global_memory, d_sums_from_global,
            d_inputs_for_global, d_values, num_sums, N, block_size,
            access_pattern, std::placeholders::_1)};

      float const latency_constant_memory{measure_performance(
          bound_function_constant_memory, stream, num_repeats, num_warmups)};

      float const latency_global_memory{measure_performance(
          bound_function_global_memory, stream, num_repeats, num_warmups)};
      // std::cout << " difference = "<<latency_global_memory-latency_constant_memory
      //   <<" blocks = "<<(num_sums + block_size - 1) / block_size<<std::endl;
      std::cout << "Finished for num_sums = " << num_sums
        << "Latency for Add using constant memory: "
        << latency_constant_memory << " ms" << " global memory: "
        << latency_global_memory << " ms" << std::endl << std::endl;
      gt[p-START] = latency_global_memory;
      ct[p-START] = latency_constant_memory;
      

      CHECK_CUDA_ERROR(cudaFree(d_values));
      CHECK_CUDA_ERROR(cudaFree(d_inputs_for_constant));
      CHECK_CUDA_ERROR(cudaFree(d_inputs_for_global));
      CHECK_CUDA_ERROR(cudaFree(d_sums_from_constant));
      CHECK_CUDA_ERROR(cudaFree(d_sums_from_global));
    }
      CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    std::cout<<"normal_t = [";
    for (int i = 0; i<TIMINGS; i++)
    {
      std::cout<<gt[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    std::cout<<"const_t = [";
    for (int i = 0; i<TIMINGS; i++)
    {
      std::cout<<ct[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    return 0;
}
