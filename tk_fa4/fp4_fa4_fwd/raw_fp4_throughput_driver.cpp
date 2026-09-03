#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    throw std::runtime_error(
        std::string(operation) + ": " + (name ? name : "unknown") +
        " (" + (message ? message : "no description") + ")");
}

struct alignas(8) ProbeGlobals {
    CUdeviceptr a_fp4;
    CUdeviceptr b_fp4;
    CUdeviceptr a_bf16;
    CUdeviceptr b_bf16;
    CUdeviceptr cycles;
    CUdeviceptr start_globaltimer;
    CUdeviceptr end_globaltimer;
    CUdeviceptr smids;
    int32_t iterations;
    int32_t blocks;
};

static_assert(sizeof(ProbeGlobals) == 72);

template <typename T>
double median(std::vector<T> values) {
    if (values.empty()) {
        throw std::runtime_error("cannot take median of an empty sample");
    }
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 != 0) {
        return static_cast<double>(values[middle]);
    }
    const T upper = values[middle];
    std::nth_element(
        values.begin(), values.begin() + middle - 1, values.begin() + middle);
    return (static_cast<double>(values[middle - 1]) +
            static_cast<double>(upper)) /
           2.0;
}

int parse_int(char **argv, int argc, int index, int fallback) {
    return index < argc ? std::stoi(argv[index]) : fallback;
}

}  // namespace

int main(int argc, char **argv) try {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " CUBIN KERNEL_SYMBOL [BLOCKS] [ITERATIONS]"
                     " [WARMUP] [SAMPLES] [DYNAMIC_SMEM_KB] [CLUSTER_X]"
                     " [USEFUL_K]\n";
        return 2;
    }
    const std::string cubin_path = argv[1];
    const std::string kernel_symbol = argv[2];
    const int blocks = parse_int(argv, argc, 3, 1776);
    const int iterations = parse_int(argv, argc, 4, 256);
    const int warmup = parse_int(argv, argc, 5, 20);
    const int samples = parse_int(argv, argc, 6, 101);
    const int dynamic_smem_kb = parse_int(argv, argc, 7, 144);
    const int cluster_x = parse_int(argv, argc, 8, 1);
    const int useful_k = parse_int(argv, argc, 9, 192);
    if (cluster_x != 1 && cluster_x != 2) {
        throw std::runtime_error("CLUSTER_X must be 1 or 2");
    }
    if (blocks % cluster_x != 0) {
        throw std::runtime_error("BLOCKS must contain whole clusters");
    }
    if (useful_k <= 0) {
        throw std::runtime_error("USEFUL_K must be positive");
    }

    check(cuInit(0), "cuInit");
    CUdevice device;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context;
    check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");

    CUmodule module;
    check(cuModuleLoad(&module, cubin_path.c_str()), "cuModuleLoad");
    CUfunction kernel;
    check(
        cuModuleGetFunction(&kernel, module, kernel_symbol.c_str()),
        "cuModuleGetFunction");
    const int dynamic_smem = dynamic_smem_kb * 1024;
    check(
        cuFuncSetAttribute(
            kernel,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            dynamic_smem),
        "cuFuncSetAttribute");
    int occupancy_blocks_per_sm = 0;
    check(
        cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy_blocks_per_sm, kernel, 128, dynamic_smem),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor");

    ProbeGlobals globals{};
    check(cuMemAlloc(&globals.a_fp4, 128 * 64), "cuMemAlloc(a_fp4)");
    check(cuMemAlloc(&globals.b_fp4, 128 * 64), "cuMemAlloc(b_fp4)");
    check(cuMemAlloc(&globals.a_bf16, 128 * 128 * 2), "cuMemAlloc(a_bf16)");
    check(cuMemAlloc(&globals.b_bf16, 128 * 128 * 2), "cuMemAlloc(b_bf16)");
    check(cuMemAlloc(&globals.cycles, blocks * sizeof(uint64_t)),
          "cuMemAlloc(cycles)");
    check(cuMemAlloc(&globals.start_globaltimer, blocks * sizeof(uint64_t)),
          "cuMemAlloc(start_globaltimer)");
    check(cuMemAlloc(&globals.end_globaltimer, blocks * sizeof(uint64_t)),
          "cuMemAlloc(end_globaltimer)");
    check(cuMemAlloc(&globals.smids, blocks * sizeof(int32_t)),
          "cuMemAlloc(smids)");
    globals.iterations = iterations;
    globals.blocks = blocks;

    check(cuMemsetD8(globals.a_fp4, 0x11, 128 * 64), "cuMemsetD8(a_fp4)");
    check(cuMemsetD8(globals.b_fp4, 0x11, 128 * 64), "cuMemsetD8(b_fp4)");
    check(cuMemsetD8(globals.a_bf16, 0x3f, 128 * 128 * 2),
          "cuMemsetD8(a_bf16)");
    check(cuMemsetD8(globals.b_bf16, 0x3f, 128 * 128 * 2),
          "cuMemsetD8(b_bf16)");
    check(cuMemsetD8(globals.cycles, 0, blocks * sizeof(uint64_t)),
          "cuMemsetD8(cycles)");
    check(cuMemsetD8(globals.start_globaltimer, 0, blocks * sizeof(uint64_t)),
          "cuMemsetD8(start_globaltimer)");
    check(cuMemsetD8(globals.end_globaltimer, 0, blocks * sizeof(uint64_t)),
          "cuMemsetD8(end_globaltimer)");
    check(cuMemsetD8(globals.smids, 0xff, blocks * sizeof(int32_t)),
          "cuMemsetD8(smids)");

    auto launch = [&] {
        void *parameters[] = {&globals};
        if (cluster_x == 1) {
            check(
                cuLaunchKernel(
                    kernel,
                    blocks, 1, 1,
                    128, 1, 1,
                    dynamic_smem,
                    nullptr,
                    parameters,
                    nullptr),
                "cuLaunchKernel");
        } else {
            CUlaunchAttribute attribute{};
            attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            attribute.value.clusterDim.x = cluster_x;
            attribute.value.clusterDim.y = 1;
            attribute.value.clusterDim.z = 1;
            CUlaunchConfig config{};
            config.gridDimX = blocks;
            config.gridDimY = 1;
            config.gridDimZ = 1;
            config.blockDimX = 128;
            config.blockDimY = 1;
            config.blockDimZ = 1;
            config.sharedMemBytes = dynamic_smem;
            config.hStream = nullptr;
            config.attrs = &attribute;
            config.numAttrs = 1;
            check(
                cuLaunchKernelEx(&config, kernel, parameters, nullptr),
                "cuLaunchKernelEx");
        }
    };

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    check(cuCtxSynchronize(), "warmup synchronize");

    CUevent start;
    CUevent stop;
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
    std::vector<float> times_ms;
    times_ms.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        check(cuEventRecord(start, nullptr), "cuEventRecord(start)");
        launch();
        check(cuEventRecord(stop, nullptr), "cuEventRecord(stop)");
        check(cuEventSynchronize(stop), "cuEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        check(cuEventElapsedTime(&elapsed_ms, start, stop),
              "cuEventElapsedTime");
        times_ms.push_back(elapsed_ms);
    }

    std::vector<uint64_t> cycle_values(blocks);
    std::vector<uint64_t> start_globaltimer_values(blocks);
    std::vector<uint64_t> end_globaltimer_values(blocks);
    std::vector<int32_t> smid_values(blocks);
    check(
        cuMemcpyDtoH(
            cycle_values.data(), globals.cycles,
            blocks * sizeof(uint64_t)),
        "cuMemcpyDtoH(cycles)");
    check(
        cuMemcpyDtoH(
            start_globaltimer_values.data(), globals.start_globaltimer,
            blocks * sizeof(uint64_t)),
        "cuMemcpyDtoH(start_globaltimer)");
    check(
        cuMemcpyDtoH(
            end_globaltimer_values.data(), globals.end_globaltimer,
            blocks * sizeof(uint64_t)),
        "cuMemcpyDtoH(end_globaltimer)");
    check(
        cuMemcpyDtoH(
            smid_values.data(), globals.smids,
            blocks * sizeof(int32_t)),
        "cuMemcpyDtoH(smids)");
    cycle_values.erase(
        std::remove(cycle_values.begin(), cycle_values.end(), uint64_t{0}),
        cycle_values.end());
    std::set<int32_t> observed_sms;
    for (int32_t smid : smid_values) {
        if (smid >= 0) {
            observed_sms.insert(smid);
        }
    }
    std::map<int32_t, std::vector<std::pair<uint64_t, int>>> events_by_sm;
    for (int block = 0; block < blocks; ++block) {
        if (smid_values[block] < 0 ||
            start_globaltimer_values[block] == 0 ||
            end_globaltimer_values[block] <= start_globaltimer_values[block]) {
            continue;
        }
        auto &events = events_by_sm[smid_values[block]];
        events.emplace_back(start_globaltimer_values[block], 1);
        events.emplace_back(end_globaltimer_values[block], -1);
    }
    std::vector<int> maximum_concurrency_by_sm;
    for (auto &[smid, events] : events_by_sm) {
        std::sort(events.begin(), events.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first < rhs.first ||
                (lhs.first == rhs.first && lhs.second < rhs.second);
        });
        int active = 0;
        int maximum = 0;
        for (const auto &[timestamp, delta] : events) {
            active += delta;
            maximum = std::max(maximum, active);
        }
        maximum_concurrency_by_sm.push_back(maximum);
    }

    const double median_ms = median(times_ms);
    const double flops_per_cta_iteration =
        2.0 * 128.0 * 128.0 * static_cast<double>(useful_k);
    const double tflops =
        blocks * static_cast<double>(iterations) * flops_per_cta_iteration /
        (median_ms * 1.0e9);
    std::cout << std::setprecision(12)
              << "{\n"
              << "  \"cubin\": \"" << cubin_path << "\",\n"
              << "  \"kernel\": \"" << kernel_symbol << "\",\n"
              << "  \"blocks\": " << blocks << ",\n"
              << "  \"iterations\": " << iterations << ",\n"
              << "  \"dynamic_smem_kb\": " << dynamic_smem_kb << ",\n"
              << "  \"cluster_x\": " << cluster_x << ",\n"
              << "  \"useful_k\": " << useful_k << ",\n"
              << "  \"occupancy_blocks_per_sm\": "
              << occupancy_blocks_per_sm << ",\n"
              << "  \"median_ms\": " << median_ms << ",\n"
              << "  \"tflops\": " << tflops << ",\n"
              << "  \"median_cta_cycles\": " << median(cycle_values) << ",\n"
              << "  \"median_observed_concurrency\": "
              << median(maximum_concurrency_by_sm) << ",\n"
              << "  \"max_observed_concurrency\": "
              << *std::max_element(
                     maximum_concurrency_by_sm.begin(),
                     maximum_concurrency_by_sm.end()) << ",\n"
              << "  \"observed_sms\": " << observed_sms.size() << "\n"
              << "}\n";

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
