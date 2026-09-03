#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kCandidateCount = 4;
constexpr int kThresholdRadiusUlps = 65536;
constexpr size_t kRandomCount = 1 << 20;

constexpr std::array<const char*, kCandidateCount> kCandidateNames = {
    "forward_affine_bias_2p37",
    "aligned_affine_bias_2p3561437",
    "midpoint_affine_bias_2p3707170",
    "forward_magic_simd_scale_1p826841",
};

__device__ __forceinline__ float2 native_exp2_pair(float2 value) {
    float2 output;
    asm(
        "ex2.approx.ftz.f32 %0, %2;\n\t"
        "ex2.approx.ftz.f32 %1, %3;\n"
        : "=f"(output.x), "=f"(output.y)
        : "f"(value.x), "f"(value.y));
    return output;
}

__device__ __forceinline__ uint32_t native_e2m1_pair(float2 value) {
    uint32_t output;
    asm(
        "{\n\t"
        ".reg .b8 packed;\n\t"
        "cvt.rn.relu.satfinite.e2m1x2.f32 packed, %2, %1;\n\t"
        "mov.b32 %0, {packed, 0, 0, 0};\n\t"
        "}\n"
        : "=r"(output)
        : "f"(value.x), "f"(value.y));
    return output;
}

template <uint32_t BiasBits, uint32_t LowThresholdBits>
__device__ __forceinline__ uint32_t affine_floor_pair(float2 log_value) {
    uint32_t packed;
    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .f32 tx, ty;\n\t"
        ".reg .s32 x, y, low;\n\t"
        "fma.rn.f32 tx, %1, 0f40000000, %3;\n\t"
        "fma.rn.f32 ty, %2, 0f40000000, %3;\n\t"
        "cvt.rmi.s32.f32 x, tx;\n\t"
        "max.s32 x, x, 0;\n\t"
        "min.s32 x, x, 7;\n\t"
        "setp.ge.f32 p, %1, %4;\n\t"
        "selp.s32 low, 1, 0, p;\n\t"
        "max.s32 x, x, low;\n\t"
        "setp.ge.f32 p, %1, %5;\n\t"
        "selp.s32 low, 2, 0, p;\n\t"
        "max.s32 x, x, low;\n\t"
        "cvt.rmi.s32.f32 y, ty;\n\t"
        "max.s32 y, y, 0;\n\t"
        "min.s32 y, y, 7;\n\t"
        "setp.ge.f32 p, %2, %4;\n\t"
        "selp.s32 low, 1, 0, p;\n\t"
        "max.s32 y, y, low;\n\t"
        "setp.ge.f32 p, %2, %5;\n\t"
        "selp.s32 low, 2, 0, p;\n\t"
        "max.s32 y, y, low;\n\t"
        "shl.b32 y, y, 4;\n\t"
        "or.b32 %0, x, y;\n\t"
        "}\n"
        : "=r"(packed)
        : "f"(log_value.x),
          "f"(log_value.y),
          "f"(__uint_as_float(BiasBits)),
          "f"(__uint_as_float(LowThresholdBits)),
          "f"(__uint_as_float(0xbed47fcbu)));
    return packed;
}

__device__ __forceinline__ uint32_t magic_simd_pair(float2 log_value) {
    constexpr float kCodeScale = 1.826841f;
    const float2 encoded = {
        __fmaf_rn(log_value.x, kCodeScale, 12582914.0f),
        __fmaf_rn(log_value.y, kCodeScale, 12582914.0f),
    };
    uint32_t candidates = __byte_perm(
        __float_as_uint(encoded.x), __float_as_uint(encoded.y), 0x5410);
    const uint32_t prefixes = __byte_perm(
        __float_as_uint(log_value.x),
        __float_as_uint(log_value.y),
        0x7632);
    const uint32_t code_one = __vsetleu2(prefixes, 0xc000c000u);
    candidates = __vmaxs2(candidates, code_one);
    candidates = __vmins2(candidates, 0x00070007u);
    const uint32_t high_code = candidates >> 12;
    return (candidates & 0x0fu) | (high_code & 0xf0u);
}

__global__ void classify_kernel(
    const float* input,
    uint8_t* native_codes,
    uint8_t* candidate_codes,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float value = input[index];
    const float2 pair = {value, value};
    const uint32_t native = native_e2m1_pair(native_exp2_pair(pair));
    native_codes[index] = static_cast<uint8_t>(native & 0x0fu);
    candidate_codes[index] = static_cast<uint8_t>(
        affine_floor_pair<0x4017ae14u, 0xc0000000u>(pair) & 0x0fu);
    candidate_codes[count + index] = static_cast<uint8_t>(
        affine_floor_pair<0x4016cb0fu, 0xbffffffeu>(pair) & 0x0fu);
    candidate_codes[2 * count + index] = static_cast<uint8_t>(
        affine_floor_pair<0x4017b9d4u, 0xbffffffeu>(pair) & 0x0fu);
    candidate_codes[3 * count + index] = static_cast<uint8_t>(
        magic_simd_pair(pair) & 0x0fu);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t ordered_key(float value) {
    const uint32_t bits = float_bits(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

float ordered_float(uint32_t key) {
    const uint32_t bits =
        (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
    return bits_float(bits);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            output << '\\';
        }
        output << character;
    }
    return output.str();
}

struct Buffers {
    float* input = nullptr;
    uint8_t* native = nullptr;
    uint8_t* candidates = nullptr;

    ~Buffers() {
        cudaFree(input);
        cudaFree(native);
        cudaFree(candidates);
    }
};

struct CandidateStats {
    size_t mismatches = 0;
    uint64_t absolute_code_error = 0;
    unsigned maximum_code_error = 0;
    std::array<std::array<uint64_t, 8>, 8> transitions{};
};

std::array<CandidateStats, kCandidateCount> compare(
    const std::vector<float>& input
) {
    Buffers buffers;
    const size_t input_bytes = input.size() * sizeof(float);
    const size_t code_bytes = input.size() * sizeof(uint8_t);
    check_cuda(cudaMalloc(&buffers.input, input_bytes), "cudaMalloc(input)");
    check_cuda(cudaMalloc(&buffers.native, code_bytes), "cudaMalloc(native)");
    check_cuda(
        cudaMalloc(&buffers.candidates, kCandidateCount * code_bytes),
        "cudaMalloc(candidates)");
    check_cuda(
        cudaMemcpy(
            buffers.input,
            input.data(),
            input_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(input)");
    constexpr int kThreads = 256;
    const int blocks =
        static_cast<int>((input.size() + kThreads - 1) / kThreads);
    classify_kernel<<<blocks, kThreads>>>(
        buffers.input, buffers.native, buffers.candidates, input.size());
    check_cuda(cudaGetLastError(), "classify_kernel launch");

    std::vector<uint8_t> native(input.size());
    std::vector<uint8_t> candidates(kCandidateCount * input.size());
    check_cuda(
        cudaMemcpy(
            native.data(),
            buffers.native,
            code_bytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(native)");
    check_cuda(
        cudaMemcpy(
            candidates.data(),
            buffers.candidates,
            kCandidateCount * code_bytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(candidates)");

    std::array<CandidateStats, kCandidateCount> stats;
    for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
        for (size_t index = 0; index < input.size(); ++index) {
            const unsigned reference = native[index];
            const unsigned actual =
                candidates[candidate * input.size() + index];
            const unsigned error = reference > actual
                ? reference - actual
                : actual - reference;
            stats[candidate].mismatches += error != 0;
            stats[candidate].absolute_code_error += error;
            stats[candidate].maximum_code_error = std::max(
                stats[candidate].maximum_code_error, error);
            if (reference < 8 && actual < 8) {
                ++stats[candidate].transitions[reference][actual];
            }
        }
    }
    return stats;
}

struct CategoryResult {
    std::string name;
    size_t samples;
    std::array<CandidateStats, kCandidateCount> stats;
};

std::vector<float> representative_inputs() {
    const std::array<float, 7> thresholds = {
        -2.0f,
        -0.41503748297691345f,
        0.32192814350128174f,
        0.80735480785369873f,
        1.3219281435012817f,
        1.8073548078536987f,
        2.3219282627105713f,
    };
    std::vector<float> values = {
        -std::numeric_limits<float>::infinity(),
        -128.0f,
        -32.0f,
        -24.0f,
        -16.0f,
        -8.0f,
        -4.0f,
        -3.0f,
        -2.5f,
        -2.0f,
        -1.5f,
        -1.0f,
        -0.5f,
        0.0f,
        0.5f,
        1.0f,
        1.5f,
        2.0f,
        static_cast<float>(std::log2(6.0)),
        3.0f,
        8.0f,
        std::numeric_limits<float>::infinity(),
    };
    for (float threshold : thresholds) {
        values.push_back(std::nextafter(
            threshold, -std::numeric_limits<float>::infinity()));
        values.push_back(threshold);
        values.push_back(std::nextafter(
            threshold, std::numeric_limits<float>::infinity()));
    }
    return values;
}

std::vector<float> threshold_inputs() {
    const std::array<double, 7> theoretical = {
        -2.0,
        -0.41503749927884381,
        0.32192809488736235,
        0.80735492205760406,
        1.3219280948873624,
        1.8073549220576042,
        2.3219280948873622,
    };
    std::vector<float> values;
    values.reserve(
        theoretical.size() * (2 * kThresholdRadiusUlps + 1));
    for (double boundary : theoretical) {
        const uint32_t center = ordered_key(static_cast<float>(boundary));
        for (int delta = -kThresholdRadiusUlps;
             delta <= kThresholdRadiusUlps;
             ++delta) {
            values.push_back(ordered_float(
                static_cast<uint32_t>(
                    static_cast<int64_t>(center) + delta)));
        }
    }
    return values;
}

std::vector<float> uniform_random_inputs() {
    std::mt19937 generator(20260820u);
    std::uniform_real_distribution<float> distribution(
        -24.0f, static_cast<float>(std::log2(6.0)));
    std::vector<float> values(kRandomCount);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

std::vector<float> replay_like_random_inputs() {
    std::mt19937 generator(20260821u);
    std::normal_distribution<float> tail(0.0f, 3.5f);
    std::uniform_real_distribution<float> anchor(-0.25f, 0.0f);
    const float maximum = static_cast<float>(std::log2(6.0));
    std::vector<float> values(kRandomCount);
    for (float& value : values) {
        value = std::max(-32.0f, maximum + anchor(generator) -
            std::abs(tail(generator)));
    }
    return values;
}

void write_candidate_stats(
    std::ostream& output,
    const CandidateStats& stats,
    size_t samples,
    const std::string& indent
) {
    output << "{\n";
    output << indent << "  \"mismatches\": " << stats.mismatches << ",\n";
    output << indent << "  \"mismatch_rate\": "
           << std::setprecision(17)
           << static_cast<double>(stats.mismatches) / samples << ",\n";
    output << indent << "  \"mean_absolute_code_error\": "
           << static_cast<double>(stats.absolute_code_error) / samples
           << ",\n";
    output << indent << "  \"maximum_code_error\": "
           << stats.maximum_code_error << ",\n";
    output << indent << "  \"nonzero_transitions\": [";
    bool first = true;
    for (int reference = 0; reference < 8; ++reference) {
        for (int actual = 0; actual < 8; ++actual) {
            const uint64_t count = stats.transitions[reference][actual];
            if (count == 0 || reference == actual) {
                continue;
            }
            output << (first ? "\n" : ",\n") << indent
                   << "    {\"native\": " << reference
                   << ", \"candidate\": " << actual
                   << ", \"count\": " << count << "}";
            first = false;
        }
    }
    if (!first) {
        output << "\n" << indent << "  ";
    }
    output << "]\n" << indent << "}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error(
                "usage: calibrate_replay_affine_magic_e2m1 OUTPUT.json");
        }
        const std::filesystem::path output_path(argv[1]);
        if (std::filesystem::exists(output_path)) {
            throw std::runtime_error(
                "refusing to overwrite output: " + output_path.string());
        }

        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        if (properties.major != 10 || properties.minor != 0) {
            throw std::runtime_error("calibration requires SM100");
        }

        std::vector<CategoryResult> results;
        auto run_category = [&results](
            const std::string& name, std::vector<float> values
        ) {
            const size_t samples = values.size();
            results.push_back({name, samples, compare(values)});
        };
        run_category("representative", representative_inputs());
        run_category("threshold_neighborhoods", threshold_inputs());
        run_category("uniform_random", uniform_random_inputs());
        run_category("replay_like_random", replay_like_random_inputs());

        std::ofstream output(output_path, std::ios::out | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot open output: " + output_path.string());
        }
        output << "{\n";
        output << "  \"schema\": \"replay_affine_magic_e2m1_calibration_v1\",\n";
        output << "  \"gpu\": {\"name\": \""
               << json_escape(properties.name) << "\", \"major\": "
               << properties.major << ", \"minor\": "
               << properties.minor << "},\n";
        output << "  \"threshold_radius_ulps\": "
               << kThresholdRadiusUlps << ",\n";
        output << "  \"categories\": [\n";
        for (size_t category = 0; category < results.size(); ++category) {
            const CategoryResult& result = results[category];
            output << "    {\n";
            output << "      \"name\": \"" << result.name << "\",\n";
            output << "      \"samples\": " << result.samples << ",\n";
            output << "      \"candidates\": {\n";
            for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
                output << "        \"" << kCandidateNames[candidate]
                       << "\": ";
                write_candidate_stats(
                    output,
                    result.stats[candidate],
                    result.samples,
                    "        ");
                output << (candidate + 1 == kCandidateCount ? "\n" : ",\n");
            }
            output << "      }\n";
            output << "    }" << (category + 1 == results.size() ? "\n" : ",\n");
        }
        output << "  ]\n";
        output << "}\n";
        output.close();
        std::cout << output_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
