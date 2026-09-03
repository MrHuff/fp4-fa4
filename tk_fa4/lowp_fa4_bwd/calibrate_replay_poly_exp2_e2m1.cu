#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kCandidateCount = 2;
constexpr int kBoundaryCount = 7;
constexpr int kBoundaryRadiusUlps = 65536;
constexpr size_t kBroadCount = 1 << 20;
constexpr int kReplayGroupsPerSetting = 8192;
constexpr int kReplayGroupWidth = 32;

constexpr std::array<const char*, kCandidateCount> kCandidateNames = {
    "degree1",
    "degree2",
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

// Literal copies of tk_exp2_alu_degree{1,2}_f32x2.  Keeping these as inline
// PTX makes the payload calibration exercise the same range reduction and
// packed ALU instructions as the CuTe replay candidate.
__device__ __forceinline__ float2 degree1_exp2_pair(float2 value) {
    float2 output;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5;\n\t"
        ".reg .b64 l1, l2, l3, l4, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f5, 0f3F317218;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l4, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=f"(output.x), "=f"(output.y)
        : "f"(value.x), "f"(value.y));
    return output;
}

__device__ __forceinline__ float2 degree2_exp2_pair(float2 value) {
    float2 output;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f6, 0f3EA903CA;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F2A70E4;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l5, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
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
    const float2 pair = {input[index], input[index]};
    native_codes[index] = static_cast<uint8_t>(
        native_e2m1_pair(native_exp2_pair(pair)) & 0x0fu);
    candidate_codes[index] = static_cast<uint8_t>(
        native_e2m1_pair(degree1_exp2_pair(pair)) & 0x0fu);
    candidate_codes[count + index] = static_cast<uint8_t>(
        native_e2m1_pair(degree2_exp2_pair(pair)) & 0x0fu);
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

struct Category {
    std::string name;
    std::string construction;
    std::vector<float> input;
    std::array<CandidateStats, kCandidateCount> stats;
};

std::vector<float> boundary_inputs() {
    const std::array<double, kBoundaryCount> boundaries = {
        -2.0,
        std::log2(0.75),
        std::log2(1.25),
        std::log2(1.75),
        std::log2(2.5),
        std::log2(3.5),
        std::log2(5.0),
    };
    const size_t span = static_cast<size_t>(2 * kBoundaryRadiusUlps + 1);
    std::vector<float> input(kBoundaryCount * span);
    for (int boundary = 0; boundary < kBoundaryCount; ++boundary) {
        const uint32_t center_key = ordered_key(
            static_cast<float>(boundaries[boundary]));
        for (int offset = -kBoundaryRadiusUlps;
             offset <= kBoundaryRadiusUlps;
             ++offset) {
            input[static_cast<size_t>(boundary) * span
                  + offset + kBoundaryRadiusUlps] =
                ordered_float(center_key + offset);
        }
    }
    return input;
}

std::vector<float> broad_inputs() {
    std::vector<float> input(kBroadCount + 6);
    uint32_t state = 0x20260820u;
    for (size_t index = 0; index < kBroadCount; ++index) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float unit =
            static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
        input[index] = -32.0f + unit * (32.0f + std::log2(6.0f));
    }
    input[kBroadCount + 0] = -std::numeric_limits<float>::infinity();
    input[kBroadCount + 1] = -126.0f;
    input[kBroadCount + 2] = -32.0f;
    input[kBroadCount + 3] = -2.0f;
    input[kBroadCount + 4] = std::log2(6.0f);
    input[kBroadCount + 5] = std::numeric_limits<float>::infinity();
    return input;
}

std::vector<float> replay_inputs(float standard_deviation, uint32_t seed) {
    // The replay log is score_log2 + log2(6) - group_exponent.  The forward
    // exponent is floor(group_max_log2 + log2(4/3)); groups contain the 32
    // scores in one physical N32 scale group.  Varying valid counts models
    // diagonal causal groups without synthesizing invalid -inf payloads.
    constexpr std::array<int, 6> valid_counts = {1, 2, 4, 8, 16, 32};
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(
        0.0f, standard_deviation);
    std::vector<float> input;
    input.reserve(
        static_cast<size_t>(kReplayGroupsPerSetting)
        * valid_counts.size() * kReplayGroupWidth);
    const float log2_six = std::log2(6.0f);
    const float exponent_bias = std::log2(4.0f / 3.0f);
    for (const int valid_count : valid_counts) {
        for (int group = 0; group < kReplayGroupsPerSetting; ++group) {
            std::array<float, kReplayGroupWidth> scores;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int lane = 0; lane < valid_count; ++lane) {
                scores[lane] = distribution(generator);
                maximum = std::max(maximum, scores[lane]);
            }
            const float exponent = std::floor(maximum + exponent_bias);
            for (int lane = 0; lane < valid_count; ++lane) {
                input.push_back(scores[lane] + log2_six - exponent);
            }
        }
    }
    return input;
}

void write_stats(
    std::ostream& output,
    const CandidateStats& stats,
    size_t samples
) {
    output << "{\"mismatches\": " << stats.mismatches
           << ", \"mismatch_rate\": "
           << static_cast<double>(stats.mismatches) / samples
           << ", \"absolute_code_error\": "
           << stats.absolute_code_error
           << ", \"maximum_code_error\": "
           << stats.maximum_code_error
           << ", \"off_diagonal_transitions\": [";
    bool first = true;
    for (int reference = 0; reference < 8; ++reference) {
        for (int actual = 0; actual < 8; ++actual) {
            if (reference == actual ||
                stats.transitions[reference][actual] == 0) {
                continue;
            }
            output << (first ? "" : ", ")
                   << "{\"native\": " << reference
                   << ", \"candidate\": " << actual
                   << ", \"count\": "
                   << stats.transitions[reference][actual] << "}";
            first = false;
        }
    }
    output << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: " << argv[0] << " OUTPUT_JSON\n";
            return 64;
        }
        const std::string output_path = argv[1];
        std::ifstream existing(output_path);
        if (existing.good()) {
            throw std::runtime_error("refusing to overwrite " + output_path);
        }

        std::vector<Category> categories;
        categories.push_back({
            "boundary_ulp_windows",
            "seven E2M1 midpoints +/-65536 ordered FP32 ULPs",
            boundary_inputs(),
            {},
        });
        categories.push_back({
            "broad_uniform",
            "deterministic uniform log2 inputs in [-32,log2(6)]",
            broad_inputs(),
            {},
        });
        for (int setting = 0; setting < 4; ++setting) {
            const float sigma = std::array<float, 4>{0.5f, 1.0f, 2.0f, 4.0f}[
                setting];
            categories.push_back({
                "replay_gaussian_sigma_" + std::to_string(sigma),
                "Gaussian score_log2; causal valid counts 1,2,4,8,16,32; "
                "forward 32-score physical-group exponent rule",
                replay_inputs(sigma, 0x20260820u + setting),
                {},
            });
        }
        for (auto& category : categories) {
            category.stats = compare(category.input);
        }

        int device = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        std::ofstream output(output_path, std::ios::out | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot create output: " + output_path + ": "
                + std::strerror(errno));
        }
        output << std::setprecision(17);
        output << "{\n";
        output << "  \"schema\": \"replay_poly_exp2_e2m1_calibration_v1\",\n";
        output << "  \"gpu\": {\"name\": \"" << properties.name
               << "\", \"major\": " << properties.major
               << ", \"minor\": " << properties.minor << "},\n";
        output << "  \"candidate_order\": [\"degree1\", \"degree2\"],\n";
        output << "  \"categories\": [\n";
        for (size_t category_index = 0;
             category_index < categories.size();
             ++category_index) {
            const auto& category = categories[category_index];
            output << "    {\"name\": \"" << category.name
                   << "\", \"construction\": \""
                   << category.construction
                   << "\", \"samples\": " << category.input.size()
                   << ", \"candidates\": {";
            for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
                output << (candidate == 0 ? "" : ", ") << "\""
                       << kCandidateNames[candidate] << "\": ";
                write_stats(
                    output,
                    category.stats[candidate],
                    category.input.size());
            }
            output << "}}"
                   << (category_index + 1 == categories.size() ? "\n" : ",\n");
        }
        output << "  ]\n";
        output << "}\n";
        output.close();
        if (!output) {
            throw std::runtime_error("failed writing output: " + output_path);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "calibration failed: " << error.what() << "\n";
        return 1;
    }
}
