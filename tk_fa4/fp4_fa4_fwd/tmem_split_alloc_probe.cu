#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

__global__ void tmem_split_alloc_probe(uint32_t* addresses) {
    __shared__ uint32_t main_addr;
    __shared__ uint32_t extra_addr;

    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;\n"
            :
            : "l"(reinterpret_cast<uint64_t>(&main_addr))
            : "memory");
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;\n"
            :
            : "l"(reinterpret_cast<uint64_t>(&extra_addr))
            : "memory");
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }

    asm volatile("tcgen05.fence::before_thread_sync;\n");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n");

    if (threadIdx.x == 0) {
        addresses[0] = main_addr;
        addresses[1] = extra_addr;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;\n"
            :
            : "r"(extra_addr));
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;\n"
            :
            : "r"(main_addr));
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

} // namespace

int main() {
    uint32_t* addresses = nullptr;
    if (!check(cudaMallocManaged(&addresses, 2 * sizeof(uint32_t)), "cudaMallocManaged")) {
        return 1;
    }

    addresses[0] = 0xffffffffu;
    addresses[1] = 0xffffffffu;
    tmem_split_alloc_probe<<<1, 32>>>(addresses);
    if (!check(cudaGetLastError(), "kernel launch") ||
        !check(cudaDeviceSynchronize(), "kernel execution")) {
        cudaFree(addresses);
        return 2;
    }

    std::printf(
        "main_addr=%u extra_addr=%u distinct=%d\n",
        addresses[0],
        addresses[1],
        addresses[0] != addresses[1]);
    const bool valid = addresses[0] != 0xffffffffu &&
                       addresses[1] != 0xffffffffu &&
                       addresses[0] != addresses[1];
    cudaFree(addresses);
    return valid ? 0 : 3;
}
