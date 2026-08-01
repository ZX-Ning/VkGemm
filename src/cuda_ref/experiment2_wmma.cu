/**
 * CUDA WMMA counterpart of shaders/gemm/experiment2.slang.
 *
 * Requirements:
 *   - M, N and K are multiples of 64.
 *   - A, B and C use column-major storage.
 *   - A and B are 16-byte aligned.
 *   - Launch with block=(256, 1, 1) and grid=(M / 64, N / 64, 1).
 *
 * One block computes a 64x64 output tile. Its eight warps are arranged as a
 * 2x4 grid. Each warp owns a 32x16 output region represented by two vertically
 * adjacent 16x16 WMMA accumulators.
 */

#ifdef CUDA_REF

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <mma.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cuda_ref.hpp"
#endif

namespace {

namespace wmma = nvcuda::wmma;

constexpr int WMMA_TILE_SIZE = 16;
constexpr int BLOCK_TILE_SIZE = 64;
constexpr int SHARED_SKEW = 8;
constexpr int SHARED_STRIDE = BLOCK_TILE_SIZE + SHARED_SKEW;
constexpr int SHARED_ELEMENT_COUNT = BLOCK_TILE_SIZE * SHARED_STRIDE;
constexpr int LOAD_HALF_COUNT = 8;
constexpr int BLOCK_THREAD_COUNT = 256;
constexpr int PANEL_VECTOR_COUNT =
    BLOCK_TILE_SIZE * BLOCK_TILE_SIZE / LOAD_HALF_COUNT;

using FragmentA = wmma::fragment<
    wmma::matrix_a,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    __half,
    wmma::col_major>;
using FragmentB = wmma::fragment<
    wmma::matrix_b,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    __half,
    wmma::col_major>;
using FragmentAccumulator = wmma::fragment<
    wmma::accumulator,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    float>;

__device__ __forceinline__ void store_packed_half2(
    std::uint32_t packed,
    __half* destination
) {
    destination[0] =
        __ushort_as_half(static_cast<unsigned short>(packed));
    destination[1] = __ushort_as_half(
        static_cast<unsigned short>(packed >> 16)
    );
}

/**
 * Unpack one 16-byte vector into eight consecutive shared-memory half values.
 * Keeping these as scalar stores mirrors experiment2.slang's load path.
 */
__device__ __forceinline__ void store_packed_half8(
    const uint4& packed,
    __half* destination
) {
    store_packed_half2(packed.x, destination);
    store_packed_half2(packed.y, destination + 2);
    store_packed_half2(packed.z, destination + 4);
    store_packed_half2(packed.w, destination + 6);
}

/**
 * Stage one logical 64x64 A panel and one logical 64x64 B panel in shared
 * memory. Each thread moves two 16-byte vectors from each input.
 */
__device__ __forceinline__ void load_input_panels_to_shared(
    const uint4* __restrict__ mat_a,
    const uint4* __restrict__ mat_b,
    __half* shared_a,
    __half* shared_b,
    int row_offset,
    int col_offset,
    int k_offset,
    int m,
    int k
) {
    for (int vector_index = static_cast<int>(threadIdx.x);
         vector_index < PANEL_VECTOR_COUNT;
         vector_index += BLOCK_THREAD_COUNT) {
        const int move = vector_index * LOAD_HALF_COUNT;
        const int move_row = move & (BLOCK_TILE_SIZE - 1);
        const int move_col = move >> 6;

        const int a_index =
            (k_offset + move_col) * m + row_offset + move_row;
        const int b_index =
            (col_offset + move_col) * k + k_offset + move_row;

        // Add eight padding elements for each preceding logical column.
        const int shared_index = move + (move_col << 3);

        const uint4 a_values = mat_a[a_index / LOAD_HALF_COUNT];
        const uint4 b_values = mat_b[b_index / LOAD_HALF_COUNT];

        store_packed_half8(a_values, shared_a + shared_index);
        store_packed_half8(b_values, shared_b + shared_index);
    }
}

__device__ __forceinline__ void load_accumulator(
    FragmentAccumulator& accumulator,
    const float* mat_c,
    int global_row,
    int global_col,
    int m
) {
    wmma::load_matrix_sync(
        accumulator,
        mat_c + global_col * m + global_row,
        m,
        wmma::mem_col_major
    );
}

__device__ __forceinline__ void store_accumulator(
    const FragmentAccumulator& accumulator,
    float* mat_c,
    int global_row,
    int global_col,
    int m
) {
    wmma::store_matrix_sync(
        mat_c + global_col * m + global_row,
        accumulator,
        m,
        wmma::mem_col_major
    );
}

}  // namespace

extern "C" __global__ __launch_bounds__(BLOCK_THREAD_COUNT) void gemm_experiment2_wmma(
    const __half* __restrict__ mat_a,
    const __half* __restrict__ mat_b,
    float* __restrict__ mat_c,
    int m,
    int n,
    int k
) {
    // A stride of 72 half values is 144 bytes, matching experiment2.slang's
    // skewed shared-memory layout. The explicit alignment also satisfies the
    // WMMA load alignment requirement for every 16x16 fragment origin.
    __shared__ __align__(32) __half shared_a[SHARED_ELEMENT_COUNT];
    __shared__ __align__(32) __half shared_b[SHARED_ELEMENT_COUNT];

    const int warp_index = static_cast<int>(threadIdx.x) >> 5;
    const int warp_row = warp_index & 1;
    const int warp_col = warp_index >> 1;

    const int row_offset = static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE;
    const int col_offset = static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE;
    const int shared_row = warp_row * WMMA_TILE_SIZE * 2;
    const int shared_col = warp_col * WMMA_TILE_SIZE;
    const int global_row = row_offset + shared_row;
    const int global_col = col_offset + shared_col;

    FragmentAccumulator accumulator_top;
    FragmentAccumulator accumulator_bottom;
    load_accumulator(
        accumulator_top,
        mat_c,
        global_row,
        global_col,
        m
    );
    load_accumulator(
        accumulator_bottom,
        mat_c,
        global_row + WMMA_TILE_SIZE,
        global_col,
        m
    );

    const auto* mat_a_vectors = reinterpret_cast<const uint4*>(mat_a);
    const auto* mat_b_vectors = reinterpret_cast<const uint4*>(mat_b);

    for (int k_offset = 0;
         k_offset < k;
         k_offset += BLOCK_TILE_SIZE) {
        load_input_panels_to_shared(
            mat_a_vectors,
            mat_b_vectors,
            shared_a,
            shared_b,
            row_offset,
            col_offset,
            k_offset,
            m,
            k
        );
        __syncthreads();

#pragma unroll
        for (int shared_k = 0;
             shared_k < BLOCK_TILE_SIZE;
             shared_k += WMMA_TILE_SIZE) {
            FragmentA a_top;
            FragmentA a_bottom;
            FragmentB b;

            wmma::load_matrix_sync(
                a_top,
                shared_a + shared_k * SHARED_STRIDE + shared_row,
                SHARED_STRIDE
            );
            wmma::load_matrix_sync(
                a_bottom,
                shared_a + shared_k * SHARED_STRIDE +
                    shared_row + WMMA_TILE_SIZE,
                SHARED_STRIDE
            );
            wmma::load_matrix_sync(
                b,
                shared_b + shared_col * SHARED_STRIDE + shared_k,
                SHARED_STRIDE
            );

            wmma::mma_sync(
                accumulator_top,
                a_top,
                b,
                accumulator_top
            );
            wmma::mma_sync(
                accumulator_bottom,
                a_bottom,
                b,
                accumulator_bottom
            );
        }

        __syncthreads();
    }

    store_accumulator(
        accumulator_top,
        mat_c,
        global_row,
        global_col,
        m
    );
    store_accumulator(
        accumulator_bottom,
        mat_c,
        global_row + WMMA_TILE_SIZE,
        global_col,
        m
    );

    // N is part of the public GEMM shape even though this no-edge fast path
    // only needs it to define grid.y at launch time.
    (void)n;
}

#ifdef CUDA_REF

#define CUDA_CHECK(expression)                                          \
    do {                                                                \
        const cudaError_t error = (expression);                         \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)    \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (false)

size_t runCudaExperiment2(
    uint32_t matSize,
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3
) {
    if (matSize % BLOCK_TILE_SIZE != 0) {
        fmt::println(
            stderr,
            "CUDA Experiment2 requires matrix size to be a multiple of {}",
            BLOCK_TILE_SIZE
        );
        std::exit(EXIT_FAILURE);
    }

    static_assert(sizeof(Eigen::half) == sizeof(__half));

    const size_t elementCount =
        static_cast<size_t>(matSize) * matSize;
    const size_t inputBytes = elementCount * sizeof(__half);
    const size_t outputBytes = elementCount * sizeof(float);

    __half* deviceA = nullptr;
    __half* deviceB = nullptr;
    float* deviceC = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&deviceA),
        inputBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&deviceB),
        inputBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&deviceC),
        outputBytes
    ));

    CUDA_CHECK(cudaMemcpy(
        deviceA,
        mat1.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        deviceB,
        mat2.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        deviceC,
        mat3.data(),
        outputBytes,
        cudaMemcpyHostToDevice
    ));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 block(BLOCK_THREAD_COUNT, 1, 1);
    const dim3 grid(
        matSize / BLOCK_TILE_SIZE,
        matSize / BLOCK_TILE_SIZE,
        1
    );

    fmt::println(
        "Begin CUDA Experiment2 Compute. Matrix size: {}x{}",
        matSize,
        matSize
    );

    CUDA_CHECK(cudaEventRecord(start));
    gemm_experiment2_wmma<<<grid, block>>>(
        deviceA,
        deviceB,
        deviceC,
        static_cast<int>(matSize),
        static_cast<int>(matSize),
        static_cast<int>(matSize)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMilliseconds,
        start,
        stop
    ));
    const size_t timeUsed =
        static_cast<size_t>(elapsedMilliseconds + 0.5f);

    fmt::println(
        "CUDA Experiment2 Compute Done. Time: {:.3f} ms",
        elapsedMilliseconds
    );

    CUDA_CHECK(cudaMemcpy(
        mat3.data(),
        deviceC,
        outputBytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));

    return timeUsed;
}

#undef CUDA_CHECK

#endif
