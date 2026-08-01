/**
 * CUDA WMMA counterpart of shaders/gemm/coopmat_128x128_16acc.slang.
 *
 * Requirements:
 *   - M and N are multiples of 128; K is a multiple of 32.
 *   - A, B and C use column-major storage.
 *   - A and B are 16-byte aligned.
 *   - Launch with block=(128, 1, 1) and grid=(M / 128, N / 128, 1).
 *
 * One block computes a 128x128 output tile. Its four warps form a 2x2
 * grid, and each warp owns a 64x64 region represented by sixteen 16x16
 * WMMA accumulators.
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
constexpr int BLOCK_TILE_SIZE = 128;
constexpr int K_TILE_SIZE = 32;
constexpr int SHARED_SKEW = 8;
constexpr int SHARED_STRIDE = K_TILE_SIZE + SHARED_SKEW;
constexpr int SHARED_ELEMENT_COUNT = BLOCK_TILE_SIZE * SHARED_STRIDE;
constexpr int LOAD_HALF_COUNT = 8;
constexpr int LOAD_WORD_COUNT = LOAD_HALF_COUNT / 2;
constexpr int BLOCK_THREAD_COUNT = 128;
constexpr int PANEL_VECTOR_COUNT =
    BLOCK_TILE_SIZE * K_TILE_SIZE / LOAD_HALF_COUNT;
constexpr int ACC_TILES_PER_AXIS = 4;
constexpr int WARP_TILE_SIZE = WMMA_TILE_SIZE * ACC_TILES_PER_AXIS;

using FragmentA = wmma::fragment<
    wmma::matrix_a,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    WMMA_TILE_SIZE,
    __half,
    wmma::row_major>;
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

__device__ __forceinline__ __half unpack_low_half(std::uint32_t packed) {
    return __ushort_as_half(static_cast<unsigned short>(packed));
}

__device__ __forceinline__ __half unpack_high_half(std::uint32_t packed) {
    return __ushort_as_half(static_cast<unsigned short>(packed >> 16));
}

/**
 * Scatter eight adjacent global A values into eight shared-memory rows.
 *
 * Global A is column-major, so a uint4 contains eight rows at one K. The
 * shared A panel is row-major with K as its contiguous dimension.
 */
__device__ __forceinline__ void store_a_vector_to_shared(
    const uint4& packed,
    __half* shared_a,
    int shared_row,
    int shared_k
) {
    const std::uint32_t words[LOAD_WORD_COUNT] = {
        packed.x,
        packed.y,
        packed.z,
        packed.w,
    };

#pragma unroll
    for (int word = 0; word < LOAD_WORD_COUNT; ++word) {
        const int row = shared_row + word * 2;
        shared_a[row * SHARED_STRIDE + shared_k] =
            unpack_low_half(words[word]);
        shared_a[(row + 1) * SHARED_STRIDE + shared_k] =
            unpack_high_half(words[word]);
    }
}

/**
 * Store eight adjacent global B values into the contiguous K dimension of
 * one shared-memory column.
 */
__device__ __forceinline__ void store_b_vector_to_shared(
    const uint4& packed,
    __half* shared_b,
    int shared_col,
    int shared_k
) {
    const std::uint32_t words[LOAD_WORD_COUNT] = {
        packed.x,
        packed.y,
        packed.z,
        packed.w,
    };
    __half* destination =
        shared_b + shared_col * SHARED_STRIDE + shared_k;

#pragma unroll
    for (int word = 0; word < LOAD_WORD_COUNT; ++word) {
        destination[word * 2] = unpack_low_half(words[word]);
        destination[word * 2 + 1] = unpack_high_half(words[word]);
    }
}

/**
 * Stage one 128x32 A panel and one 32x128 B panel in shared memory. Each
 * thread moves four 16-byte vectors from A and four from B.
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
        const int a_row_group =
            vector_index % (BLOCK_TILE_SIZE / LOAD_HALF_COUNT);
        const int a_row = a_row_group * LOAD_HALF_COUNT;
        const int a_k =
            vector_index / (BLOCK_TILE_SIZE / LOAD_HALF_COUNT);
        const int a_index =
            (k_offset + a_k) * m + row_offset + a_row;

        const int b_k_group =
            vector_index % (K_TILE_SIZE / LOAD_HALF_COUNT);
        const int b_k = b_k_group * LOAD_HALF_COUNT;
        const int b_col =
            vector_index / (K_TILE_SIZE / LOAD_HALF_COUNT);
        const int b_index =
            (col_offset + b_col) * k + k_offset + b_k;

        const uint4 a_values = mat_a[a_index / LOAD_HALF_COUNT];
        const uint4 b_values = mat_b[b_index / LOAD_HALF_COUNT];

        store_a_vector_to_shared(
            a_values,
            shared_a,
            a_row,
            a_k
        );
        store_b_vector_to_shared(
            b_values,
            shared_b,
            b_col,
            b_k
        );
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

extern "C" __global__ __launch_bounds__(BLOCK_THREAD_COUNT) void
gemm_coopmat_128x128_16acc_wmma(
    const __half* __restrict__ mat_a,
    const __half* __restrict__ mat_b,
    float* __restrict__ mat_c,
    int m,
    int n,
    int k
) {
    // The logical shared shape is 128x32. The eight-half skew gives a
    // physical row/column stride of 40 while retaining 32-byte alignment at
    // every 16x16 WMMA fragment origin.
    __shared__ __align__(32) __half shared_a[SHARED_ELEMENT_COUNT];
    __shared__ __align__(32) __half shared_b[SHARED_ELEMENT_COUNT];

    const int warp_index = static_cast<int>(threadIdx.x) >> 5;
    const int warp_row = warp_index & 1;
    const int warp_col = warp_index >> 1;

    const int row_offset = static_cast<int>(blockIdx.x) * BLOCK_TILE_SIZE;
    const int col_offset = static_cast<int>(blockIdx.y) * BLOCK_TILE_SIZE;
    const int shared_row = warp_row * WARP_TILE_SIZE;
    const int shared_col = warp_col * WARP_TILE_SIZE;
    const int global_row = row_offset + shared_row;
    const int global_col = col_offset + shared_col;

    FragmentAccumulator
        accumulators[ACC_TILES_PER_AXIS][ACC_TILES_PER_AXIS];

#pragma unroll
    for (int acc_col = 0; acc_col < ACC_TILES_PER_AXIS; ++acc_col) {
#pragma unroll
        for (int acc_row = 0; acc_row < ACC_TILES_PER_AXIS; ++acc_row) {
            load_accumulator(
                accumulators[acc_col][acc_row],
                mat_c,
                global_row + acc_row * WMMA_TILE_SIZE,
                global_col + acc_col * WMMA_TILE_SIZE,
                m
            );
        }
    }

    const auto* mat_a_vectors = reinterpret_cast<const uint4*>(mat_a);
    const auto* mat_b_vectors = reinterpret_cast<const uint4*>(mat_b);

    for (int k_offset = 0; k_offset < k; k_offset += K_TILE_SIZE) {
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
             shared_k < K_TILE_SIZE;
             shared_k += WMMA_TILE_SIZE) {
            FragmentA a_fragments[ACC_TILES_PER_AXIS];
            FragmentB b_fragments[ACC_TILES_PER_AXIS];

#pragma unroll
            for (int fragment = 0;
                 fragment < ACC_TILES_PER_AXIS;
                 ++fragment) {
                wmma::load_matrix_sync(
                    a_fragments[fragment],
                    shared_a +
                        (shared_row + fragment * WMMA_TILE_SIZE) *
                            SHARED_STRIDE +
                        shared_k,
                    SHARED_STRIDE
                );
                wmma::load_matrix_sync(
                    b_fragments[fragment],
                    shared_b +
                        (shared_col + fragment * WMMA_TILE_SIZE) *
                            SHARED_STRIDE +
                        shared_k,
                    SHARED_STRIDE
                );
            }

#pragma unroll
            for (int acc_col = 0;
                 acc_col < ACC_TILES_PER_AXIS;
                 ++acc_col) {
#pragma unroll
                for (int acc_row = 0;
                     acc_row < ACC_TILES_PER_AXIS;
                     ++acc_row) {
                    wmma::mma_sync(
                        accumulators[acc_col][acc_row],
                        a_fragments[acc_row],
                        b_fragments[acc_col],
                        accumulators[acc_col][acc_row]
                    );
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int acc_col = 0; acc_col < ACC_TILES_PER_AXIS; ++acc_col) {
#pragma unroll
        for (int acc_row = 0; acc_row < ACC_TILES_PER_AXIS; ++acc_row) {
            store_accumulator(
                accumulators[acc_col][acc_row],
                mat_c,
                global_row + acc_row * WMMA_TILE_SIZE,
                global_col + acc_col * WMMA_TILE_SIZE,
                m
            );
        }
    }

    // N is represented by grid.y in this no-edge fast path.
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

size_t runCuda128x12816Acc(
    uint32_t matSize,
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3
) {
    if (matSize % BLOCK_TILE_SIZE != 0) {
        fmt::println(
            stderr,
            "CUDA 128x128 16ACC requires matrix size to be a multiple of {}",
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
        "Begin CUDA 128x128 16ACC Compute. Matrix size: {}x{}",
        matSize,
        matSize
    );

    CUDA_CHECK(cudaEventRecord(start));
    gemm_coopmat_128x128_16acc_wmma<<<grid, block>>>(
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
        "CUDA 128x128 16ACC Compute Done. Time: {:.3f} ms",
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
