#ifdef CUBLAS

#include "cublas.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <print>

#include "../utils.hpp"

#define CUDA_CHECK(expr)                                                \
    do {                                                                \
        const cudaError_t error = (expr);                               \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)    \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (false)

#define CUBLAS_CHECK(expr)                                              \
    do {                                                                \
        const cublasStatus_t status = (expr);                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                          \
            std::cerr << "cuBLAS error: " << status                     \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (false)

void runCuBlas(
    uint32_t matSize,
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3
) {
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bufSize = matSize * matSize;

    CUDA_CHECK(cudaMalloc(&d_A, bufSize * 2));
    CUDA_CHECK(cudaMalloc(&d_B, bufSize * 2));
    CUDA_CHECK(cudaMalloc(&d_C, bufSize * 4));

    std::println("Begin cuBlas Compute");

    CUDA_CHECK(cudaMemcpy(
        d_A,
        mat1.data(),
        bufSize * 2,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        mat2.data(),
        bufSize * 2,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_C,
        mat3.data(),
        bufSize * 4,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
    size_t time1 = getTimestampMs();

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 1.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,

        CUBLAS_OP_N,
        CUBLAS_OP_N,

        matSize,
        matSize,
        matSize,

        &alpha,

        d_A,
        CUDA_R_16F,
        matSize,

        d_B,
        CUDA_R_16F,
        matSize,

        &beta,

        d_C,
        CUDA_R_32F,
        matSize,

        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
    std::println("cuBlas Compute Done. Time: {} ms", getTimestampMs() - time1);

    CUDA_CHECK(cudaMemcpy(
        mat3.data(),
        d_C,
        bufSize * 4,
        cudaMemcpyDeviceToHost
    ));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

#endif
