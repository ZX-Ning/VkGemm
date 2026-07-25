#pragma once
#ifdef CUBLAS
#include <Eigen/Dense>
size_t runCuBlas(
    uint32_t matSize,
    Eigen::MatrixX<Eigen::half>& mat1,
    Eigen::MatrixX<Eigen::half>& mat2,
    Eigen::MatrixXf& mat3
);
#endif
