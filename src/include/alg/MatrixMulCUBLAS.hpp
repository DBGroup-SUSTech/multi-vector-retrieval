//
// Created by bianzheng on 2024/7/23.
//

#ifndef VECTORSETSEARCH_MATRIXMULCUBLAS_HPP
#define VECTORSETSEARCH_MATRIXMULCUBLAS_HPP

#include <cublas_v2.h>

#include "include/util/CUDAUtil.hpp"

namespace VectorSetSearch
{
    void VectorInnerProduct(cublasHandle_t handle, const float* vec1, const float* vec2, const uint32_t& vec_dim,
                            float& result)
    {
        cublasCheckErrors(cublasSdot(handle, (int) vec_dim, vec1, 1, vec2, 1, &result));
    }

    void MatrixTimesVector(cublasHandle_t handle, const float* matrix, const float* vec,
                           const uint32_t& matrix_n_vec, const uint32_t& vec_dim,
                           float* result)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasCheckErrors(cublasSgemv(handle, CUBLAS_OP_T,
            (int) matrix_n_vec, (int) vec_dim, &alpha, matrix,
            (int) vec_dim, vec, 1, &beta,
            result, 1));
    }

    void MatrixMultiply(cublasHandle_t handle, const float* matrix1, const float* matrix2,
                        const uint32_t n_vec1, const uint32_t n_vec2,
                        const uint32_t vec_dim,
                        float* result)
    {
        // matrix1: n_vec1 * vec_dim, matrix2: n_vec2 * vec_dim -> result: n_vec1 * n_vec2
        const float alpha = 1.0f;
        const float beta = 0.0f;

        //  cudaPointerAttributes attr;
        //  cudaPointerGetAttributes(&attr, matrix1);  // 获取指针属性
        //  assert(attr.type == cudaMemoryTypeDevice);
        //  cudaPointerGetAttributes(&attr, matrix2);  // 获取指针属性
        //  assert(attr.type == cudaMemoryTypeDevice);
        //  cudaPointerGetAttributes(&attr, result);  // 获取指针属性
        //  assert(attr.type == cudaMemoryTypeDevice);

        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)n_vec2, (int)n_vec1, (int)vec_dim, &alpha,
                    matrix2, (int)vec_dim,
                    matrix1, (int)vec_dim,
                    &beta, result, (int)n_vec2);

        //  cublasCheckErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        //                                (int) n_vec1, (int) n_vec2, (int) vec_dim, &alpha,
        //                                matrix1, (int) vec_dim, matrix2, (int) vec_dim, &beta,
        //                                result, (int) n_vec2));
    }

    void MatrixMultiplyMultiPrecision(cublasHandle_t handle, const float* matrix1, const float* matrix2,
                                      const uint32_t& n_vec1, const uint32_t& n_vec2,
                                      const uint32_t& vec_dim,
                                      float* result)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasCheckErrors(cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)n_vec2, (int)n_vec1, (int)vec_dim,
            &alpha,
            matrix2, CUDA_R_32F, (int)vec_dim,
            matrix1, CUDA_R_32F, (int)vec_dim,
            &beta,
            result, CUDA_R_32F, (int)n_vec2,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT));
    }
}
#endif //VECTORSETSEARCH_MATRIXMULCUBLAS_HPP
