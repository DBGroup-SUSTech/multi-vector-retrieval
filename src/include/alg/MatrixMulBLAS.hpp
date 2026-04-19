//
// Created by 13172 on 2024/2/19.
//

#ifndef VECTORSETSEARCH_MATRIXMULBLAS_HPP
#define VECTORSETSEARCH_MATRIXMULBLAS_HPP

#include <cblas.h>

namespace VectorSetSearch
{
    void VectorInnerProduct(const float* vec1, const float* vec2, const uint32_t& vec_dim, float& result)
    {
        result = cblas_sdot((int)vec_dim, vec1, 1, vec2, 1);
    }

    float EuclideanDistance(const float* vec1, const float* vec2, const uint32_t& vec_dim)
    {
        float result = 0.0f;
        for (uint32_t i = 0; i < vec_dim; i++)
        {
            result += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        }
        return result;
    }

    double EuclideanDistance(const double* vec1, const double* vec2, const uint32_t& vec_dim)
    {
        double result = 0.0f;
        for (uint32_t i = 0; i < vec_dim; i++)
        {
            result += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        }
        return result;
    }


    // (n_vec, vec_dim) * (vec_dim) -> (n_vec)
    void MatrixTimesVector(const float* matrix, const float* vec,
                           const uint32_t& matrix_n_vec, const uint32_t& vec_dim,
                           float* result)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)matrix_n_vec, (int)vec_dim, alpha, matrix,
                    (int)vec_dim, vec, 1, beta,
                    result, 1);
    }

    // (vec_dim) * (vec_dim, n_vec) -> (n_vec)
    void VectorTimesMatrix(const float* vec, const float* matrix,
                           const uint32_t& vec_dim, const uint32_t& matrix_n_vec,
                           float* result)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    (int)vec_dim, (int)matrix_n_vec, alpha, matrix,
                    (int)matrix_n_vec, vec, 1, beta,
                    result, 1);
    }

    // (n_vec1, vec_dim) * (n_vec2, vec_dim) -> (n_vec1, n_vec2)
    void MatrixMultiply(const float* matrix1, const float* matrix2,
                        const uint32_t& n_vec1, const uint32_t& n_vec2,
                        const uint32_t& vec_dim,
                        float* result)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)n_vec1, (int)n_vec2, (int)vec_dim, alpha,
                    matrix1, (int)vec_dim, matrix2, (int)vec_dim, beta,
                    result, (int)n_vec2);
    }

    // (n_vec1, vec_dim) * (vec_dim, n_vec2) -> (n_vec1, n_vec2)
    void MatrixMultiplyNoTrans(const float* matrix1, const float* matrix2,
                               const uint32_t& n_vec1, const uint32_t& n_vec2,
                               const uint32_t& vec_dim,
                               float* result)
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)n_vec1, (int)n_vec2, (int)vec_dim, alpha,
                    matrix1, (int)vec_dim, matrix2, (int)n_vec2, beta,
                    result, (int)n_vec2);
    }

    void MatrixMultiply(const double* matrix1, const double* matrix2,
                        const uint32_t& n_vec1, const uint32_t& n_vec2,
                        const uint32_t& vec_dim,
                        double* result)
    {
        const double alpha = 1.0f;
        const double beta = 0.0f;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)n_vec1, (int)n_vec2, (int)vec_dim, alpha,
                    matrix1, (int)vec_dim, matrix2, (int)vec_dim, beta,
                    result, (int)n_vec2);
    }
}
#endif //VECTORSETSEARCH_MATRIXMULBLAS_HPP
