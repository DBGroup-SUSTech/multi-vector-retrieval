//
// Created by Administrator on 2025/9/5.
//

#ifndef GENERATEVQSCORECANDIDATEPP_HPP
#define GENERATEVQSCORECANDIDATEPP_HPP

#include <raft/matrix/detail/select_k-inl.cuh>
#include <cublasLt.h>
#include <cstdint>
#include "include/alg/igp_gpu_pp/DataStructPP.hpp"

namespace VectorSetSearch
{

class GenerateVQScoreCandidatePP
{
 public:
  const ItemVecInfoPP* item_vec_info_;
  const VQInfoPP* vq_info_;
  const ResourcePP* resource_;

  uint32_t threadID_;
  uint32_t query_n_vec_, nprobe_;

  // for matrix multiplication then add 1
  cublasLtHandle_t cublaslt_handle_;
  float* bias_gpu_;
  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t mat_a_desc_, mat_b_desc_, mat_c_desc_;

  float* vq_score_table_gpu_; // query_n_vec * n_centroid
  float* filter_cent_score_gpu_; // query_n_vec * nprobe_
  uint32_t* filter_centID_l_gpu_; // query_n_vec * nprobe_

  GenerateVQScoreCandidatePP() = default;

  GenerateVQScoreCandidatePP(const ItemVecInfoPP* item_vec_info, const VQInfoPP* vq_info,
                                const ResourcePP* resource,
                                const uint32_t threadID,
                                const uint32_t query_n_vec, const uint32_t nprobe)
  {
      this->item_vec_info_ = item_vec_info;
      this->vq_info_ = vq_info;
      this->resource_ = resource;

      this->threadID_ = threadID;
      this->query_n_vec_ = query_n_vec;
      this->nprobe_ = nprobe;

      cublasCheckErrors(cublasLtCreate(&cublaslt_handle_));
      CHECK(cudaMalloc(&bias_gpu_, vq_info_->n_centroid_ * sizeof(float)));
      thrust::fill(thrust::device, bias_gpu_, bias_gpu_ + vq_info_->n_centroid_, 1.0f);
      cublasCheckErrors(cublasLtMatmulDescCreate(&matmul_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      cublasCheckErrors(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
      cublasCheckErrors(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_gpu_, sizeof(bias_gpu_)));
      // A = query_gpu (m=query_n_vec_, k=vec_dim_), op = N
      // B = centroid_l_gpu (k=vec_dim_, n=n_centroid_), op = T
      // C = vq_score_table_gpu (m=query_n_vec_, n=n_centroid_)
      const int m = query_n_vec_;
      const int n = vq_info_->n_centroid_;
      const int k = vq_info_->vec_dim_;

      // Matrix A (query_gpu): m x k, row-major storage
      cublasCheckErrors(cublasLtMatrixLayoutCreate(&mat_a_desc_, CUDA_R_32F, m, k, k));
      // Matrix B (centroid_l_gpu): n x k, row-major storage
      cublasCheckErrors(cublasLtMatrixLayoutCreate(&mat_b_desc_, CUDA_R_32F, n, k, k));
      // Matrix C (result): m x n, row-major storage
      cublasCheckErrors(cublasLtMatrixLayoutCreate(&mat_c_desc_, CUDA_R_32F, m, n, n));

      cublasOperation_t trans_a = CUBLAS_OP_N;
      cublasOperation_t trans_b = CUBLAS_OP_T;
      cublasCheckErrors(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
      cublasCheckErrors(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));


      CHECK(cudaMalloc(&vq_score_table_gpu_, query_n_vec_ * vq_info_->n_centroid_ * sizeof(float)));
      CHECK(cudaMalloc(&filter_cent_score_gpu_, query_n_vec_ * nprobe_ * sizeof(float)));
      CHECK(cudaMalloc(&filter_centID_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));



      size_t free_mem, total_mem;
      CHECK(cudaMemGetInfo(&free_mem, &total_mem));

      double free_mem_mb = free_mem / (1024.0 * 1024.0);
      double total_mem_mb = total_mem / (1024.0 * 1024.0);
      double used_mem_mb = total_mem_mb - free_mem_mb;
      double used_percentage = (used_mem_mb / total_mem_mb) * 100.0;

      std::cout.precision(2);
      std::cout << std::fixed;
      std::cout << "[" << "after generate VQScoreCandidatePP" << "] "
                << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
                << "free: " << free_mem_mb << " MB, "
                << "total: " << total_mem_mb << " MB" << std::endl;
  }

  // void matrixMultiplyTest(const float* vec1_l, const float* vec2_l,
  //                         int n_vec1, int n_vec2, int vec_dim, float* result)
  // {
  //     for (int vecID1 = 0; vecID1 < n_vec1; ++vecID1)
  //     {
  //         for (int vecID2 = 0; vecID2 < n_vec2; ++vecID2)
  //         {
  //             float sum = 0.0f;
  //             for (int dim = 0; dim < vec_dim; ++dim)
  //             {
  //                 sum += vec1_l[vecID1 * vec_dim + dim] * vec2_l[vecID2 * vec_dim + dim];
  //             }
  //             result[vecID1 * n_vec2 + vecID2] = sum;
  //         }
  //     }
  // }

  void generateCandidate(const float* query_gpu)
  {
      nvtxRangePushA("similar fetch");

      // const float alpha = 1.0f;
      // const float beta = 0.0f;
      //
      // cublasCheckErrors(cublasLtMatmul(cublaslt_handle_,
      //                           matmul_desc_,
      //                           &alpha,
      //                           query_gpu, mat_a_desc_,
      //                           vq_info_.centroid_l_gpu_, mat_b_desc_,
      //                           &beta,
      //                           vq_score_table_gpu_, mat_c_desc_,
      //                           vq_score_table_gpu_, mat_c_desc_,
      //                           nullptr, nullptr, 0, resource_->stream_l_[threadID_]));

      // compute and sort the score matrix
      MatrixMultiply(resource_->handle_l_[threadID_], query_gpu, vq_info_->centroid_l_gpu_,
                     query_n_vec_, vq_info_->n_centroid_, vq_info_->vec_dim_,
                     vq_score_table_gpu_);

      auto exec_policy = thrust::cuda::par.on(resource_->stream_l_[threadID_]);
      thrust::for_each(exec_policy, vq_score_table_gpu_,
                       vq_score_table_gpu_ + query_n_vec_ * vq_info_->n_centroid_,
                       [] __device__(float& x) { x += 1.0f; });

      // test
      // thrust::for_each(
      //     thrust::device,
      //     thrust::make_zip_iterator(thrust::make_tuple(vq_score_table_sort_gpu_, vq_score_table_gpu_)),
      //     thrust::make_zip_iterator(thrust::make_tuple(
      //         vq_score_table_sort_gpu_ + query_n_vec_ * vq_info_.n_centroid_,
      //         vq_score_table_gpu_ + query_n_vec_ * vq_info_.n_centroid_)),
      //     [=]__device__(auto& tup)
      //     {
      //         assert(thrust::get<0>(tup) == thrust::get<1>(tup));
      //     }
      // );
      // std::vector<float> vq_score_table_before(query_n_vec_ * vq_info_.n_centroid_);
      // CHECK(cudaMemcpy(vq_score_table_before.data(), vq_score_table_gpu_,
      //     query_n_vec_ * vq_info_.n_centroid_ * sizeof(float), cudaMemcpyDeviceToHost));
      // CHECK(cudaDeviceSynchronize());
      // end test

      const uint32_t* input_idx_l = nullptr;
      bool select_min = false;
      raft::matrix::detail::select_k(resource_->handle_raft_l_[threadID_],
                                     vq_score_table_gpu_,
                                     input_idx_l,
                                     query_n_vec_,
                                     vq_info_->n_centroid_,
                                     nprobe_,
                                     filter_cent_score_gpu_,
                                     filter_centID_l_gpu_,
                                     select_min);

      CHECK(cudaStreamSynchronize(resource_->stream_l_[threadID_]));
      nvtxRangePop();
  }

  void finishCompute()
  {
      cudaMemFreeMarco(this->bias_gpu_);
      cublasCheckErrors(cublasLtDestroy(cublaslt_handle_));
      cublasCheckErrors(cublasLtMatmulDescDestroy(matmul_desc_));
      cublasCheckErrors(cublasLtMatrixLayoutDestroy(mat_a_desc_));
      cublasCheckErrors(cublasLtMatrixLayoutDestroy(mat_b_desc_));
      cublasCheckErrors(cublasLtMatrixLayoutDestroy(mat_c_desc_));

      cudaMemFreeMarco(this->vq_score_table_gpu_);
      cudaMemFreeMarco(this->filter_cent_score_gpu_);
      cudaMemFreeMarco(this->filter_centID_l_gpu_);
  }
};
}
#endif //GENERATEVQSCORECANDIDATEPP_HPP
