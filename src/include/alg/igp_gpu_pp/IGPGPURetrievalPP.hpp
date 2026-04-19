//
// Created by Administrator on 2025/9/5.
//

#ifndef IGPGPURETRIEVALPP_HPP
#define IGPGPURETRIEVALPP_HPP

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <nvtx3/nvToolsExt.h>

#include "include/util/CUDAUtil.hpp"
#include "include/alg/MatrixMulCUBLAS.hpp"
#include "include/alg/igp_gpu_pp/DataStructPP.hpp"
#include "include/alg/igp_gpu_pp/GenerateVQScoreCandidatePP.hpp"
#include "include/alg/igp_gpu_pp/FilterPP.hpp"
#include "include/alg/igp_gpu_pp/RerankPP.hpp"

namespace VectorSetSearch {
class GPGPURetrievalPP {
 public:
  const ItemVecInfoPP *item_vec_info_;
  const VQInfoPP *vq_info_;
  const IVFInfoPP *ivf_info_;
  const SQInfoPP *sq_info_;
  const ResourcePP *resource_;
  uint32_t threadID_;

  uint32_t query_n_vec_, topk_, nprobe_, probe_topk_;

  GenerateVQScoreCandidatePP gen_cand_ins_;
  FilterPP filter_ins_;
  RerankPP rerank_ins_;

  std::vector<float> cand_score_l_; // probe_topk
  std::vector<std::pair<float, uint32_t>> item_cand_l_; // probe_topk

  GPGPURetrievalPP() = default;

  GPGPURetrievalPP(const ItemVecInfoPP *item_vec_info, const VQInfoPP *vq_info,
                   const IVFInfoPP *ivf_info, const SQInfoPP *sq_info,
                   const ResourcePP *resource, const uint32_t threadID) {
    item_vec_info_ = item_vec_info;
    vq_info_ = vq_info;
    ivf_info_ = ivf_info;
    sq_info_ = sq_info;

    resource_ = resource;
    threadID_ = threadID;
  }

  void initSearch(const uint32_t query_n_vec, const uint32_t topk,
                  const uint32_t nprobe, const uint32_t probe_topk) {
    this->query_n_vec_ = query_n_vec;
    this->topk_ = topk;
    this->nprobe_ = nprobe;
    this->probe_topk_ = probe_topk;
    if (nprobe > vq_info_->n_centroid_) {
      spdlog::error("nprobe is larger than n_centroid, nprobe {}, n_centroid {}",
                    nprobe, vq_info_->n_centroid_);
      exit(-1);
    }


    // cublasCheckErrors(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));

    gen_cand_ins_ = GenerateVQScoreCandidatePP(item_vec_info_, vq_info_, resource_,
                                               threadID_, query_n_vec_, nprobe_);
    filter_ins_ = FilterPP(item_vec_info_, vq_info_, ivf_info_, resource_,
                           threadID_, query_n_vec_, nprobe_, probe_topk_);
    rerank_ins_ = RerankPP(item_vec_info_, vq_info_, sq_info_, resource_,
                           threadID_, query_n_vec_, probe_topk_);

    cand_score_l_.resize(probe_topk_);
    item_cand_l_.resize(probe_topk_);
  }

  void reset() {
    nvtxRangePushA("Reset");
    // another method is to initialize in cpu and copy from it
    filter_ins_.reset();
    rerank_ins_.reset();
    nvtxRangePop();
  }

  std::vector<std::pair<float, uint32_t>> search(const float *query_gpu,
                                                 const uint32_t queryID) {
    //    n_scan_vq_score = 0;
    //    n_add_vq_score = 0;
    //    n_seen_item = 0;
    resource_->AcquireCompResource(threadID_);
    reset();

    gen_cand_ins_.generateCandidate(query_gpu);

    filter_ins_.filter(gen_cand_ins_.filter_cent_score_gpu_, gen_cand_ins_.filter_centID_l_gpu_);
    resource_->ReleaseCompResource(threadID_);

    // refine
    rerank_ins_.rerank(query_gpu,
                       filter_ins_.filter_itemID_l_gpu_,
                       queryID);

    nvtxRangePushA("trans result");
    CHECK(cudaMemcpy(cand_score_l_.data(), rerank_ins_.score_l_gpu_,
                     probe_topk_ * sizeof(float),
                     cudaMemcpyDeviceToHost));
    nvtxRangePop();

    for (uint32_t i = 0; i < probe_topk_; i++) {
      item_cand_l_[i] = std::make_pair(cand_score_l_[i], rerank_ins_.itemID_l_[i]);
    }
    std::sort(item_cand_l_.begin(), item_cand_l_.begin() + probe_topk_,
              [](const std::pair<float, uint32_t> &l, const std::pair<float, uint32_t> &r) {
                return l.first > r.first;
              });

    // test
    // uint32_t n_diff = 0;
    // for (uint32_t candID = 0; candID < probe_topk_; candID++)
    // {
    //     if (rerank_ins_.rerank_itemID_l_[candID] != item_cand_l_[candID].second)
    //     {
    //         n_diff++;
    //     }
    // }
    // assert(n_diff != 0);
    // end test

    return item_cand_l_;
  }

  void FinishCompute() {
    gen_cand_ins_.finishCompute();
    filter_ins_.finishCompute();
    rerank_ins_.finishCompute();
  }
};
}
#endif //IGPGPURETRIEVALPP_HPP
