//
// Created by Administrator on 2025/9/5.
//

#ifndef IGPGPUPP_HPP
#define IGPGPUPP_HPP

#include <spdlog/spdlog.h>
#include <fstream>
#include <cnpy.h>
#include <omp.h>

#include "include/struct/IVFIndex.hpp"
#include "include/alg/refine/ResidualScalarQuantizationCPP.hpp"
#include "include/util/TimeMemory.hpp"
#include "include/alg/igp_gpu_pp/IGPGPURetrievalPP.hpp"

namespace VectorSetSearch::Method {
struct IGPGPUPPResult {
  std::vector<float> result_score_l; // n_query * topk
  std::vector<uint32_t> result_ID_l; // n_query * topk
  std::vector<double> compute_time_l; // n_query

  double batch_query_time;
  uint32_t n_query, topk;
};

class IGPGPUPP {
 public:
  cnpy::NpyArray item_n_vec_l_npy_;
  uint32_t _n_item{}, max_item_n_vec_{}, min_item_n_vec_{};
  size_t _n_vecs{};
  const uint32_t *_item_n_vecs_l; // n_item
  std::vector<size_t> _item_n_vecs_offset_l; // n_item

  cnpy::NpyArray centroid_l_npy_;
  uint32_t _n_centroid{}, _vec_dim{};
  const float *_centroid_l; // _n_centroid * vec_dim
  cnpy::NpyArray vq_code_l_npy_;
  const uint32_t *_code_l; // n_vecs

  std::vector<std::vector<uint32_t>> _centroid2itemID_l; // n_centroid
  size_t n_ele_ivf_;
  std::vector<size_t> ivf_offset_l_; // n_centroid
  std::vector<uint32_t> ivf_size_l_; // n_centroid
  std::vector<uint32_t> ivf_; // n_ele_ivf_

  uint32_t n_bit_, n_val_per_byte_, n_packed_val_per_vec_;

  cnpy::NpyArray weight_l_npy_;
  uint32_t n_weight_;
  const float *weight_l_; // n_weight

  cnpy::NpyArray residual_code_npy_;
  const uint8_t *residual_code_l_; // n_vecs * n_packed_val_per_vec_

  std::unique_ptr<ItemVecInfoPP> item_vec_info_;
  std::unique_ptr<VQInfoPP> vq_info_;
  std::unique_ptr<IVFInfoPP> ivf_info_;
  std::unique_ptr<SQInfoPP> sq_info_;
  std::unique_ptr<ResourcePP> resource_;

  std::vector<GPGPURetrievalPP> retrieval_ins_l_;

  IGPGPUPP() = default;

  IGPGPUPP(const std::string &item_n_vec_l_fname) {
    item_n_vec_l_npy_ = cnpy::npy_load(item_n_vec_l_fname);
    assert(item_n_vec_l_npy_.word_size == sizeof(uint32_t));
    assert(item_n_vec_l_npy_.shape.size() == 1);
    const uint32_t n_item = item_n_vec_l_npy_.shape[0];
    const uint32_t *item_n_vec_l = item_n_vec_l_npy_.data<uint32_t>();

    _n_item = n_item;
    auto max_ptr = std::max_element(item_n_vec_l, item_n_vec_l + n_item);
    max_item_n_vec_ = *max_ptr;
    auto min_ptr = std::min_element(item_n_vec_l, item_n_vec_l + n_item);
    min_item_n_vec_ = *min_ptr;

    _item_n_vecs_l = item_n_vec_l_npy_.data<uint32_t>();
    _item_n_vecs_offset_l.resize(_n_item);

    size_t n_vecs = _item_n_vecs_l[0];
    _item_n_vecs_offset_l[0] = 0;
    for (uint32_t itemID = 1; itemID < _n_item; itemID++) {
      n_vecs += _item_n_vecs_l[itemID];
      _item_n_vecs_offset_l[itemID] = _item_n_vecs_offset_l[itemID - 1] + _item_n_vecs_l[itemID - 1];
    }
    _n_vecs = n_vecs;

    const uint32_t n_cpu_thread = 3;
    resource_ = std::make_unique<ResourcePP>(n_cpu_thread);
  }

  void buildIVFIndex() {
    ivf_offset_l_.resize(_n_centroid);
    ivf_offset_l_[0] = 0;
    for (uint32_t centID = 1; centID < _n_centroid; centID++) {
      ivf_offset_l_[centID] = ivf_offset_l_[centID - 1] + _centroid2itemID_l[centID - 1].size();
    }

    ivf_size_l_.resize(_n_centroid);
    for (uint32_t centID = 0; centID < _n_centroid; centID++) {
      ivf_size_l_[centID] = _centroid2itemID_l[centID].size();
    }

    size_t n_ele_ivf = 0;
    for (uint32_t centID = 0; centID < _n_centroid; centID++) {
      n_ele_ivf += _centroid2itemID_l[centID].size();
    }
    ivf_.resize(n_ele_ivf);
    this->n_ele_ivf_ = n_ele_ivf;

    for (uint32_t centID = 0; centID < _n_centroid; centID++) {
      const size_t offset = ivf_offset_l_[centID];
      const uint32_t n_ele = _centroid2itemID_l[centID].size();
      std::memcpy(ivf_.data() + offset, _centroid2itemID_l[centID].data(), n_ele * sizeof(uint32_t));
    }
  }

  bool buildIndex(const std::string &centroid_l_fname, const std::string &vq_code_l_fname,
                  const std::string &ivf_index_fname,
                  const std::string &weight_l_fname, const std::string &residual_code_l_fname,
                  const uint32_t n_centroid, const uint32_t n_bit) {
    centroid_l_npy_ = cnpy::npy_load(centroid_l_fname);
    assert(centroid_l_npy_.word_size == sizeof(float));
    assert(centroid_l_npy_.shape.size() == 2);
    _n_centroid = centroid_l_npy_.shape[0];
    _vec_dim = centroid_l_npy_.shape[1];
    _centroid_l = centroid_l_npy_.data<float>();
    assert(_n_centroid == n_centroid);

    vq_code_l_npy_ = cnpy::npy_load(vq_code_l_fname);
    assert(vq_code_l_npy_.word_size == sizeof(uint32_t));
    assert(vq_code_l_npy_.shape.size() == 1);
    assert(vq_code_l_npy_.shape[0] == _n_vecs);
    _code_l = vq_code_l_npy_.data<uint32_t>();

#ifndef NDEBUG
    for (uint32_t vecID = 0; vecID < _n_vecs; vecID++) {
      assert(_code_l[vecID] < _n_centroid);
    }
#endif

    _centroid2itemID_l = Method::readIVFIndex(ivf_index_fname);

#ifndef NDEBUG
    assert(_centroid2itemID_l.size() == _n_centroid);
    size_t n_element = 0;
    for (size_t centID = 0; centID < _n_centroid; centID++) {
      n_element += _centroid2itemID_l[centID].size();
    }
    assert(n_element >= _n_item);
#endif
    // _centroid2itemID_l -> n_ele_ivf_, ivf_offset_l_, ivf_size_l_, ivf_
    buildIVFIndex();

    n_bit_ = n_bit;
    constexpr uint32_t n_bit_per_byte = 8;
    n_val_per_byte_ = n_bit_per_byte / n_bit_;
    assert(n_bit_per_byte % n_bit_ == 0);
    n_packed_val_per_vec_ = (_vec_dim + n_val_per_byte_ - 1) / n_val_per_byte_;

    weight_l_npy_ = cnpy::npy_load(weight_l_fname);
    assert(weight_l_npy_.word_size == sizeof(float));
    assert(weight_l_npy_.shape.size() == 1);
    n_weight_ = weight_l_npy_.shape[0];
    weight_l_ = weight_l_npy_.data<float>();

    residual_code_npy_ = cnpy::npy_load(residual_code_l_fname);
    assert(residual_code_npy_.word_size == sizeof(uint8_t));
    assert(residual_code_npy_.shape.size() == 1);
    assert(residual_code_npy_.shape[0] == _n_vecs * n_packed_val_per_vec_);
    residual_code_l_ = residual_code_npy_.data<uint8_t>();

    size_t free_mem, total_mem;
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    double free_mem_mb = free_mem / (1024.0 * 1024.0);
    double total_mem_mb = total_mem / (1024.0 * 1024.0);
    double used_mem_mb = total_mem_mb - free_mem_mb;
    double used_percentage = (used_mem_mb / total_mem_mb) * 100.0;
    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "[" << "before init" << "] "
              << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
              << "free: " << free_mem_mb << " MB, "
              << "total: " << total_mem_mb << " MB" << std::endl;

    item_vec_info_ = std::make_unique<ItemVecInfoPP>(_n_item, max_item_n_vec_, min_item_n_vec_, _vec_dim, _n_vecs,
                                                        _item_n_vecs_l, _item_n_vecs_offset_l.data());
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    free_mem_mb = free_mem / (1024.0 * 1024.0);
    total_mem_mb = total_mem / (1024.0 * 1024.0);
    used_mem_mb = total_mem_mb - free_mem_mb;
    used_percentage = (used_mem_mb / total_mem_mb) * 100.0;
    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "[" << "after ItemVecInfoPP" << "] "
              << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
              << "free: " << free_mem_mb << " MB, "
              << "total: " << total_mem_mb << " MB" << std::endl;

    vq_info_ = std::make_unique<VQInfoPP>(n_centroid, _vec_dim, _n_vecs, _centroid_l, _code_l);
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    free_mem_mb = free_mem / (1024.0 * 1024.0);
    total_mem_mb = total_mem / (1024.0 * 1024.0);
    used_mem_mb = total_mem_mb - free_mem_mb;
    used_percentage = (used_mem_mb / total_mem_mb) * 100.0;
    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "[" << "after VQInfoPP" << "] "
              << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
              << "free: " << free_mem_mb << " MB, "
              << "total: " << total_mem_mb << " MB" << std::endl;

    ivf_info_ = std::make_unique<IVFInfoPP>(n_centroid, n_ele_ivf_,
                                               ivf_offset_l_.data(), ivf_size_l_.data(), ivf_.data());
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    free_mem_mb = free_mem / (1024.0 * 1024.0);
    total_mem_mb = total_mem / (1024.0 * 1024.0);
    used_mem_mb = total_mem_mb - free_mem_mb;
    used_percentage = (used_mem_mb / total_mem_mb) * 100.0;
    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "[" << "after IVFInfoPP" << "] "
              << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
              << "free: " << free_mem_mb << " MB, "
              << "total: " << total_mem_mb << " MB" << std::endl;

    sq_info_ = std::make_unique<SQInfoPP>(_n_vecs, _vec_dim,
                                             n_bit, n_val_per_byte_, n_packed_val_per_vec_,
                                             n_weight_, weight_l_, residual_code_l_);
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    free_mem_mb = free_mem / (1024.0 * 1024.0);
    total_mem_mb = total_mem / (1024.0 * 1024.0);
    used_mem_mb = total_mem_mb - free_mem_mb;
    used_percentage = (used_mem_mb / total_mem_mb) * 100.0;
    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "[" << "after SQInfoPP" << "] "
              << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
              << "free: " << free_mem_mb << " MB, "
              << "total: " << total_mem_mb << " MB" << std::endl;

    retrieval_ins_l_.resize(resource_->n_cpu_thread_);
    for(uint32_t threadID=0;threadID < resource_->n_cpu_thread_;threadID++) {
      retrieval_ins_l_[threadID] = GPGPURetrievalPP(item_vec_info_.get(), vq_info_.get(),
                                                       ivf_info_.get(), sq_info_.get(), resource_.get(), threadID);
    }

    return true;
  }

  IGPGPUPPResult search(const float *query_l, const uint32_t n_query, const uint32_t query_n_vec,
                           const uint32_t topk,
                           const uint32_t nprobe, const uint32_t probe_topk) {
    if (probe_topk < topk) {
      spdlog::error("the number of refined topk is smaller than the number of returned topk, program exit");
      exit(-1);
    }
    //            if (!(refine_topk <= nprobe && nprobe <= _n_centroid)) {
    //                spdlog::error(
    //                        "nprobe is either smaller than refine_topk or larger than n_centroid, program exit");
    //                exit(-1);
    //            }


    for(uint32_t threadID = 0;threadID < resource_->n_cpu_thread_;threadID++) {
      retrieval_ins_l_[threadID].initSearch(query_n_vec, topk, nprobe, probe_topk);
    }

    std::vector<float> result_score_l((int64_t) n_query * topk);
    std::vector<uint32_t> result_ID_l((int64_t) n_query * topk);
    std::vector<double> compute_time_l(n_query);

    float* query_l_gpu;
    CHECK(cudaMalloc(&query_l_gpu, n_query * query_n_vec * _vec_dim * sizeof(float)));
    CHECK(cudaMemcpy(query_l_gpu, query_l, n_query * query_n_vec * _vec_dim * sizeof(float),
                     cudaMemcpyHostToDevice));

//#pragma omp parallel for default(none) shared(n_query, query_l, query_n_vec) num_threads(gpu_resource_->n_query_stream_)
    for(uint32_t queryID = 0; queryID < std::min(n_query, 10u); queryID++) {
      const float *query = query_l_gpu + queryID * query_n_vec * _vec_dim;
      std::vector<std::pair<float, uint32_t>> res = retrieval_ins_l_[0].search(query, -1);
    }
    spdlog::info("finish warming");

    TimeRecord batch_query_record;
    batch_query_record.reset();
#pragma omp parallel for default(none) shared(n_query, query_l, query_l_gpu, query_n_vec, topk, result_score_l, result_ID_l, compute_time_l) num_threads(resource_->n_cpu_thread_)
    for (uint32_t queryID = 0; queryID < n_query; queryID++) {
      if (queryID % 100 == 0) {
        spdlog::info("start processing queryID {}", queryID);
      }
      TimeRecord record;
//      spdlog::info("queryID {}", queryID);
      const uint32_t threadID = omp_get_thread_num();

      nvtxRangePushA((std::string("queryID ") + std::to_string(queryID)).c_str());
      record.reset();
      const float *query = query_l_gpu + queryID * query_n_vec * _vec_dim;

      std::vector<std::pair<float, uint32_t>> res = retrieval_ins_l_[threadID].search(query, queryID);

      for (uint32_t candID = 0; candID < topk; candID++) {
        const int64_t insert_offset = (int64_t) queryID * topk;
        result_score_l[insert_offset + (int64_t) candID] = res[candID].first;
        result_ID_l[insert_offset + (int64_t) candID] = res[candID].second;
      }
      nvtxRangePop();

      compute_time_l[queryID] = record.get_elapsed_time_second();
    }

    const double batch_query_time = batch_query_record.get_elapsed_time_second();
    for (uint32_t qstreamID = 0; qstreamID < resource_->n_cpu_thread_; qstreamID++) {
      retrieval_ins_l_[qstreamID].FinishCompute();
    }
    item_vec_info_->FinishCompute();
    vq_info_->FinishCompute();
    ivf_info_->FinishCompute();
    sq_info_->FinishCompute();
    resource_->FinishCompute();
    cudaMemFreeMarco(query_l_gpu);

    IGPGPUPPResult result;
    result.result_score_l = result_score_l;
    result.result_ID_l = result_ID_l;
    result.compute_time_l = compute_time_l;

    result.batch_query_time = batch_query_time;
    result.n_query = n_query;
    result.topk = topk;

    return result;
  }

};
}
#endif //IGPGPUPP_HPP
