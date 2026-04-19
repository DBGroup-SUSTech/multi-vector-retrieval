//
// Created by bianzheng on 2026/1/7.
//

#ifndef VECTORSETSEARCH_SRC_IMPL_IGP_HPP_
#define VECTORSETSEARCH_SRC_IMPL_IGP_HPP_

#include <omp.h>
#include <spdlog/spdlog.h>
#include <fstream>

#include "include/alg/refine/ResidualScalarQuantizationIGPPipeline.hpp"
#include "include/alg/probe/IGP/IGPAlg.hpp"
#include "include/alg/Distance.hpp"

#include "include/struct/MethodBase.hpp"
#include "include/struct/PositionHeap.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch::Method {
class IGP {
 public:
  uint32_t _n_item{}, _vec_dim{}, max_item_n_vec_{};
  size_t _n_vecs{};
  std::vector<uint32_t> _item_n_vecs_l; // n_item
  std::vector<size_t> _item_n_vecs_offset_l; // n_item

  uint32_t _n_centroid{};
  std::vector<std::vector<uint32_t> > _centroid2itemID_l; // n_centroid
  std::vector<float> _centroid_l; // _n_centroid * vec_dim
  std::vector<uint32_t> _code_l; // n_vecs

  pyarray_uint8 residual_code_l_py_;
  const uint8_t *residual_code_l_;

  pyarray_float weight_l_py_;
  const float *weight_l_;
  uint32_t n_weight_;
  uint32_t n_bit_;

  IGP() = default;

  IGP(const std::vector<uint32_t> &item_n_vecs_l,
                  const uint32_t &n_item,
                  const uint32_t &vec_dim,
                  const uint32_t &n_centroid,
                  const uint32_t n_bit) {
    _n_item = n_item;
    _vec_dim = vec_dim;
    auto max_ptr = std::max_element(item_n_vecs_l.begin(), item_n_vecs_l.end());
    max_item_n_vec_ = *max_ptr;

    _item_n_vecs_l = item_n_vecs_l;
    assert(item_n_vecs_l.size() == _n_item);
    _item_n_vecs_offset_l.resize(_n_item);

    size_t n_vecs = _item_n_vecs_l[0];
    _item_n_vecs_offset_l[0] = 0;
    for (uint32_t itemID = 1; itemID < _n_item; itemID++) {
      n_vecs += _item_n_vecs_l[itemID];
      _item_n_vecs_offset_l[itemID] = _item_n_vecs_offset_l[itemID - 1] + _item_n_vecs_l[itemID - 1];
    }
    _n_vecs = n_vecs;

    _n_centroid = n_centroid;
    n_bit_ = n_bit;
    _centroid2itemID_l.resize(_n_centroid);
    _centroid_l.resize(_n_centroid * _vec_dim);
    _code_l.resize(_n_vecs);
  }

  bool loadQuantizationIndex(const pyarray_float &centroid_l_py,
                             const pyarray_uint32 &code_l_py,
                             const pyarray_float &weight_l_py,
                             pyarray_uint8 &residual_code_l_py) {
    if (centroid_l_py.ndim() != 2)
      throw std::runtime_error("codebook should be 2D array");
    if (code_l_py.ndim() != 1)
      throw std::runtime_error("codeword should be 1D array");

    assert(centroid_l_py.shape(0) == _n_centroid);
    assert(centroid_l_py.shape(1) == _vec_dim);
    const float *centroid_l = centroid_l_py.data();
    std::memcpy(_centroid_l.data(), centroid_l, sizeof(float) * _n_centroid * _vec_dim);

    assert(code_l_py.shape(0) == _n_vecs);
    const uint32_t *code_l = code_l_py.data();
    std::memcpy(_code_l.data(), code_l, sizeof(uint32_t) * _n_vecs);

#ifndef NDEBUG
    for (uint32_t vecID = 0; vecID < _n_vecs; vecID++) {
      assert(_code_l[vecID] < _n_centroid);
    }
#endif

    assert(_centroid2itemID_l.size() == _n_centroid);

    for (uint32_t itemID = 0; itemID < _n_item; itemID++) {
      const size_t start_vecID = _item_n_vecs_offset_l[itemID];
      const uint32_t item_n_vecs = _item_n_vecs_l[itemID];
      std::set<uint32_t> centroidID_s;
      for (size_t item_vecID = 0; item_vecID < item_n_vecs; item_vecID++) {
        const size_t vecID = item_vecID + start_vecID;
        const uint32_t centroidID = code_l[vecID];
        centroidID_s.insert(centroidID);
      }

      for (size_t centroidID : centroidID_s) {
        _centroid2itemID_l[centroidID].push_back(itemID);
      }
    }

#ifndef NDEBUG
    assert(_centroid2itemID_l.size() == _n_centroid);
    size_t n_element = 0;
    for (size_t centID = 0; centID < _n_centroid; centID++) {
      n_element += _centroid2itemID_l[centID].size();
    }
    assert(n_element >= _n_item);
#endif

    residual_code_l_py_ = std::move(residual_code_l_py);
    residual_code_l_ = residual_code_l_py_.data();

    weight_l_py_ = std::move(weight_l_py);
    assert(weight_l_py_.ndim() == 1);
    weight_l_ = weight_l_py_.data();
    n_weight_ = weight_l_py_.shape(0);

    return true;
  }

  py::tuple search(const vector_set_list &qry_embeddings,
                   const uint32_t topk,
                   const uint32_t nprobe,
                   const uint32_t probe_topk,
                   const uint32_t &n_thread) {
    if (qry_embeddings.ndim() != 3)
      throw std::runtime_error("the vector set list dimension should be 3");
    if (qry_embeddings.shape(2) != _vec_dim)
      throw std::runtime_error("the embedding dimension should equal to the pre-assignment");
    if (probe_topk < topk) {
      spdlog::error("the number of refined topk is smaller than the number of returned topk, program exit");
      exit(-1);
    }
    //            if (!(refine_topk <= nprobe && nprobe <= _n_centroid)) {
    //                spdlog::error(
    //                        "nprobe is either smaller than refine_topk or larger than n_centroid, program exit");
    //                exit(-1);
    //            }

    std::vector<IGPAlg> filter_ins_l(n_thread);
    std::vector<ResidualCodeIGPPipeline> residual_code_ins_l(n_thread);
#pragma omp parallel for default(none) shared(n_thread, filter_ins_l, residual_code_ins_l) num_threads(n_thread)
    for (uint32_t threadID = 0; threadID < n_thread; threadID++) {
      filter_ins_l[threadID] = IGPAlg(_centroid2itemID_l.data(),
                                        _centroid_l.data(),
                                        _n_item,
                                        _n_vecs,
                                        _n_centroid,
                                        _vec_dim);

      residual_code_ins_l[threadID] = ResidualCodeIGPPipeline(residual_code_l_,
                                                              weight_l_,
                                                              _item_n_vecs_l.data(),
                                                              _item_n_vecs_offset_l.data(),
                                                              _centroid_l.data(),
                                                              _code_l.data(),
                                                              n_weight_,
                                                              _n_item,
                                                              _n_vecs,
                                                              _vec_dim,
                                                              n_bit_);
    }

    const uint32_t n_query = qry_embeddings.shape(0);
    const uint32_t query_n_vecs = qry_embeddings.shape(1);
    const float *query_start_ptr = qry_embeddings.data();

    for (uint32_t threadID = 0; threadID < n_thread; threadID++) {
      filter_ins_l[threadID].set_retrieval_parameter(query_n_vecs,
                                                     topk,
                                                     nprobe,
                                                     probe_topk);
    }

    float *result_score_l = new float[(int64_t) n_query * topk];
    uint32_t *result_ID_l = new uint32_t[(int64_t) n_query * topk];
    double *compute_time_l = new double[n_query];
    double *filter_time_l = new double[n_query];
    double *decode_time_l = new double[n_query];
    double *refine_time_l = new double[n_query];
    uint32_t *n_seen_item_l = new uint32_t[n_query];
    uint32_t *incremental_graph_n_compute_l = new uint32_t[n_query];
    size_t *n_vq_score_linear_scan_l = new size_t[n_query];

    std::vector<BatchMaxSim> batch_max_sim_l(n_thread);
    std::vector<std::vector<float> > fine_item_l_l(n_thread);
    std::vector<std::vector<uint32_t> > fine_item_n_vec_accu_l_l(n_thread);
    std::vector<std::vector<uint32_t> > fine_item_n_vec_l_l(n_thread);

    std::vector<std::vector<std::pair<float, uint32_t> > > item_candidate_cache_l_l(n_thread);

    for (uint32_t threadID = 0; threadID < n_thread; threadID++) {
      batch_max_sim_l[threadID] = BatchMaxSim(query_n_vecs, probe_topk, max_item_n_vec_);
      fine_item_l_l[threadID] = std::vector<float>(probe_topk * max_item_n_vec_ * _vec_dim);
      fine_item_n_vec_accu_l_l[threadID] = std::vector<uint32_t>(probe_topk);
      fine_item_n_vec_l_l[threadID] = std::vector<uint32_t>(probe_topk);
      item_candidate_cache_l_l[threadID] = std::vector<std::pair<float, uint32_t> >(_n_item); // n_item
    }
    TimeRecord total_record;
    total_record.reset();

#pragma omp parallel for default(none) shared(n_query, filter_ins_l, query_start_ptr, query_n_vecs, item_candidate_cache_l_l, fine_item_n_vec_l_l, fine_item_n_vec_accu_l_l, residual_code_ins_l, fine_item_l_l, batch_max_sim_l, topk, result_score_l, result_ID_l, compute_time_l, filter_time_l, decode_time_l, refine_time_l, n_seen_item_l, incremental_graph_n_compute_l, n_vq_score_linear_scan_l) num_threads(n_thread)
    for (uint32_t queryID = 0; queryID < n_query; queryID++) {
      const int threadID = omp_get_thread_num();
      TimeRecord record;
      if (queryID % 100 == 0) {
        spdlog::info("start processing queryID {}", queryID);
      }
      record.reset();
      filter_ins_l[threadID].reset();

      const float *query = query_start_ptr + queryID * query_n_vecs * _vec_dim;

      // perform the agg. topk
      uint32_t n_filter_item = _n_item;
      uint32_t n_compute_score = 0;
      filter_ins_l[threadID].refine(query,
                                    item_candidate_cache_l_l[threadID].data(),
                                    n_filter_item,
                                    n_compute_score,

                                    queryID,
                                    _code_l.data(),
                                    _item_n_vecs_l.data(),
                                    _item_n_vecs_offset_l.data());
      const double filter_time = record.get_elapsed_time_second();

      for (uint32_t candID = 0; candID < n_filter_item; candID++) {
        const uint32_t itemID = item_candidate_cache_l_l[threadID][candID].second;
        const uint32_t item_n_vec = _item_n_vecs_l[itemID];
        assert(item_n_vec <= max_item_n_vec_);
        fine_item_n_vec_l_l[threadID][candID] = item_n_vec;

        const uint32_t item_n_vec_offset =
            candID == 0
            ? 0
            : fine_item_n_vec_accu_l_l[threadID][candID - 1] + fine_item_n_vec_l_l[threadID][candID - 1];
        fine_item_n_vec_accu_l_l[threadID][candID] = item_n_vec_offset;
        residual_code_ins_l[threadID].Decode(itemID, fine_item_l_l[threadID].data() + item_n_vec_offset * _vec_dim);
      }
      const double decode_time = record.get_elapsed_time_second() - filter_time;

      batch_max_sim_l[threadID].compute(query,
                                        fine_item_l_l[threadID].data(),
                                        fine_item_n_vec_l_l[threadID].data(),
                                        fine_item_n_vec_accu_l_l[threadID].data(),
                                        _vec_dim,
                                        n_filter_item,
                                        item_candidate_cache_l_l[threadID].data());

      std::sort(item_candidate_cache_l_l[threadID].begin(),
                item_candidate_cache_l_l[threadID].begin() + n_filter_item,
                [](const std::pair<float, uint32_t> &l, const std::pair<float, uint32_t> &r) {
                  return l.first > r.first;
                });
      const double refine_time =
          record.get_elapsed_time_second() - decode_time - filter_time;

      for (uint32_t candID = 0; candID < topk; candID++) {
        const int64_t insert_offset = (int64_t) queryID * topk;
        result_score_l[insert_offset + (int64_t) candID] = item_candidate_cache_l_l[threadID][candID].first;
        result_ID_l[insert_offset + (int64_t) candID] = item_candidate_cache_l_l[threadID][candID].second;
        //                    if (queryID == 78) {
        //                        spdlog::info("queryID {}, score {:.3f}, ID {}",
        //                                     queryID, item_candidate_cache_l[candID].first,
        //                                     item_candidate_cache_l[candID].second);
        //                    }
      }

      compute_time_l[queryID] = record.get_elapsed_time_second();
      filter_time_l[queryID] = filter_time;
      decode_time_l[queryID] = decode_time;
      refine_time_l[queryID] = refine_time;

      n_seen_item_l[queryID] = filter_ins_l[threadID]._n_seen_item;
      incremental_graph_n_compute_l[queryID] = n_compute_score;
      n_vq_score_linear_scan_l[queryID] = filter_ins_l[threadID]._n_vq_score_linear_scan;
    }
    const double total_retrieval_time = total_record.get_elapsed_time_second();

    py::capsule handle_result_score_ptr(result_score_l, Method::PtrDelete<float>);
    py::capsule handle_result_ID_ptr(result_ID_l, Method::PtrDelete<uint32_t>);
    py::capsule handle_compute_time_ptr(compute_time_l, Method::PtrDelete<double>);

    py::capsule handle_filter_time_ptr(filter_time_l, Method::PtrDelete<double>);
    py::capsule handle_decode_time_ptr(decode_time_l, Method::PtrDelete<double>);
    py::capsule handle_refine_time_ptr(refine_time_l, Method::PtrDelete<double>);

    py::capsule handle_n_seen_item_ptr(n_seen_item_l, Method::PtrDelete<uint32_t>);
    py::capsule handle_incremental_graph_n_compute_ptr(incremental_graph_n_compute_l,
                                                       Method::PtrDelete<uint32_t>);
    py::capsule handle_n_vq_score_linear_scan_ptr(n_vq_score_linear_scan_l, Method::PtrDelete<size_t>);

    return py::make_tuple(
        total_retrieval_time,
        py::array_t<float>(
            {n_query, topk},
            {topk * sizeof(float), sizeof(float)},
            result_score_l,
            handle_result_score_ptr
        ),
        py::array_t<uint32_t>(
            {n_query, topk},
            {topk * sizeof(uint32_t), sizeof(uint32_t)},
            result_ID_l,
            handle_result_ID_ptr
        ),
        py::array_t<double>(
            {n_query},
            {sizeof(double)},
            compute_time_l,
            handle_compute_time_ptr
        ),

        py::array_t<double>(
            {n_query},
            {sizeof(double)},
            filter_time_l,
            handle_filter_time_ptr
        ),
        py::array_t<double>(
            {n_query},
            {sizeof(double)},
            decode_time_l,
            handle_decode_time_ptr
        ),
        py::array_t<double>(
            {n_query},
            {sizeof(double)},
            refine_time_l,
            handle_refine_time_ptr
        ),

        py::array_t<uint32_t>(
            {n_query},
            {sizeof(uint32_t)},
            n_seen_item_l,
            handle_n_seen_item_ptr
        ),
        py::array_t<uint32_t>(
            {n_query},
            {sizeof(uint32_t)},
            incremental_graph_n_compute_l,
            handle_incremental_graph_n_compute_ptr
        ),
        py::array_t<size_t>(
            {n_query},
            {sizeof(size_t)},
            n_vq_score_linear_scan_l,
            handle_n_vq_score_linear_scan_ptr
        )
    );
  }
};
}
#endif //VECTORSETSEARCH_SRC_IMPL_IGP_HPP_
