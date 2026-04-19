//
// Created by bianzheng on 2026/1/7.
//

#ifndef VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPPROXIMITYGRAPHPROBE_HPP_
#define VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPPROXIMITYGRAPHPROBE_HPP_

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/struct/BoolArray.hpp"
#include "include/alg/probe/IGP/hnsw_probe/hnswalg.hpp"
#include "include/alg/probe/IGP/IGPStruct.hpp"
#include "include/alg/probe/ProbeHeap.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch {
class IGPProximityGraphProbe {
  // the input of this class is the centroid and the query
  // the output is the probe element when the function are called
 public:
  uint32_t _query_n_vecs, _n_centroid, _vec_dim;

  const float *_centroid_l; // n_centroid * vec_dim
  const float *_query; // query_n_vecs * vec_dim

  std::unique_ptr<hnswlib_probe::InnerProductSpace> _ip_space;
  std::unique_ptr<hnswlib_probe::HierarchicalNSW<float>> _hnsw_index;

  IGPProximityGraphProbe() = default;

  IGPProximityGraphProbe(const float *centroid_l,
                           const uint32_t n_centroid,
                           const uint32_t vec_dim
  ) {
    this->_centroid_l = centroid_l;
    this->_n_centroid = n_centroid;
    this->_vec_dim = vec_dim;

    build_hnsw(centroid_l, _n_centroid, vec_dim);
  }

  void build_hnsw(const float *centroid_l,
                  const uint32_t n_centroid, const uint32_t vec_dim) {
    const size_t max_element = n_centroid;
    const uint32_t M = 32;
    const uint32_t efConstruction = 200;
    _ip_space = std::make_unique<hnswlib_probe::InnerProductSpace>(vec_dim);
    _hnsw_index = std::make_unique<hnswlib_probe::HierarchicalNSW<float>>(_ip_space.get(),
                                                                          max_element, M, efConstruction);

    if (_hnsw_index->cur_element_count == 0) {
      const uint32_t centID = 0;
      const float *centroid = centroid_l + centID * vec_dim;
      _hnsw_index->addPoint((void *) centroid, centID);
    }

    // #pragma omp parallel for ordered default(none) shared(centroid_l, vec_dim)
    for (uint32_t centID = 1; centID < _n_centroid; centID++) {
      const float *centroid = centroid_l + centID * vec_dim;
      _hnsw_index->addPoint((void *) centroid, centID);
    }
  }

  void set_query_info(const uint32_t query_n_vecs, const uint32_t nprobe) {
    _hnsw_index->set_query_info(query_n_vecs, _vec_dim, nprobe);
    this->_query_n_vecs = query_n_vecs;
  }

  void reset() {
    _hnsw_index->reset();
  }

  void set_query(const float *query) {
    _hnsw_index->set_query(query);
    this->_query = query;
  }

  // ele_l is the output of this function
  // it should be the upper bound of all query score in the last probe
  void next_probe_element(const uint32_t nprobe, const uint32_t qvecID,
                          IGPProbeEle *ele_l,
                          uint32_t &n_computation) {
    // add the probe element, may perform the argsort if necessary
    // find the topk of all query vectors
    // for every query, iteratively add the result to the sort_probe_ele_l
    _hnsw_index->search_topk(qvecID, n_computation);

#ifndef NDEBUG
    const uint32_t score_offset = qvecID * _n_centroid;

    const uint32_t n_sort_after = std::min(_hnsw_index->ef_, _n_centroid);
    for (uint32_t candID = 0; candID < n_sort_after - 1; candID++) {
      const float prev_score = _hnsw_index->sorted_probe_ele_l_[score_offset + candID].score;
      const float this_score = _hnsw_index->sorted_probe_ele_l_[score_offset + candID + 1].score;
      assert(prev_score >= this_score);
    }
#endif

    for (uint32_t probeID = 0; probeID < nprobe; probeID++) {
      const uint32_t score_offset = qvecID * _n_centroid;
      const uint32_t centID = _hnsw_index->sorted_probe_ele_l_[
          score_offset + probeID].centroidID;
      const float centroid_score = _hnsw_index->sorted_probe_ele_l_[
          score_offset + probeID].score;

      ele_l[probeID] = IGPProbeEle(centID, centroid_score);
    }

#ifndef NDEBUG
    for (uint32_t probeID = 0; probeID < nprobe - 1; probeID++) {
      const IGPProbeEle probe = ele_l[probeID];

      assert(ele_l[probeID].score >=
             ele_l[(probeID + 1)].score);
    }
#endif
  }
};
}
#endif //VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPPROXIMITYGRAPHPROBE_HPP_
