//
// Created by bianzheng on 2026/1/7.
//

#ifndef VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPALG_HPP_
#define VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPALG_HPP_

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/struct/BoolArray.hpp"
#include "include/alg/probe/IGP/IGPProximityGraphProbe.hpp"
#include "include/alg/probe/IGP/IGPStruct.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch {
class IGPAlg {
 public:
  // know in build index
  const std::vector<uint32_t> *_centroid2itemID_l; // n_centroid
  const float *_centroid_l; // n_centroid * vec_dim
  uint32_t _n_item;
  size_t _n_vecs;
  uint32_t _n_centroid, _vec_dim;
  // know before retrieval
  uint32_t _query_n_vecs;

  // data structure for sorting the array
  // know in build index
  IGPProximityGraphProbe _pg_probe;
  // know before retrieval
  // caching the sorted probe element
  std::vector<IGPProbeEle> _probe_ele_l; // query_n_vec * nprobe

  // data structure used for probing
  // know in build index
  // store the lower bound of each item
  uint32_t n_cand_item_;
  std::vector<char> _visit_item_l; // n_item, defaule value: false
  std::vector<uint32_t> _candID2itemID_l; // n_item, store the candidates item of the probe score
  std::vector<uint32_t> _itemID2candID_l; // n_item
  std::vector<float> _vec_max_score_l; // n_item, index is the candID
  std::vector<float> _item_appr_scr_l; // _n_item, index is the candID
  // know before retrieval
  // min heap that stores the item with top-k score
  IGPTopkItemMinHeap _topk_score_q; // max_size: probe_topk

  // input parameter
  uint32_t _nprobe{}, _probe_topk{};

  TimeRecord _record;
  // output indicator
  uint32_t _n_seen_item;
  size_t _n_vq_score_linear_scan;

  IGPAlg() = default;

  IGPAlg(const std::vector<uint32_t> *centroid2itemID_l,
           const float *centroid_l,
           const uint32_t n_item, const size_t n_vecs,
           const uint32_t n_centroid, const uint32_t vec_dim
  ) {
    this->_centroid2itemID_l = centroid2itemID_l;
    this->_centroid_l = centroid_l;
    this->_n_item = n_item;
    this->_n_vecs = n_vecs;
    this->_n_centroid = n_centroid;
    this->_vec_dim = vec_dim;

    this->_pg_probe = IGPProximityGraphProbe(centroid_l, n_centroid, vec_dim);

    n_cand_item_ = n_item;
    this->_visit_item_l.resize(n_item);
    this->_candID2itemID_l.resize(n_item);
    this->_itemID2candID_l.resize(n_item);
    this->_vec_max_score_l.resize(n_item);
    this->_item_appr_scr_l.resize(n_item);
  }

  void set_retrieval_parameter(
      const uint32_t query_n_vecs, const uint32_t topk,
      const size_t nprobe, const uint32_t probe_topk
  ) {
    this->_query_n_vecs = query_n_vecs;

    this->_pg_probe.set_query_info(query_n_vecs, nprobe);

    this->_probe_ele_l.resize(query_n_vecs * nprobe);
    this->_topk_score_q = IGPTopkItemMinHeap(probe_topk);

    this->_nprobe = nprobe;
    this->_probe_topk = probe_topk;
    assert(0 <= _probe_topk && _probe_topk <= _n_item);
  }

  void reset() {
    this->_pg_probe.reset();

    _visit_item_l.assign((size_t) _n_item, false);
    _item_appr_scr_l.assign((size_t) n_cand_item_, 0.0f);

    _n_seen_item = 0;
    _n_vq_score_linear_scan = 0;
  }

  void compute_lower_bound(const IGPProbeEle *ele_l, uint32_t &nprobe) {
    _vec_max_score_l.assign((size_t) n_cand_item_, 0.0f);

    for (uint32_t probeID = 0; probeID < nprobe; probeID++) {
      const IGPProbeEle probe = ele_l[probeID];
      const uint32_t centID = probe.centroidID;
      const float centroid_score = probe.score;
      /*
       * the global minimum score of each query-centroid score is -1 because of unit norm
       * we add 1 to all query-centroid score, this makes the minimum query-centroid score as 0
       */

      const float cent_score_comp = centroid_score + 1.0f;
      // refine the score of that query centroid pair
      for (uint32_t itemID : _centroid2itemID_l[centID]) {
        const uint32_t candID = _itemID2candID_l[itemID];
        assert(0 <= candID && candID < n_cand_item_);
        _vec_max_score_l[candID] = std::max(_vec_max_score_l[candID], cent_score_comp);
        _n_vq_score_linear_scan++;
      }
    }

    for(uint32_t candID=0;candID < n_cand_item_;candID++){
      _item_appr_scr_l[candID] += _vec_max_score_l[candID];
    }
  }

  void compute_topk_lower_bound() {
    _topk_score_q.Reset();

    for (uint32_t candID=0;candID < n_cand_item_; candID++) {
      const float lb_score = _item_appr_scr_l[candID];
      const uint32_t itemID = _candID2itemID_l[candID];
      const uint32_t n_qvec_not_refine = 0;

      _topk_score_q.Update(lb_score, itemID, n_qvec_not_refine);
    }
  }

  void compute_candidate_item(const IGPProbeEle *ele_l, const uint32_t nprobe) {
    uint32_t cand_cnt = 0;
    for (uint32_t qvecID = 0; qvecID < _query_n_vecs; qvecID++) {
      for (uint32_t probeID = 0; probeID < nprobe; probeID++) {
        const IGPProbeEle probe = ele_l[qvecID * nprobe + probeID];
        const uint32_t centID = probe.centroidID;
        for (uint32_t itemID : _centroid2itemID_l[centID]) {
          if (!_visit_item_l[itemID]) {
            _visit_item_l[itemID] = true;
            _candID2itemID_l[cand_cnt] = itemID;
            _itemID2candID_l[itemID] = cand_cnt;
            cand_cnt++;
          }
        }
      }
    }
    n_cand_item_ = cand_cnt;
    _n_seen_item = n_cand_item_;

  }

  void refine(const float *query,
              std::pair<float, uint32_t> *item_candidate_cache_l,
              uint32_t &n_filter_item, uint32_t &n_compute_score,

              const uint32_t queryID,
              const uint32_t *vq_code_l,
              const uint32_t *item_n_vec_l,
              const size_t *item_n_vec_offset_l) {
    this->_pg_probe.set_query(query);

    for (uint32_t qvecID = 0; qvecID < _query_n_vecs; qvecID++) {
      this->_pg_probe.next_probe_element(_nprobe, qvecID,
                                         _probe_ele_l.data() + qvecID * _nprobe,
                                         n_compute_score);
    }

    compute_candidate_item(_probe_ele_l.data(), _nprobe);

    for (uint32_t qvecID = 0; qvecID < _query_n_vecs; qvecID++) {
      compute_lower_bound(_probe_ele_l.data() + qvecID * _nprobe, _nprobe);
    }

    compute_topk_lower_bound();

    // compute the candidates
    assert(_topk_score_q.Size() == _probe_topk);
    const ItemProbeEle *topk_lb_l = _topk_score_q.Data();
    for (uint32_t candID = 0; candID < _topk_score_q.Size(); candID++) {
      //                if(queryID == 78){
      //                    spdlog::info("candidate queryID {}, score {:.3f}, itemID {}", queryID, topk_lb_l[candID].first, topk_lb_l[candID].second);
      //                }
      item_candidate_cache_l[candID] = std::make_pair(topk_lb_l[candID].score_,
                                                      topk_lb_l[candID].itemID_);
      assert(0 <= item_candidate_cache_l[candID].second &&
             item_candidate_cache_l[candID].second < _n_item);
    }
    n_filter_item = _probe_topk;

  }
};
}
#endif //VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPALG_HPP_
