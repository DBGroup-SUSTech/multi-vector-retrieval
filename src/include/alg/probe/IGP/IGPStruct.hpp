//
// Created by bianzheng on 2026/1/7.
//

#ifndef VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPSTRUCT_HPP_
#define VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPSTRUCT_HPP_

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"
#include "include/alg/probe/ProbeEleMinHeap.hpp"

namespace VectorSetSearch {

class IGPProbeEle {
 public:
  uint32_t centroidID;
  float score;

  IGPProbeEle() = default;

  IGPProbeEle(const uint32_t centroidID, const float score) {
    this->centroidID = centroidID;
    this->score = score;
  }
};

class IGPTopkItemMinHeap {

  uint32_t max_size_;
  uint32_t cur_size_;
  std::vector<ItemProbeEle> heap_;

  struct {
    bool operator()(const ItemProbeEle &l, const ItemProbeEle &r) const {
      if (l.score_ == r.score_) {
        return l.n_qvec_not_refine_ < r.n_qvec_not_refine_;
      }
      return l.score_ > r.score_;
    }
  } min_cmp;

 public:
  IGPTopkItemMinHeap() = default;

  IGPTopkItemMinHeap(const uint32_t &max_size) :
      cur_size_(0), max_size_(max_size) {
    heap_.resize(max_size_);
  }

  uint32_t Size() {
    assert(cur_size_ <= heap_.size());
    return cur_size_;
  }

  void Reset() {
    cur_size_ = 0;
  }

  void Update(const float item_lb, const uint32_t itemID, const uint32_t n_qvec_not_refine) {
    if (cur_size_ < max_size_) {
      heap_[cur_size_] = ItemProbeEle(item_lb, itemID, n_qvec_not_refine);
      cur_size_++;
      if (cur_size_ == max_size_) {
        std::make_heap(heap_.begin(), heap_.end(), min_cmp);
      }
    } else {
      assert(cur_size_ == max_size_);
      if (item_lb > heap_[0].score_) {
        std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
        heap_[max_size_ - 1] = ItemProbeEle(item_lb, itemID, n_qvec_not_refine);
        std::push_heap(heap_.begin(), heap_.end(), min_cmp);
      }
    }

  }

  const ItemProbeEle &MinEle() {
    // should force the input call when cur_size_ == max_size_
    assert(cur_size_ == max_size_);

#ifndef NDEBUG
    uint32_t n_smaller = 0;
    for (uint32_t i = 0; i < max_size_; i++) {
      if (heap_[i].score_ < heap_[0].score_) {
        n_smaller++;
      }
    }
    assert(n_smaller == 0);
#endif

    return heap_[0];
  }

  const ItemProbeEle *Data() {
    return heap_.data();
  }
};

}
#endif //VECTORSETSEARCH_SRC_INCLUDE_ALG_PROBE_IGP_IGPSTRUCT_HPP_
