//
// Created by bianzheng on 2024/5/30.
//

#ifndef VECTORSETSEARCH_MINHEAP_HPP
#define VECTORSETSEARCH_MINHEAP_HPP

#include <vector>

namespace VectorSetSearch {

    // max heap in lower bound, max heap in lower bound
    // min heap is >, so max heap is <

    class ProbeEle {
    public:
        float score_;
        uint32_t itemID_;
        uint32_t first_probeID_;

        ProbeEle() = default;

        ProbeEle(const float &score, const uint32_t &itemID, const uint32_t &first_probeID) {
            this->score_ = score;
            this->itemID_ = itemID;
            this->first_probeID_ = first_probeID;
        }
    };

    class MinHeap {

        uint32_t max_size_;
        uint32_t cur_size_;
        std::vector<ProbeEle> heap_;

        struct {
            bool operator()(const ProbeEle &l, const ProbeEle &r) const {
                if (l.score_ == r.score_) {
                    return l.first_probeID_ < r.first_probeID_;
                }
                return l.score_ > r.score_;
            }
        } min_cmp;

    public:
        MinHeap() = default;

        MinHeap(const uint32_t &max_size) :
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

        void Update(const ProbeEle &item) {
            if (cur_size_ < max_size_) {
                heap_[cur_size_] = item;
                cur_size_++;
                if (cur_size_ == max_size_) {
                    std::make_heap(heap_.begin(), heap_.end(), min_cmp);
                }
            } else {
                assert(cur_size_ == max_size_);
                if (item.score_ > heap_[0].score_ ||
                    (item.score_ == heap_[0].score_ && item.first_probeID_ < heap_[0].first_probeID_)) {
                    assert(cur_size_ == max_size_);
                    std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
                    heap_[max_size_ - 1] = item;
                    std::push_heap(heap_.begin(), heap_.end(), min_cmp);
                }
            }

        }


        void Update(const float &item_lb, const uint32_t &itemID, const uint32_t &first_nprobe) {
            if (cur_size_ < max_size_) {
                heap_[cur_size_] = ProbeEle(item_lb, itemID, first_nprobe);
                cur_size_++;
                if (cur_size_ == max_size_) {
                    std::make_heap(heap_.begin(), heap_.end(), min_cmp);
                }
            } else {
                assert(cur_size_ == max_size_);
                if (item_lb > heap_[0].score_ ||
                    (item_lb == heap_[0].score_ && first_nprobe < heap_[0].first_probeID_)) {
                    assert(cur_size_ == max_size_);
                    std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
                    heap_[max_size_ - 1] = ProbeEle(item_lb, itemID, first_nprobe);
                    std::push_heap(heap_.begin(), heap_.end(), min_cmp);
                }
            }

        }

        ProbeEle Top() {
            if (cur_size_ < max_size_) {
                spdlog::error("The heap is not full, the top item is not valid");
                exit(-1);
            } else {
                assert(cur_size_ == max_size_);
                return heap_[0];
            }
        }

        ProbeEle *Data() {
            return heap_.data();
        }

    };

}
#endif //VECTORSETSEARCH_MINHEAP_HPP
