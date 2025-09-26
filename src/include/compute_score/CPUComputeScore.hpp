//
// Created by bianzheng on 2023/7/15.
//

#ifndef VECTORSETSEARCH_CPUCOMPUTESCORE_HPP
#define VECTORSETSEARCH_CPUCOMPUTESCORE_HPP

#include <memory>
#include <vector>

namespace VectorSetSearch {

    class CPUComputeScore {

        std::vector<float> query_vecs_l_;
        uint32_t n_query_, query_n_vecs_, vec_dim_;

    public:
        CPUComputeScore() = default;

        inline CPUComputeScore(const float *query_vecs_l,
                               const uint32_t &n_query, const uint32_t &query_n_vecs, const uint32_t &vec_dim) {

            this->n_query_ = n_query;
            this->query_n_vecs_ = query_n_vecs;
            this->vec_dim_ = vec_dim;

            query_vecs_l_ = std::vector<float>(n_query_ * query_n_vecs_ * vec_dim_);
            std::memcpy(query_vecs_l_.data(), query_vecs_l, sizeof(float) * n_query_ * query_n_vecs_ * vec_dim_);
        }

        void computeItemScore(const float **item_vecs_l,
                              const uint32_t *item_n_vecs_l, const uint32_t &n_item,
                              float *const distance_l) {

#pragma omp parallel for default(none) shared(distance_l, n_item, item_vecs_l, item_n_vecs_l)
            for (uint32_t queryID = 0; queryID < n_query_; queryID++) {
                const float *tmp_query_vecs = query_vecs_l_.data() + (int64_t) queryID * vec_dim_ * query_n_vecs_;

                for (uint32_t itemID = 0; itemID < n_item; itemID++) {
                    const float *tmp_item_vecs = item_vecs_l[itemID];
                    const float score = vectorSetDistance(tmp_query_vecs, query_n_vecs_,
                                                          tmp_item_vecs, item_n_vecs_l[itemID], vec_dim_);
                    distance_l[(int64_t) queryID * n_item + itemID] = score;
                }
            }

        }

        void finishCompute() {
        }
    };

}
#endif //VECTORSETSEARCH_CPUCOMPUTESCORE_HPP
