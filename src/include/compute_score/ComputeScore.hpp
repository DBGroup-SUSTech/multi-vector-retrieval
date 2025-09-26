//
// Created by bianzheng on 2023/7/15.
//

#ifndef VECTORSETSEARCH_COMPUTESCORE_HPP
#define VECTORSETSEARCH_COMPUTESCORE_HPP

#include <vector>
#include <parallel/algorithm>
#include <thread>
#include <spdlog/spdlog.h>

#ifdef USE_GPU

#include "include/compute_score/GPUComputeScore.hpp"

#else

#include "include/compute_score/CPUComputeScore.hpp"

#endif


namespace VectorSetSearch {
    class ComputeScore {

#ifdef USE_GPU
        GPUComputeScore gpu_;
#else
        CPUComputeScore cpu_;
#endif
    public:

        ComputeScore() = default;

        inline ComputeScore(const float *query_vecs_l,
                            const uint32_t &n_query, const uint32_t &query_n_vecs, const uint32_t &vec_dim) {

#ifdef USE_GPU
            spdlog::info("use GPU");
            gpu_ = GPUComputeScore(query_vecs_l, n_query, query_n_vecs, vec_dim);
#else
            spdlog::info("use CPU");
            cpu_ = CPUComputeScore(query_vecs_l, n_query, query_n_vecs, vec_dim);
#endif

        }

        void computeItemScore(const float** item_vecs_l,
                              const uint32_t* item_n_vecs_l, const uint32_t &n_item,
                              float *const distance_l) {

#ifdef USE_GPU
            gpu_.computeItemScore(item_vecs_l, item_n_vecs_l, n_item, distance_l);
#else
            cpu_.computeItemScore(item_vecs_l, item_n_vecs_l, n_item, distance_l);
#endif

        }

        void finishCompute() {
#ifdef USE_GPU
            gpu_.finishCompute();
#else
            cpu_.finishCompute();
#endif
        }
    };

}
#endif //VECTORSETSEARCH_COMPUTESCORE_HPP
