//
// Created by bianzheng on 2023/2/16.
//

#ifndef VECTORSETSEARCH_BRUTEFORCE_HPP
#define VECTORSETSEARCH_BRUTEFORCE_HPP

#include <stdexcept>
#include <cstdint>
#include <Eigen/Eigen>
#include <limits>
#include <queue>
#include <iostream>
#include <mutex>

#include "include/struct/MethodBase.hpp"
#include "include/alg/Distance.hpp"
#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch::Method {

    class BruteForce : public BaseMethodIndex {
    private:
        uint32_t _n_item{}, _vec_dim{}, _current_size{};
        std::vector<std::unique_ptr<float[]>> _item_vecs_l;
        std::vector<uint32_t> _item_n_vecs_l;
        std::vector<uint32_t> _item_id_to_external_id;
        uint32_t _n_thread;

    public:
        BruteForce() = default;

        BruteForce(const uint32_t &n_item, const uint32_t &vec_dim) : _n_item(n_item), _vec_dim(vec_dim),
                                                                      _current_size(0) {
            _item_vecs_l.resize(_n_item);
            _item_n_vecs_l.resize(_n_item);
            _item_id_to_external_id.resize(_n_item);
        }

        bool addItem(const vector_set &item_embeddings,
                     const uint32_t &externalID) {
            const auto buffer = item_embeddings.request();
            if (buffer.ndim != 2) {
                throw std::runtime_error("the item embeddings' should be 2 dimension");
            }
            if (buffer.shape[1] != _vec_dim) {
                throw std::runtime_error("the item embeddings dimenision should be equal to the preassign");
            }

            uint32_t num_vectors = buffer.shape[0];
            _item_n_vecs_l[_current_size] = num_vectors;
            const float *item_data = static_cast<float *>(buffer.ptr);

            std::unique_ptr<float[]> item_ptr = std::make_unique<float[]>(num_vectors * _vec_dim);
            std::memcpy(item_ptr.get(), item_data, sizeof(float) * num_vectors * _vec_dim);
            _item_vecs_l[_current_size] = std::move(item_ptr);

            _item_id_to_external_id[_current_size] = externalID;

            _current_size++;

            if (_current_size > _n_item)
                throw std::runtime_error("the current size is larger than the total number of document");

            return true;
        }

        bool addItemBatch(const std::vector<vector_set> &item_embeddings_l,
                          const std::vector<uint32_t> &externalID_l) override {
            const size_t n_item_batch = item_embeddings_l.size();
            if (item_embeddings_l.size() != externalID_l.size())
                throw std::runtime_error("the item embedding size is not euqal to the externalID size");

            std::vector<size_t> item_n_vecs_l(n_item_batch);
            std::vector<std::unique_ptr<float[]>> item_data_l(n_item_batch);
            for (uint32_t itemID = 0; itemID < n_item_batch; itemID++) {
                const vector_set &item_embeddings = item_embeddings_l[itemID];
                auto buffer = item_embeddings.request();
                if (buffer.ndim != 2) {
                    throw std::runtime_error("the item embeddings' should be 2 dimension");
                }
                if (buffer.shape[1] != _vec_dim) {
                    throw std::runtime_error("the item embeddings dimenision should be equal to the preassign");
                }
                const uint32_t item_n_vecs = buffer.shape[0];
                item_n_vecs_l[itemID] = item_n_vecs;

                const float *item_data = static_cast<float *>(buffer.ptr);
                std::unique_ptr<float[]> item_ptr = std::make_unique<float[]>(item_n_vecs * _vec_dim);
                std::memcpy(item_ptr.get(), item_data, sizeof(float) * item_n_vecs * _vec_dim);
                item_data_l[itemID] = std::move(item_ptr);
            }

#pragma omp parallel for default(none) shared(n_item_batch, item_n_vecs_l, item_data_l, externalID_l)
            for (uint32_t itemID = 0; itemID < n_item_batch; itemID++) {
                const size_t item_n_vecs = item_n_vecs_l[itemID];
                const uint32_t externalID = externalID_l[itemID];

                const uint32_t current_size = _current_size + itemID;
                _item_n_vecs_l[current_size] = item_n_vecs;

                _item_vecs_l[current_size] = std::move(item_data_l[itemID]);

                _item_id_to_external_id[current_size] = externalID;

                if (current_size > _n_item)
                    throw std::runtime_error("the current size is larger than the total number of document");
            }

            _current_size += n_item_batch;

            return true;
        }

        void setNumThread(const uint32_t& n_thread){
            this->_n_thread = n_thread;
        }

        py::tuple searchKNN(const vector_set_list &qry_embeddings,
                            const uint32_t &topk) override {
            auto buffer = qry_embeddings.request();
            if (buffer.ndim != 3)
                throw std::runtime_error("the vector set list dimension should be 3");
            if (buffer.shape[2] != _vec_dim)
                throw std::runtime_error("the embedding dimension should equal to the pre-assignment");
            if (topk > _n_item)
                throw std::runtime_error("topk is larger than the number of item");
            if (_current_size != _n_item)
                throw std::runtime_error("the current size is not equal to the number of item");

            const uint32_t n_query = buffer.shape[0];

            const float *query_start_ptr = static_cast<float *>(buffer.ptr);

            float *result_score_l = new float[n_query * topk];
            uint32_t *result_ID_l = new uint32_t[n_query * topk];
            double *compute_time_l = new double[n_query];
            uint64_t *n_IP_compute_l = new uint64_t[n_query];
            const uint32_t query_n_vecs = buffer.shape[1];

            TimeRecord process_record;
            for (uint32_t queryID = 0; queryID < n_query; queryID++) {
                process_record.reset();

                const float *query_ptr = query_start_ptr + queryID * query_n_vecs * _vec_dim;
                std::vector<float> score_l(_n_item);
                std::vector<uint32_t> itemIP_compute_l(_n_item);

#pragma omp parallel for default(none) shared(query_ptr, query_n_vecs, score_l, topk, itemIP_compute_l) num_threads(_n_thread)
                for (uint32_t itemID = 0; itemID < _n_item; itemID++) {
                    const float *item_ptr = _item_vecs_l[itemID].get();
                    const uint32_t item_n_vecs = _item_n_vecs_l[itemID];

                    const float eval_score = vectorSetDistance(query_ptr, query_n_vecs, item_ptr, item_n_vecs,
                                                               _vec_dim);
                    score_l[itemID] = eval_score;
                    itemIP_compute_l[itemID] = query_n_vecs * item_n_vecs;
                }

                //min heap
                struct pairGreater {
                    bool operator()(const std::pair<float, uint32_t> l, const std::pair<float, uint32_t> r) const {
                        return l.first > r.first;
                    }
                };
                std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, pairGreater> min_heap;

                for (uint32_t itemID = 0; itemID < _n_item; itemID++) {
                    const float eval_score = score_l[itemID];
                    if (min_heap.size() == topk) {
                        std::pair<float, uint32_t> topk_pair = min_heap.top();

                        if (eval_score > topk_pair.first) {
                            min_heap.pop();
                            min_heap.emplace(eval_score, itemID);
                        }
                    } else {
                        min_heap.emplace(eval_score, itemID);
                    }
                }

                int pop_count = 0;
                while (not min_heap.empty()) {
                    const std::pair<float, uint32_t> pair = min_heap.top();
                    min_heap.pop();
                    const uint32_t append_idx = (queryID + 1) * topk - pop_count - 1;
                    result_score_l[append_idx] = pair.first;
                    result_ID_l[append_idx] = _item_id_to_external_id[pair.second];
                    pop_count++;
                }
//                const int report_every = 100;
//                if (queryID % report_every == 0) {
//                    std::cout << "preprocessed " << queryID / (0.01 * n_query) << "% query, Mem: "
//                              << get_current_RSS() / 1000000 << " Mb, " << process_record.get_elapsed_time_second()
//                              << " s/iter" << std::endl;
//                }
                assert(pop_count == topk);
                compute_time_l[queryID] = process_record.get_elapsed_time_second();
                n_IP_compute_l[queryID] = std::accumulate(itemIP_compute_l.begin(), itemIP_compute_l.end(), 0u);
            }

            py::capsule capsule_result_score_l(result_score_l, PtrDelete<float>);

            py::capsule capsule_result_ID_l(result_ID_l, PtrDelete<uint32_t>);

            py::capsule capsule_compute_time_l(compute_time_l, PtrDelete<double>);

            py::capsule capsule_n_IP_compute_l(n_IP_compute_l, PtrDelete<uint64_t>);

            return py::make_tuple(
                    py::array_t<float>(
                            {n_query, topk},
                            {topk * sizeof(float), sizeof(float)},
                            result_score_l,
                            capsule_result_score_l
                    ),
                    py::array_t<uint32_t>(
                            {n_query, topk},
                            {topk * sizeof(uint32_t), sizeof(uint32_t)},
                            result_ID_l,
                            capsule_result_ID_l
                    ),
                    py::array_t<double>(
                            {n_query},
                            {sizeof(double)},
                            compute_time_l,
                            capsule_compute_time_l
                    ),
                    py::array_t<uint64_t>(
                            {n_query},
                            {sizeof(uint64_t)},
                            n_IP_compute_l,
                            capsule_n_IP_compute_l
                    )
            );
        }

    };
}
#endif //VECTORSETSEARCH_BRUTEFORCE_HPP
