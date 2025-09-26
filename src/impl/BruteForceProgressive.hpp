//
// Created by bianzheng on 2023/7/18.
//

#ifndef VECTORSETSEARCH_BRUTEFORCEPROGRESSIVE_HPP
#define VECTORSETSEARCH_BRUTEFORCEPROGRESSIVE_HPP

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
#include "include/compute_score/ComputeScore.hpp"

namespace VectorSetSearch::Method {

    class BruteForceProgressive {
    private:
        uint32_t _n_query{}, _query_n_vecs{}, _vec_dim{};
        ComputeScore cs_;

    public:
        BruteForceProgressive() = default;

        BruteForceProgressive(const vector_set_list &qry_embeddings,
                              const uint32_t &vec_dim) : _vec_dim(vec_dim) {

            auto query_buffer = qry_embeddings.request();
            if (query_buffer.ndim != 3)
                throw std::runtime_error("the vector set list dimension should be 3");
            if (query_buffer.shape[2] != _vec_dim)
                throw std::runtime_error("the embedding dimension should equal to the pre-assignment");

            const uint32_t n_query = query_buffer.shape[0];
            const uint32_t query_n_vecs = query_buffer.shape[1];
            assert(query_buffer.shape[2] == vec_dim);
            this->_n_query = n_query;
            this->_query_n_vecs = query_n_vecs;
            this->_vec_dim = vec_dim;

            const float *query_start_ptr = static_cast<float *>(query_buffer.ptr);
            cs_ = ComputeScore(query_start_ptr, n_query, query_n_vecs, _vec_dim);
        }

        void convertItem(const std::vector<vector_set> &item_embeddings_l,
                         const std::vector<uint32_t> &externalID_l,
                         const float **item_vecs_l,
                         uint32_t *item_n_vecs_l,
                         uint32_t *itemID2externalID_l) {

            const size_t n_item = item_embeddings_l.size();
            if (item_embeddings_l.size() != externalID_l.size())
                throw std::runtime_error("the item embedding size does not equal to the externalID size");

//#pragma omp parallel for default(none) shared(_vec_dim, n_item, item_embeddings_l, item_vecs_l, item_n_vecs_l, externalID_l, itemID2externalID_l)
//            for (uint32_t itemID = 0; itemID < n_item; itemID++) {
//                auto buffer = item_embeddings_l[itemID].request();
////                if (buffer.ndim != 2) {
////                    throw std::runtime_error("the item embeddings' should be 2 dimension");
////                }
////                if (buffer.shape[1] != _vec_dim) {
////                    throw std::runtime_error("the item embeddings dimenision should be equal to the preassign");
////                }
//                const float *item_data = static_cast<float *>(buffer.ptr);
//                item_vecs_l[itemID] = item_data;
//
//                const uint32_t item_n_vecs = buffer.shape[0];
//                item_n_vecs_l[itemID] = item_n_vecs;
//            }

            for (uint32_t itemID = 0; itemID < n_item; itemID++) {
                vector_set vecs_set = item_embeddings_l[itemID];
                if (vecs_set.ndim() != 2) {
                    throw std::runtime_error("the item embeddings' should be 2 dimension");
                }
                if (vecs_set.shape(1) != _vec_dim) {
                    throw std::runtime_error("the item embeddings dimenision should be equal to the preassign");
                }
            }

#pragma omp parallel for default(none) shared(std::wcout, _vec_dim, n_item, item_embeddings_l, item_vecs_l, item_n_vecs_l, externalID_l, itemID2externalID_l)
            for (uint32_t itemID = 0; itemID < n_item; itemID++) {
                vector_set vecs_set = item_embeddings_l[itemID];

                const float *item_data = vecs_set.data();
                item_vecs_l[itemID] = item_data;

                const uint32_t item_n_vecs = vecs_set.shape(0);;
                item_n_vecs_l[itemID] = item_n_vecs;
            }

            std::memcpy(itemID2externalID_l, externalID_l.data(), sizeof(uint32_t) * n_item);

        }

        uint32_t _n_item = 0;
        float *_overall_distance_ptr = nullptr;
        uint32_t *_itemID2externalID_ptr = nullptr;

        void computeQueryItemScore(const std::vector<vector_set> &item_embeddings_l,
                                   const std::vector<uint32_t> &externalID_l) {
            assert(item_embeddings_l.size() == externalID_l.size());
            const uint32_t n_item = item_embeddings_l.size();

            TimeRecord record;
            record.reset();
            _n_item = n_item;
            if (_overall_distance_ptr != nullptr) {
                delete[] _overall_distance_ptr;
                _overall_distance_ptr = nullptr;
            }
            if (_itemID2externalID_ptr != nullptr) {
                delete[] _itemID2externalID_ptr;
                _itemID2externalID_ptr = nullptr;
            }
            _overall_distance_ptr = new float[(uint64_t) _n_query * _n_item];
            _itemID2externalID_ptr = new uint32_t[_n_item];
//            const double compute_score_init_time = record.get_elapsed_time_second();
//            spdlog::info("compute score init time {}s", compute_score_init_time);

//            record.reset();
            std::vector<const float *> item_vecs_l(_n_item);
            std::vector<uint32_t> item_n_vecs_l(_n_item);
            convertItem(item_embeddings_l, externalID_l,
                        item_vecs_l.data(), item_n_vecs_l.data(), _itemID2externalID_ptr);
//            const double convert_time = record.get_elapsed_time_second();
//            spdlog::info("convert time {}s", convert_time);

            record.reset();
            cs_.computeItemScore(item_vecs_l.data(), item_n_vecs_l.data(), _n_item, _overall_distance_ptr);
            const double compute_score_time = record.get_elapsed_time_second();
            spdlog::info("compute score {}s", compute_score_time);
        }

        void
        findTopkInScoreList(const float *batch_distance_l, float *batch_result_score_l, uint32_t *batch_result_itemID_l,
                            const uint32_t &topk) {

#pragma omp parallel for default(none) shared(_n_query, batch_distance_l, topk, batch_result_score_l, batch_result_itemID_l)
            for (int queryID = 0; queryID < _n_query; queryID++) {
                const float *score_l = batch_distance_l + (int64_t) queryID * _n_item;

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
                    const uint32_t append_idx = (uint32_t) queryID * topk + pop_count;
                    batch_result_score_l[append_idx] = pair.first;
                    batch_result_itemID_l[append_idx] = _itemID2externalID_ptr[pair.second];
                    pop_count++;
                }
//                const int report_every = 100;
//                if (queryID % report_every == 0) {
//                    std::cout << "preprocessed " << queryID / (0.01 * _n_query) << "% query, Mem: "
//                              << get_current_RSS() / 1000000 << " Mb, " << process_record.get_elapsed_time_second()
//                              << " s/iter" << std::endl;
//                }
                if (topk <= _n_item) {
                    assert(pop_count == topk);
                }

            }

        }

        py::tuple searchKNN(const uint32_t &topk) {

            if (topk > _n_item) {
                spdlog::warn("topk is larger than # item, just return all of the item");
            }

            float *result_score_l = new float[_n_query * topk];
            uint32_t *result_ID_l = new uint32_t[_n_query * topk];
            for (uint32_t ID = 0; ID < _n_query * topk; ID++) {
                result_score_l[ID] = -std::numeric_limits<float>::max();
                result_ID_l[ID] = -1;
            }

//            TimeRecord operation_record;
//            operation_record.reset();
            findTopkInScoreList(_overall_distance_ptr,
                                result_score_l, result_ID_l,
                                topk);
//            const double find_topk = operation_record.get_elapsed_time_second();
//            spdlog::info("find topk {}, time {}s", topk, find_topk);

            py::capsule handle_float_ptr(result_score_l, [](void *ptr) {
                float *tmp_ptr = static_cast<float *>(ptr);
                delete[] tmp_ptr;
            });

            py::capsule handle_int32_ptr(result_ID_l, [](void *ptr) {
                uint32_t *tmp_ptr = static_cast<uint32_t *>(ptr);
                delete[] tmp_ptr;
            });

            return py::make_tuple(
                    py::array_t<float>(
                            {_n_query, topk},
                            {topk * sizeof(float), sizeof(float)},
                            result_score_l,
                            handle_float_ptr
                    ),
                    py::array_t<uint32_t>(
                            {_n_query, topk},
                            {topk * sizeof(uint32_t), sizeof(uint32_t)},
                            result_ID_l,
                            handle_int32_ptr
                    )
            );
        }

        void finishCompute() {
            cs_.finishCompute();
        }

        using distance_list = py::array_t<float, py::array::c_style | py::array::forcecast>;
        using ID_list = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

        static py::tuple mergeResult(const distance_list &final_distance_l, const ID_list &final_ID_l,
                                     const distance_list &distance_l, const ID_list &ID_l) {
            if (final_distance_l.ndim() == 1) {
                assert(final_ID_l.ndim() == 1);
                return py::make_tuple(
                        distance_l, ID_l
                );
            }
            auto final_distance_l_buf = final_distance_l.request();
            auto final_ID_l_buf = final_ID_l.request();
            auto distance_l_buf = distance_l.request();
            auto ID_l_buf = ID_l.request();
            assert(final_distance_l_buf.ndim == 2 && final_ID_l_buf.ndim == 2 && distance_l_buf.ndim == 2 &&
                   ID_l_buf.ndim == 2);
            assert(final_distance_l_buf.shape[0] == final_ID_l_buf.shape[0] &&
                   distance_l_buf.shape[0] == ID_l_buf.shape[0] &&
                   final_distance_l_buf.shape[0] == distance_l_buf.shape[0]);
            assert(final_distance_l_buf.shape[1] == final_ID_l_buf.shape[1] &&
                   distance_l_buf.shape[1] == ID_l_buf.shape[1] &&
                   final_distance_l_buf.shape[1] == distance_l_buf.shape[1]);
            const uint32_t n_query = final_distance_l_buf.shape[0];
            const uint32_t topk = final_distance_l_buf.shape[1];

            float *result_dist_l = new float[(int64_t) n_query * topk];
            uint32_t *result_ID_l = new uint32_t[(int64_t) n_query * topk];

            const float *final_distance_l_ptr = static_cast<float *>(final_distance_l_buf.ptr);
            const uint32_t *final_ID_l_ptr = static_cast<uint32_t *>(final_ID_l_buf.ptr);
            const float *distance_l_ptr = static_cast<float *>(distance_l_buf.ptr);
            const uint32_t *ID_l_ptr = static_cast<uint32_t *>(ID_l_buf.ptr);

#pragma omp parallel for default(none) shared(n_query, final_distance_l_ptr, final_ID_l_ptr, distance_l_ptr, ID_l_ptr, result_dist_l, result_ID_l, topk)
            for (uint32_t queryID = 0; queryID < n_query; queryID++) {
                const float *tmp_final_distance_l_ptr = final_distance_l_ptr + (int64_t) queryID * topk;
                const uint32_t *tmp_final_ID_l_ptr = final_ID_l_ptr + (int64_t) queryID * topk;
                const float *tmp_distance_l_ptr = distance_l_ptr + (int64_t) queryID * topk;
                const uint32_t *tmp_ID_l_ptr = ID_l_ptr + (int64_t) queryID * topk;

                //min heap
                struct pairGreater {
                    bool operator()(const std::pair<float, uint32_t> l, const std::pair<float, uint32_t> r) const {
                        return l.first > r.first;
                    }
                };

                std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, pairGreater> min_heap;

                for (uint32_t topkID = 0; topkID < topk; topkID++) {
                    min_heap.emplace(tmp_final_distance_l_ptr[topkID], tmp_final_ID_l_ptr[topkID]);
                }
                for (uint32_t topkID = 0; topkID < topk; topkID++) {
                    const float eval_distance = tmp_distance_l_ptr[topkID];
                    const uint32_t eval_ID = tmp_ID_l_ptr[topkID];
                    if (min_heap.size() == topk) {
                        std::pair<float, uint32_t> topk_pair = min_heap.top();

                        if (eval_distance > topk_pair.first) {
                            min_heap.pop();
                            min_heap.emplace(eval_distance, eval_ID);
                        }
                    } else {
                        min_heap.emplace(eval_distance, eval_ID);
                    }
                }
                assert(min_heap.size() == topk);

                const uint32_t query_offset = queryID * topk;
                for (uint32_t topkID = 0; topkID < topk; topkID++) {
                    const std::pair<float, uint32_t> pair = min_heap.top();
                    result_dist_l[query_offset + topkID] = pair.first;
                    result_ID_l[query_offset + topkID] = pair.second;
                    min_heap.pop();
                }

            }

            py::capsule handle_float_ptr(result_dist_l, [](void *ptr) {
                float *tmp_ptr = static_cast<float *>(ptr);
                delete[] tmp_ptr;
            });

            py::capsule handle_int32_ptr(result_ID_l, [](void *ptr) {
                uint32_t *tmp_ptr = static_cast<uint32_t *>(ptr);
                delete[] tmp_ptr;
            });

            return py::make_tuple(
                    py::array_t<float>(
                            {n_query, topk},
                            {topk * sizeof(float), sizeof(float)},
                            result_dist_l,
                            handle_float_ptr
                    ),
                    py::array_t<uint32_t>(
                            {n_query, topk},
                            {topk * sizeof(uint32_t), sizeof(uint32_t)},
                            result_ID_l,
                            handle_int32_ptr
                    )
            );
        }

    };
}
#endif //VECTORSETSEARCH_BRUTEFORCEPROGRESSIVE_HPP
