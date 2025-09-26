//
// Created by Administrator on 2025/1/17.
//

#ifndef IGPPROXIMITYGRAPHPROBE_HPP
#define IGPPROXIMITYGRAPHPROBE_HPP

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/struct/BoolArray.hpp"
#include "include/alg/probe/IGP/IGPStruct.hpp"
#include "include/alg/probe/ProbeHeap.hpp"
#include "include/alg/MatrixMulBLAS.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch
{
    class IGPProximityGraphProbe
    {
        // the input of this class is the centroid and the query
        // the output is the probe element when the function are called
    public:
        uint32_t _query_n_vecs, _n_centroid, _vec_dim;

        const float* _centroid_l; // n_centroid * vec_dim
        const float* _query; // query_n_vecs * vec_dim

        std::unique_ptr<hnswlib_probe::InnerProductSpace> _ip_space;
        std::unique_ptr<hnswlib_probe::HierarchicalNSW<float>> _hnsw_index;
        std::vector<uint32_t> _n_sort_l; // query_n_vec, # sorted element for each query vector, default value: 0
        std::vector<uint32_t> _current_nprobe_l;
        // query_n_vec, # element we have probed for each query vector, default value: 0

        IGPProximityGraphProbe() = default;

        IGPProximityGraphProbe(const float* centroid_l,
                               const uint32_t n_centroid,
                               const uint32_t vec_dim
        )
        {
            this->_centroid_l = centroid_l;
            this->_n_centroid = n_centroid;
            this->_vec_dim = vec_dim;
        }

        void build_hnsw()
        {
            const size_t max_element = _n_centroid;
            const uint32_t M = 32;
            const uint32_t efConstruction = 200;
            _ip_space = std::make_unique<hnswlib_probe::InnerProductSpace>(_vec_dim);
            _hnsw_index = std::make_unique<hnswlib_probe::HierarchicalNSW<float>>(_ip_space.get(),
                max_element, M, efConstruction);

            if (_hnsw_index->cur_element_count == 0)
            {
                const uint32_t centID = 0;
                const float* centroid = _centroid_l + centID * _vec_dim;
                _hnsw_index->addPoint((void*)centroid, centID);
            }

            // #pragma omp parallel for ordered default(none) shared(centroid_l, vec_dim)
            for (uint32_t centID = 1; centID < _n_centroid; centID++)
            {
                const float* centroid = _centroid_l + centID * _vec_dim;
                _hnsw_index->addPoint((void*)centroid, centID);
            }
        }

        void save_hnsw(const std::string filename) {
            _hnsw_index->saveIndex(filename);
        }

        void load_hnsw(const std::string filename) {
            _ip_space = std::make_unique<hnswlib_probe::InnerProductSpace>(_vec_dim);
            _hnsw_index = std::make_unique<hnswlib_probe::HierarchicalNSW<float>>(
                _ip_space.get(),
                filename, _n_centroid);
        }

        void set_query_info(const uint32_t query_n_vecs)
        {
            _hnsw_index->set_query_info(query_n_vecs, _vec_dim);
            this->_query_n_vecs = query_n_vecs;
            this->_current_nprobe_l.resize(query_n_vecs);
        }

        void set_retrieval_parameter(const uint32_t topk_per_batch, const uint32_t addition_search_size)
        {
            _hnsw_index->topk_per_batch_ = topk_per_batch;
            _hnsw_index->addition_search_size_ = addition_search_size;
        }

        void reset()
        {
            _hnsw_index->reset();

            _n_sort_l.assign(_query_n_vecs, 0);
            _current_nprobe_l.assign(_query_n_vecs, 0);
        }

        void set_query(const float* query)
        {
            _hnsw_index->set_query(query);

            this->_query = query;
        }

        void centroid_argsort(const uint32_t current_n_sort, const uint32_t qvecID,
                              uint32_t& n_computation) const
        {
            // find the topk of all query vectors
            // for every query, iteratively add the result to the sort_probe_ele_l
            _hnsw_index->search_topk(qvecID, n_computation);

#ifndef NDEBUG
            const uint32_t score_offset = qvecID * _n_centroid;

            const uint32_t n_sort = current_n_sort;
            const uint32_t n_sort_after = std::min(n_sort + _hnsw_index->topk_per_batch_, _n_centroid);
            if (!(n_sort_after <= _hnsw_index->finish_topk_l_[qvecID]))
            {
                printf("n_sort_after = %u, finish_topk_l_ %d\n", n_sort_after, _hnsw_index->finish_topk_l_[qvecID]);
            }
            assert(n_sort_after <= _hnsw_index->finish_topk_l_[qvecID]);

            for (uint32_t candID = 0; candID < n_sort_after - 1; candID++)
            {
                const float prev_score = _hnsw_index->sorted_probe_ele_l_[score_offset + candID].score;
                const float this_score = _hnsw_index->sorted_probe_ele_l_[score_offset + candID + 1].score;
                //                    assert(prev_score >= this_score);
            }
#endif
        }

        // ele_l is the output of this function
        // it should be the upper bound of all query score in the last probe
        void next_probe_element(const uint32_t nprobe_require, const uint32_t qvecID,
                                IGPProbeEle* ele_l, uint32_t& nprobe_return,
                                uint32_t& n_computation)
        {
            // add the probe element, may perform the argsort if necessary
            const uint32_t next_first_nprobe = _current_nprobe_l[qvecID] + nprobe_require;
            const uint32_t n_sort_target = std::min(next_first_nprobe, _n_centroid);
            while (_n_sort_l[qvecID] < n_sort_target)
            {
                centroid_argsort(_n_sort_l[qvecID], qvecID, n_computation);
                _n_sort_l[qvecID] += _hnsw_index->topk_per_batch_;
            }
            _n_sort_l[qvecID] = std::min(_n_sort_l[qvecID], _n_centroid);
            nprobe_return = std::min(nprobe_require, _n_centroid - _current_nprobe_l[qvecID]);


            for (uint32_t probeID = 0; probeID < nprobe_return; probeID++)
            {
                assert(_current_nprobe_l[qvecID] + probeID < _n_centroid);

                const uint32_t score_offset = qvecID * _n_centroid;
                const uint32_t centID = _hnsw_index->sorted_probe_ele_l_[
                    score_offset + _current_nprobe_l[qvecID] + probeID].centroidID;
                const float centroid_score = _hnsw_index->sorted_probe_ele_l_[
                    score_offset + _current_nprobe_l[qvecID] + probeID].score;

                ele_l[probeID] = IGPProbeEle(qvecID, centID, centroid_score);
            }

#ifndef NDEBUG
            for (uint32_t probeID = 0;
                 probeID < std::min(nprobe_require, _n_centroid - _current_nprobe_l[qvecID]) - 1; probeID++)
            {
                const IGPProbeEle probe = ele_l[probeID];
                assert(probe.qvecID == qvecID);

                assert(ele_l[probeID].score >=
                    ele_l[(probeID + 1)].score);
            }
#endif
            _current_nprobe_l[qvecID] = std::min(_current_nprobe_l[qvecID] + nprobe_require, _n_centroid);
        }
    };
}
#endif //IGPPROXIMITYGRAPHPROBE_HPP
