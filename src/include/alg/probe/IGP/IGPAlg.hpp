//
// Created by Administrator on 2025/1/17.
//

#ifndef IGPALG_HPP
#define IGPALG_HPP

#include <queue>
#include <numeric>
#include <set>
#include <complex>

#include "include/struct/BoolArray.hpp"
#include "include/alg/probe/IGP/IGPProbeController.hpp"
#include "include/alg/probe/IGP/IGPStruct.hpp"

#include "include/util/TimeMemory.hpp"
#include "include/util/util.hpp"

namespace VectorSetSearch
{
    class IGPAlg
    {
    public:
        // know in build index
        const std::vector<uint32_t>* _centroid2itemID_l; // n_centroid
        const float* _centroid_l; // n_centroid * vec_dim
        uint32_t _n_item;
        size_t _n_vecs;
        uint32_t _n_centroid, _vec_dim;
        // know before retrieval
        uint32_t _query_n_vecs;

        // data structure for sorting the array
        // know in build index
        IGPProbeController _probe_controller;
        constexpr static uint32_t _nprobe_per_batch = 1; // means how many rows are probed for each batch
        // know before retrieval
        // caching the sorted probe element
        std::vector<IGPProbeEle> _probe_ele_l; // _nprobe_per_batch

        // data structure used for probing
        // know in build index
        // store the lower bound of each item
        std::vector<float> _item_appr_scr_l; // _n_item
        // store how many query vector are first visited for each item
        std::vector<uint32_t> _qvec_n_refine_l; // _n_item, default value: 0
        // know before retrieval
        // min heap that stores the item with top-k score
        IGPTopkItemMinHeap _topk_score_q; // max_size: probe_topk
        // store whether the query vector is visited for each item
        std::vector<char> _visit_query_l; // _query_n_vecs * _n_item, default value: false

        // input parameter
        uint32_t _probe_topk{};
        size_t _n_neighborhood_fetch{};
        std::vector<size_t> _n_fetch_l{}; // query_n__vec, default value: 0

        TimeRecord _record;
        // output indicator
        uint32_t _n_sorted_score;
        uint32_t _n_seen_item, _n_refine_item;
        size_t _n_vq_score_refine, _n_vq_score_linear_scan;

        IGPAlg() = default;

        IGPAlg(const std::vector<uint32_t>* centroid2itemID_l,
               const float* centroid_l,
               const uint32_t n_item, const size_t n_vecs,
               const uint32_t n_centroid, const uint32_t vec_dim
        )
        {
            this->_centroid2itemID_l = centroid2itemID_l;
            this->_centroid_l = centroid_l;
            this->_n_item = n_item;
            this->_n_vecs = n_vecs;
            this->_n_centroid = n_centroid;
            this->_vec_dim = vec_dim;

            this->_probe_controller = IGPProbeController(centroid_l, n_centroid, vec_dim);

            this->_item_appr_scr_l.resize(n_item);
            this->_qvec_n_refine_l.resize(n_item, 0);
        }

        void build_graph_index()
        {
            _probe_controller._pg_probe.build_hnsw();
        }

        void save_graph_index(const std::string index_filename)
        {
            _probe_controller._pg_probe.save_hnsw(index_filename);
        }

        void load_graph_index(const std::string index_filename)
        {
            _probe_controller._pg_probe.load_hnsw(index_filename);
        }

        void set_retrieval_parameter(
            const uint32_t query_n_vecs, const uint32_t topk,
            const size_t n_neighborhood_fetch, const uint32_t probe_topk,
            const uint32_t topk_per_batch, const uint32_t addition_search_size
        )
        {
            this->_query_n_vecs = query_n_vecs;

            this->_probe_controller.set_retrieval_parameter(topk_per_batch, addition_search_size);
            this->_probe_controller.set_query_info(query_n_vecs);

            this->_probe_ele_l.resize(_nprobe_per_batch);

            this->_topk_score_q = IGPTopkItemMinHeap(probe_topk);
            this->_visit_query_l = std::vector<char>((size_t)query_n_vecs * _n_item, false);

            this->_n_neighborhood_fetch = n_neighborhood_fetch;
            this->_probe_topk = probe_topk;
            assert(0 <= _probe_topk && _probe_topk <= _n_item);
        }

        void reset()
        {
            this->_probe_controller.reset();

            _qvec_n_refine_l.assign(_n_item, 0);
            _visit_query_l.assign((size_t)_query_n_vecs * _n_item, false);

            _n_sorted_score = 0;
            _n_seen_item = 0;
            _n_refine_item = 0;
            _n_vq_score_refine = 0;
            _n_vq_score_linear_scan = 0;
        }

        size_t compute_lower_bound(uint32_t& actual_nprobe)
        {
            size_t n_fetch = 0;
            const uint32_t nprobe_qvec = actual_nprobe;
            assert(nprobe_qvec <= _probe_ele_l.size());
            for (uint32_t probeID = 0; probeID < nprobe_qvec; probeID++)
            {
                const IGPProbeEle probe = _probe_ele_l[probeID];
                const uint32_t qvecID = probe.qvecID;
                const uint32_t centID = probe.centroidID;
                const float centroid_score = probe.score;
                /*
                 * the global minimum score of each query-centroid score is -1 because of unit norm
                 * we add 1 to all query-centroid score, this makes the minimum query-centroid score as 0
                 */

                n_fetch += _centroid2itemID_l[centID].size();
                const float score_offset = centroid_score + 1.0f;
                // refine the score of that query centroid pair
                for (uint32_t itemID : _centroid2itemID_l[centID])
                {
                    _n_vq_score_linear_scan++;
                    if (_visit_query_l[(size_t)qvecID * _n_item + itemID])
                    {
                        continue;
                    }
                    _n_vq_score_refine++;
                    _visit_query_l[(size_t)qvecID * _n_item + itemID] = true;
                    float item_scr;
                    if (_qvec_n_refine_l[itemID] == 0)
                    {
                        _n_seen_item++;
                        item_scr = score_offset;
                    }
                    else
                    {
                        const float old_item_scr = _item_appr_scr_l[itemID];
                        item_scr = old_item_scr + score_offset;
                    }
                    _n_refine_item++;
                    assert(0 <= item_scr);
                    _qvec_n_refine_l[itemID]++;
                    assert(_qvec_n_refine_l[itemID] <= _query_n_vecs);
                    _item_appr_scr_l[itemID] = item_scr;
                }
            }
            return n_fetch;
        }

        void compute_topk_lower_bound()
        {
            _topk_score_q.Reset();

            for (uint32_t itemID = 0; itemID < _n_item; itemID++)
            {
                if (_qvec_n_refine_l[itemID] == 0)
                {
                    continue;
                }

                const float lb_score = _item_appr_scr_l[itemID];
                const uint32_t n_qvec_not_refine = _query_n_vecs - _qvec_n_refine_l[itemID];

                _topk_score_q.Update(lb_score, itemID, n_qvec_not_refine);
            }
        }

        void refine(const float* query,
                    std::pair<float, uint32_t>* item_candidate_cache_l,
                    uint32_t& n_filter_item, uint32_t& n_compute_score,

                    const uint32_t queryID,
                    const uint32_t* vq_code_l,
                    const uint32_t* item_n_vec_l,
                    const size_t* item_n_vec_offset_l)
        {
            this->_probe_controller.set_query(query);

            for (uint32_t qvecID = 0; qvecID < _query_n_vecs; qvecID++)
            {
                size_t current_fetch = 0;
                while (current_fetch < _n_neighborhood_fetch)
                {
                    uint32_t actual_nprobe;
                    _probe_controller.next_probe_element(_nprobe_per_batch, qvecID,
                                                         _probe_ele_l.data(), actual_nprobe);
                    const size_t n_fetch = compute_lower_bound(actual_nprobe);
                    current_fetch += n_fetch;
                }
            }
            n_compute_score += _probe_controller._n_computation;

            compute_topk_lower_bound();

            // compute the candidates
            assert(_topk_score_q.Size() == _probe_topk);
            const ItemProbeEle* topk_lb_l = _topk_score_q.Data();
            for (uint32_t candID = 0; candID < _topk_score_q.Size(); candID++)
            {
                //                if(queryID == 78){
                //                    spdlog::info("candidate queryID {}, score {:.3f}, itemID {}", queryID, topk_lb_l[candID].first, topk_lb_l[candID].second);
                //                }
                item_candidate_cache_l[candID] = std::make_pair(topk_lb_l[candID].score_,
                                                                topk_lb_l[candID].itemID_);
                assert(0 <= item_candidate_cache_l[candID].second &&
                    item_candidate_cache_l[candID].second < _n_item);
            }
            n_filter_item = _probe_topk;

            _n_sorted_score =
            (std::accumulate(_probe_controller._pg_probe._n_sort_l.begin(),
                             _probe_controller._pg_probe._n_sort_l.end(),
                             0) / _query_n_vecs + _probe_controller._mm_probe._n_sort) * _query_n_vecs;

#ifndef NDEBUG
            uint32_t n_refine_query_item = 0;
            for (uint32_t itemID = 0; itemID < _n_item; itemID++)
            {
                n_refine_query_item += _qvec_n_refine_l[itemID];
            }
            assert(n_refine_query_item == _n_refine_item);
#endif
        }

        void saveIndex(const std::string filename)
        {
        }
    };
}
#endif //IGPALG_HPP
