#pragma once

#include "visited_list_pool.hpp"
#include "hnswlib.hpp"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <queue>


#define DEBUG_LIB 1

namespace hnswlib_probe
{
    template <typename T>
    void writeBinaryPOD(std::ostream& out, const T& podRef)
    {
        out.write((char*)&podRef, sizeof(T));
    }

    template <typename T>
    static void readBinaryPOD(std::istream& in, T& podRef)
    {
        in.read((char*)&podRef, sizeof(T));
    }

    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    class ProbeEle
    {
    public:
        uint32_t qvecID;
        uint32_t centroidID;
        float score;

        ProbeEle() = default;

        ProbeEle(const uint32_t qvecID, const uint32_t centroidID, const float score)
        {
            this->qvecID = qvecID;
            this->centroidID = centroidID;
            this->score = score;
        }
    };

    template <typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
    {
    public:
        HierarchicalNSW(SpaceInterface<dist_t>* s)
        {
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const string &location, const uint32_t max_ele): ll_locks(max_ele){
            LoadIndex(location, s);
        }

        HierarchicalNSW(SpaceInterface<dist_t>* s, size_t maxElements, size_t M, size_t efConstruction) :
            ll_locks(maxElements), elementLevels(maxElements)
        {
            maxelements_ = maxElements;

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            efConstruction_ = efConstruction;


            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;
            //cout << offsetData_ << "\t" << label_offset_ << "\n";
            //cout << size_links_level0_ << "\t" << data_size_ << "\t" << sizeof(labeltype) << "\n";

            data_level0_memory_ = (char*)malloc(maxelements_ * size_data_per_element_);

            size_t predicted_size_per_element = size_data_per_element_ + sizeof(void*) + 8 + 8 + 2 * 8;
            //cout << "size_mb=" << maxelements_*(predicted_size_per_element) / (1000 * 1000) << "\n";
            cur_element_count = 0;

            visitedlistpool = new VisitedListPool(1, maxElements);


            //initializations for special treatment of the first node
            enterpoint_node = -1;
            maxlevel_ = -1;

            linkLists_ = (char**)malloc(sizeof(void*) * maxelements_);
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        ~HierarchicalNSW()
        {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++)
            {
                if (elementLevels[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visitedlistpool;
        }

        size_t maxelements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t efConstruction_;
        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool* visitedlistpool;
        mutex cur_element_count_guard_;
        vector<mutex> ll_locks;
        tableint enterpoint_node;

        size_t dist_calc;
        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;


        char* data_level0_memory_;
        char** linkLists_;
        vector<int> elementLevels;


        size_t data_size_;
        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void* dist_func_param_;
        std::default_random_engine generator = std::default_random_engine(100);

        uint32_t topk_per_batch_ = 8; // constant
        uint32_t addition_search_size_ = 8; // the additional buffer size, constant
        std::vector<size_t> ef_l_; // query_n_vec, default value: 0
        std::vector<uint32_t> finish_topk_l_; // query_n_vec, default value: 0

        const float* query_;
        uint32_t query_n_vec_, vec_dim_;

        std::vector<SearchResultMinHeap> qvec_heap_l_; // query_n_vec, each size is a heap, internal element
        // n_centroid == maxelement
        std::vector<uint32_t> base_layer_ep_l_; // query_n_vec, internal element

        std::vector<char> visited_point_l_; // query_n_vec * maxelement, internal element
        std::vector<char> visited_out_neighbor_l_; // query_n_vec * maxelement, internal element
        std::vector<char> returned_point_l_; // query_n_vec * maxelement, internal element

        std::vector<float> score_l_; // query_n_vec * maxelement, internal element
        std::vector<ProbeEle> sorted_probe_ele_l_; // query_n_vecs * maxelement, external element

        void inline set_query_info(const uint32_t query_n_vec, const uint32_t vec_dim)
        {
            this->query_n_vec_ = query_n_vec;
            this->vec_dim_ = vec_dim;

            ef_l_.resize(query_n_vec, topk_per_batch_ + addition_search_size_);
            finish_topk_l_.resize(query_n_vec, 0);

            this->qvec_heap_l_.resize(query_n_vec);
            for (uint32_t qvecID = 0; qvecID < query_n_vec; qvecID++)
            {
                qvec_heap_l_[qvecID] = SearchResultMinHeap(ef_l_[qvecID], maxelements_);
            }
            this->base_layer_ep_l_.resize(query_n_vec);

            this->visited_point_l_.resize(query_n_vec * maxelements_, false);
            this->visited_out_neighbor_l_.resize(query_n_vec * maxelements_, false);
            this->returned_point_l_.resize(query_n_vec * maxelements_, false);

            this->score_l_.resize(query_n_vec * maxelements_);
            this->sorted_probe_ele_l_.resize(query_n_vec * maxelements_);
        }

        void set_parameter(const uint32_t topk_per_batch, const uint32_t addition_search_size)
        {
            this->topk_per_batch_ = topk_per_batch;
            this->addition_search_size_ = addition_search_size;
        }

        void reset()
        {
            finish_topk_l_.assign(query_n_vec_, 0);
            ef_l_.assign(query_n_vec_, topk_per_batch_ + addition_search_size_);

            for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
            {
                qvec_heap_l_[qvecID].Reset(ef_l_[qvecID]);
            }

            visited_point_l_.assign(query_n_vec_ * maxelements_, false);
            visited_out_neighbor_l_.assign(query_n_vec_ * maxelements_, false);
            returned_point_l_.assign(query_n_vec_ * maxelements_, false);
        }

        void set_query(const float* query)
        {
            this->query_ = query;
        }

        void search_topk(const uint32_t qvecID, uint32_t& n_computation)
        {
            const float* qvec = query_ + qvecID * vec_dim_;
            const uint32_t score_offset = qvecID * maxelements_;
            float* qvec_score_l = score_l_.data() + score_offset;

            char* qvec_visited_point_l = visited_point_l_.data() + score_offset;
            char* qvec_visited_out_neighbor_l = visited_out_neighbor_l_.data() + score_offset;
            char* qvec_returned_point_l = returned_point_l_.data() + score_offset;

            // printf("finish_topk_: %u, topk_per_batch_: %u\n", finish_topk_, topk_per_batch_);
            dist_calc = 0;
            if (finish_topk_l_[qvecID] == 0)
            {
                // means it is the first retrieval
                searchKnn((void*)qvec, qvecID, qvec_score_l,
                          qvec_visited_point_l, qvec_visited_out_neighbor_l);
            }
            else
            {
                // qvec_heap_l_[qvecID].Reset(ef_);
                // searchKnn((void *) qvec, qvecID, qvec_score_l, qvec_visited_point_l);
                const uint32_t base_layer_ep = base_layer_ep_l_[qvecID];
                searchBaseLayerSTCache(base_layer_ep, (void*)qvec, ef_l_[qvecID], qvecID, qvec_score_l,
                                       qvec_visited_point_l, qvec_visited_out_neighbor_l,
                                       qvec_returned_point_l);
            }
            n_computation += dist_calc;

            //                printf("prev qvecID %d, qvec_heap_l_[qvecID].Size() %d, finish_topk_ %d\n", qvecID,
            //                       qvec_heap_l_[qvecID].Size(), finish_topk_);

            // if (!(qvec_heap_l_[qvecID].Size() >= finish_topk_))
            // {
            //     printf("qvec_heap_l_[qvecID].Size() %d, max size %d, topk %d, ef %d\n", qvec_heap_l_[qvecID].Size(),
            //            qvec_heap_l_[qvecID].search_max_size_, finish_topk_, ef_);
            // }
            // assert(qvec_heap_l_[qvecID].Size() >= topk_per_batch_);
            const std::vector<SearchResultEle> ele_l = qvec_heap_l_[qvecID].PopTopScore(topk_per_batch_);
            const uint32_t n_cand = ele_l.size();
            assert(ele_l.size() <= topk_per_batch_);
            for (uint32_t candID = 0; candID < n_cand; candID++)
            {
                const float score = ele_l[candID].score_;
                const uint32_t centID = getExternalLabel(ele_l[candID].itemID_);
                assert(qvec_returned_point_l[ele_l[candID].itemID_] == false);
                qvec_returned_point_l[ele_l[candID].itemID_] = true;

                sorted_probe_ele_l_[score_offset + finish_topk_l_[qvecID] + candID] = ProbeEle(qvecID, centID, score);
            }

            //                printf("sorted_probe_ele_l_ size %ld\n", sorted_probe_ele_l_.size());

            //                for (uint32_t candID = 0; candID < ef_; candID++) {
            //                    if(qvecID== 0){
            //                        printf("qvecID %d, candID %d, itemID_ %d, score_ %.3f\n",
            //                               qvecID, candID, sorted_probe_ele_l_[score_offset + candID].centroidID, sorted_probe_ele_l_[score_offset + candID].score);
            //                    }
            //                }

#ifndef NDEBUG
            if (finish_topk_l_[qvecID] + topk_per_batch_ > maxelements_)
            {
                assert(n_cand == finish_topk_l_[qvecID] + topk_per_batch_ - maxelements_);
            }
            else
            {
                if (!(n_cand == topk_per_batch_))
                {
                    printf("n_cand %d, topk_per_batch %d, finish_topk %d\n", n_cand, topk_per_batch_,
                           finish_topk_l_[qvecID]);
                }
                assert(n_cand == topk_per_batch_);
            }
            for (uint32_t i = 0; i < n_cand - 1; i++)
            {
                assert(sorted_probe_ele_l_[score_offset + i].score >=
                    sorted_probe_ele_l_[score_offset + i + 1].score);
            }
#endif

            finish_topk_l_[qvecID] = std::min(finish_topk_l_[qvecID] + topk_per_batch_, (uint32_t)maxelements_);
            ef_l_[qvecID] = std::min(topk_per_batch_ + addition_search_size_,
                                     (uint32_t)maxelements_ - finish_topk_l_[qvecID]);
        }

        inline labeltype getExternalLabel(tableint internal_id)
        {
            return *((labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_));
        }

        inline labeltype* getExternalLabeLp(tableint internal_id)
        {
            return (labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char* getDataByInternalId(tableint internal_id)
        {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double revSize)
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(generator)) * revSize;
            //cout << revSize;
            return (int)r;
        }

        struct CompareMaxHeap
        {
            constexpr bool operator()(pair<dist_t, tableint> const& a,
                                      pair<dist_t, tableint> const& b) const noexcept
            {
                if (a.first == b.first)
                {
                    return a.second > b.second;
                }
                return a.first < b.first;
            }
        };

        struct CompareMinHeap
        {
            constexpr bool operator()(pair<dist_t, tableint> const& a,
                                      pair<dist_t, tableint> const& b) const noexcept
            {
                if (a.first == b.first)
                {
                    return a.second > b.second;
                }
                return a.first > b.first;
            }
        };

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareMinHeap>
        searchBaseLayerBuildIndex(tableint ep, void* datapoint, int layer)
        {
            VisitedList* vl = visitedlistpool->getFreeVisitedList();
            vl_type* massVisited = vl->mass;
            vl_type currentV = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareMinHeap>
                topResults;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareMaxHeap>
                candidateSet;
            dist_t dist = fstdistfunc_(datapoint, getDataByInternalId(ep), dist_func_param_);

            topResults.emplace(dist, ep);
            candidateSet.emplace(dist, ep);
            massVisited[ep] = currentV;
            dist_t lowerBound = dist;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

                if (topResults.size() == efConstruction_ && curr_el_pair.first < lowerBound)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                unique_lock<mutex> lock(ll_locks[curNodeNum]);

                int* data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                    data = (int*)(data_level0_memory_ + curNodeNum * size_data_per_element_ + offsetLevel0_);
                else
                    data = (int*)(linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                int size = *data;
                tableint* datal = (tableint*)(data + 1);
                _mm_prefetch((char *) (massVisited + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (massVisited + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);

                for (int j = 0; j < size; j++)
                {
                    tableint tnum = *(datal + j);
                    _mm_prefetch((char *) (massVisited + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                    if (!(massVisited[tnum] == currentV))
                    {
                        massVisited[tnum] = currentV;
                        char* currObj1 = (getDataByInternalId(tnum));

                        dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_);
                        if (topResults.top().first < dist || topResults.size() < efConstruction_)
                        {
                            assert(topResults.size() < efConstruction_ || dist >= lowerBound);
                            candidateSet.emplace(dist, tnum);
                            _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
                            topResults.emplace(dist, tnum);
                            if (topResults.size() > efConstruction_)
                            {
                                topResults.pop();
                            }
                            lowerBound = topResults.top().first;
                        }
                    }
                }
            }
            visitedlistpool->releaseVisitedList(vl);

            return topResults;
        }

        void searchBaseLayerSTCache(tableint ep, void* datapoint, size_t ef, const uint32_t qvecID,
                                    float* qvec_score_l,
                                    char* qvec_visited_point_l,
                                    char* qvec_visited_out_neighbor_l,
                                    char* qvec_returned_point_l
        )
        {
            //            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareMinHeap> topResults;
            SearchResultMinHeap& topResults = qvec_heap_l_[qvecID];
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareMaxHeap>
                candidateSet;

            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareMinHeap>
                pre_candidateSet;

            //#ifndef NDEBUG
            //            for (uint32_t itn_centID = 0; itn_centID < maxelements_; itn_centID++) {
            //                if (qvec_returned_point_l[itn_centID]) {
            //                    assert(qvec_visited_point_l[itn_centID] && qvec_visited_out_neighbor_l[itn_centID]);
            //                }
            //                if (qvec_visited_out_neighbor_l[itn_centID]) {
            //                    assert(qvec_visited_point_l[itn_centID]);
            //                }
            //                if (qvec_visited_point_l[itn_centID]) {
            //                    assert(qvec_visited_point_l[itn_centID]);
            //                } else {
            //                    if (!(!qvec_visited_point_l[itn_centID] && !qvec_visited_out_neighbor_l[itn_centID])) {
            //                        printf(
            //                                "qvecID %d, itn_centID %d, qvec_visited_point_l[itn_centID] %d, qvec_visited_out_neighbor_l[itn_centID] %d\n",
            //                                qvecID, itn_centID, qvec_visited_point_l[itn_centID],
            //                                qvec_visited_out_neighbor_l[itn_centID]);
            //                    }
            //                    assert(!qvec_visited_point_l[itn_centID] && !qvec_visited_out_neighbor_l[itn_centID] && !
            //                            qvec_returned_point_l[itn_centID]);
            //                }
            //            }
            //#endif

            for (uint32_t itn_centID = 0; itn_centID < maxelements_; itn_centID++)
            {
                if (qvec_visited_point_l[itn_centID] && (!topResults.InHeap(itn_centID)))
                {
                    if ((!qvec_visited_out_neighbor_l[itn_centID]))
                    {
                        float dist = qvec_score_l[itn_centID];
                        //                    candidateSet.emplace(dist, itn_centID);
                        if (pre_candidateSet.size() < ef)
                        {
                            pre_candidateSet.emplace(dist, itn_centID);
                        }
                        else
                        {
                            assert(pre_candidateSet.size() == ef);
                            if (pre_candidateSet.top().first < dist)
                            {
                                pre_candidateSet.pop();
                                pre_candidateSet.emplace(dist, itn_centID);
                            }
                        }
                    }
                    if ((qvec_visited_out_neighbor_l[itn_centID]) &&
                        (!qvec_returned_point_l[itn_centID]))
                    {
                        assert(qvec_visited_point_l[itn_centID]);
                        float dist = qvec_score_l[itn_centID];
                        topResults.Update(dist, itn_centID);
                    }
                }
            }

            assert(qvec_visited_point_l[ep] && qvec_visited_out_neighbor_l[ep]);

            while (!pre_candidateSet.empty())
            {
                assert(!topResults.InHeap(pre_candidateSet.top().second));
                candidateSet.push(pre_candidateSet.top());
                pre_candidateSet.pop();
            }

            dist_t lowerBound = -std::numeric_limits<float>::max();

            //            uint32_t max_candidate_size = 0;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                //                max_candidate_size = std::max(max_candidate_size, (uint32_t) candidateSet.size());

                if (topResults.Size() == ef && curr_el_pair.first < lowerBound)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;
                assert(qvec_visited_point_l[curNodeNum] == true);
                qvec_visited_out_neighbor_l[curNodeNum] = true;
                int* data = (int*)(data_level0_memory_ + curNodeNum * size_data_per_element_ + offsetLevel0_);
                int size = *data;
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);

                for (int j = 1; j <= size; j++)
                {
                    int tnum = *(data + j);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
                    if (!(qvec_visited_point_l[tnum]))
                    {
                        char* currObj1 = (getDataByInternalId(tnum));

                        dist_t dist = fstdistfunc_(datapoint, currObj1, dist_func_param_);
                        dist_calc++;
                        qvec_score_l[tnum] = dist;
                        qvec_visited_point_l[tnum] = true;

                        if (topResults.Size() < ef || topResults.Top().score_ < dist)
                        {
                            assert(topResults.Size() < ef || dist >= lowerBound);
                            candidateSet.emplace(dist, tnum);
                            _mm_prefetch(data_level0_memory_ + candidateSet.top().second * size_data_per_element_ +
                                         offsetLevel0_, ///////////
                                         _MM_HINT_T0); ////////////////////////

                            assert(!topResults.InHeap(tnum));
                            topResults.Update(dist, tnum);

                            if (topResults.Size() == ef)
                            {
                                lowerBound = topResults.Top().score_;
                            }
                        }
                    }
                }
            }

            //            if (qvecID == 0) {
            //                printf("non-first time, qvecID %d, max_candidate_size %d\n", qvecID, max_candidate_size);
            //            }

#ifndef NDEBUG
            std::set<uint32_t> s;
            for (uint32_t candID = 0; candID < topResults.Size(); candID++)
            {
                assert(s.find(topResults.Data()[candID].itemID_) == s.end());
                s.insert(topResults.Data()[candID].itemID_);
            }
            if (!(s.size() == topResults.Size()))
            {
                printf("s.size() %ld is not equal to topResults.Size() %d\n", s.size(), topResults.Size());
            }
            assert(s.size() == topResults.Size());
#endif
        }

        void searchBaseLayerFirstTime(tableint ep, void* datapoint, size_t ef, const uint32_t qvecID,
                                      float* qvec_score_l,
                                      char* qvec_visited_point_l,
                                      char* qvec_visited_out_neighbor_l)
        {
            //            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareMinHeap> topResults;
            SearchResultMinHeap& topResults = qvec_heap_l_[qvecID];
            std::priority_queue<std::pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareMaxHeap>
                candidateSet;

            dist_t dist = fstdistfunc_(datapoint, getDataByInternalId(ep), dist_func_param_);
            dist_calc++;
            qvec_score_l[ep] = dist;
            assert((qvec_visited_point_l[ep] == false) && (qvec_visited_out_neighbor_l[ep] == false));
            qvec_visited_point_l[ep] = true;

            assert(!topResults.InHeap(ep));
            topResults.Update(dist, ep);
            candidateSet.emplace(dist, ep);
            dist_t lowerBound = -std::numeric_limits<float>::max();

            //            uint32_t max_candidate_size = 0;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                //                max_candidate_size = std::max(max_candidate_size, (uint32_t) candidateSet.size());

                if (topResults.Size() == ef && curr_el_pair.first < lowerBound)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;
                assert(qvec_visited_point_l[curNodeNum] == true);
                qvec_visited_out_neighbor_l[curNodeNum] = true;
                int* data = (int*)(data_level0_memory_ + curNodeNum * size_data_per_element_ + offsetLevel0_);
                int size = *data;
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);

                for (int j = 1; j <= size; j++)
                {
                    int tnum = *(data + j);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
                    if (!(qvec_visited_point_l[tnum]))
                    {
                        char* currObj1 = (getDataByInternalId(tnum));

                        assert(!qvec_visited_point_l[tnum]);
                        dist = fstdistfunc_(datapoint, currObj1, dist_func_param_);
                        dist_calc++;
                        qvec_score_l[tnum] = dist;
                        qvec_visited_point_l[tnum] = true;

                        if (topResults.Size() < ef || topResults.Top().score_ < dist)
                        {
                            assert(topResults.Size() < ef || dist >= lowerBound);
                            candidateSet.emplace(dist, tnum);
                            _mm_prefetch(data_level0_memory_ + candidateSet.top().second * size_data_per_element_ +
                                         offsetLevel0_, ///////////
                                         _MM_HINT_T0); ////////////////////////

                            topResults.Update(dist, tnum);

                            if (topResults.Size() == ef)
                            {
                                lowerBound = topResults.Top().score_;
                            }
                        }
                    }
                }
            }

            //            if (qvecID == 0) {
            //                printf("first time search, qvecID %d, max_candidate_size %d\n", qvecID, max_candidate_size);
            //            }
        }

        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareMinHeap>&
            topResults,
            const int NN)
        {
            if (topResults.size() < NN)
            {
                return;
            }
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareMaxHeap>
                resultSet;
            vector<std::pair<dist_t, tableint>> returnlist;
            while (topResults.size() > 0)
            {
                resultSet.emplace(topResults.top().first, topResults.top().second);
                topResults.pop();
            }

            while (resultSet.size())
            {
                if (returnlist.size() >= NN)
                    break;
                std::pair<dist_t, tableint> curen = resultSet.top();
                dist_t dist_to_query = curen.first;
                resultSet.pop();
                bool good = true;
                for (std::pair<dist_t, tableint> curen2 : returnlist)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(curen2.second), getDataByInternalId(curen.second),
                                     dist_func_param_);;
                    if (curdist > dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    returnlist.push_back(curen);
                }
            }

            for (std::pair<dist_t, tableint> curen2 : returnlist)
            {
                topResults.emplace(curen2.first, curen2.second);
            }
        }

        linklistsizeint* get_linklist0(tableint cur_c)
        {
            return (linklistsizeint*)(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint* get_linklist(tableint cur_c, int level)
        {
            return (linklistsizeint*)(linkLists_[cur_c] + (level - 1) * size_links_per_element_);
        };

        void mutuallyConnectNewElement(void* datapoint, tableint cur_c,
                                       std::priority_queue<
                                           std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                           CompareMinHeap> topResults,
                                       int level)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(topResults, M_);
            while (topResults.size() > M_)
            {
                throw exception();
                topResults.pop();
            }
            vector<tableint> rez;
            rez.reserve(M_);
            while (topResults.size() > 0)
            {
                rez.push_back(topResults.top().second);
                topResults.pop();
            }
            {
                linklistsizeint* ll_cur;
                if (level == 0)
                    ll_cur = (linklistsizeint*)(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_);
                else
                    ll_cur = (linklistsizeint*)(linkLists_[cur_c] + (level - 1) * size_links_per_element_);
                if (*ll_cur)
                {
                    cout << *ll_cur << "\n";
                    cout << elementLevels[cur_c] << "\n";
                    cout << level << "\n";
                    throw runtime_error("Should be blank");
                }
                *ll_cur = rez.size();
                tableint* data = (tableint*)(ll_cur + 1);


                for (int idx = 0; idx < rez.size(); idx++)
                {
                    if (data[idx])
                        throw runtime_error("Should be blank");
                    if (level > elementLevels[rez[idx]])
                        throw runtime_error("Bad level");

                    data[idx] = rez[idx];
                }
            }
            for (int idx = 0; idx < rez.size(); idx++)
            {
                unique_lock<mutex> lock(ll_locks[rez[idx]]);

                if (rez[idx] == cur_c)
                    throw runtime_error("Connection to the same element");
                linklistsizeint* ll_other;
                if (level == 0)
                    ll_other = (linklistsizeint*)(data_level0_memory_ + rez[idx] * size_data_per_element_ +
                        offsetLevel0_);
                else
                    ll_other = (linklistsizeint*)(linkLists_[rez[idx]] + (level - 1) * size_links_per_element_);
                if (level > elementLevels[rez[idx]])
                    throw runtime_error("Bad level");
                int sz_link_list_other = *ll_other;

                if (sz_link_list_other > Mcurmax || sz_link_list_other < 0)
                    throw runtime_error("Bad sz_link_list_other");

                if (sz_link_list_other < Mcurmax)
                {
                    tableint* data = (tableint*)(ll_other + 1);
                    data[sz_link_list_other] = cur_c;
                    *ll_other = sz_link_list_other + 1;
                }
                else
                {
                    // finding the "weakest" element to replace it with the new one
                    tableint* data = (tableint*)(ll_other + 1);
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(rez[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                        CompareMinHeap> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (int j = 0; j < sz_link_list_other; j++)
                    {
                        candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]),
                                                        dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0)
                    {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    *ll_other = indx;
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        mutex global;

        void setEf(size_t ef)
        {
            ef_l_.assign(query_n_vec_, ef);
        }

        void addPoint(void* datapoint, labeltype label, int level = -1)
        {
            tableint cur_c = 0;
            {
                unique_lock<mutex> lock(cur_element_count_guard_);
                if (cur_element_count >= maxelements_)
                {
                    cout << "The number of elements exceeds the specified limit\n";
                    throw runtime_error("The number of elements exceeds the specified limit");
                };
                cur_c = cur_element_count;
                cur_element_count++;
            }
            unique_lock<mutex> lock_el(ll_locks[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;
            elementLevels[cur_c] = curlevel;


            unique_lock<mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), datapoint, data_size_);


            if (curlevel)
            {
                linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel);
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel);
            }
            if (currObj != -1)
            {
                if (curlevel < maxlevelcopy)
                {
                    dist_t curdist = fstdistfunc_(datapoint, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            int* data;
                            unique_lock<mutex> lock(ll_locks[currObj]);
                            data = (int*)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
                            int size = *data;
                            tableint* datal = (tableint*)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > maxelements_)
                                    throw runtime_error("cand error");
                                dist_t d = fstdistfunc_(datapoint, getDataByInternalId(cand), dist_func_param_);
                                if (d > curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                for (int level = min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0)
                        throw runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                        CompareMinHeap> topResults = searchBaseLayerBuildIndex(
                        currObj, datapoint,
                        level);
                    mutuallyConnectNewElement(datapoint, cur_c, topResults, level);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node = 0;
                maxlevel_ = curlevel;
            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node = cur_c;
                maxlevel_ = curlevel;
            }
        };

        void searchKnn(void* qvec, const uint32_t qvecID,
                       float* qvec_score_l,
                       char* qvec_visited_point_l,
                       char* qvec_visited_out_neighbor_l)
        {
            tableint currObj = enterpoint_node;

            dist_t curdist = fstdistfunc_(qvec, getDataByInternalId(enterpoint_node), dist_func_param_);
            dist_calc++;

            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    int* data;
                    data = (int*)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
                    int size = *data;
                    tableint* datal = (tableint*)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > maxelements_)
                            throw runtime_error("cand error");

                        dist_t d = fstdistfunc_(qvec, getDataByInternalId(cand), dist_func_param_);
                        dist_calc++;

                        if (d > curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            base_layer_ep_l_[qvecID] = currObj;

            searchBaseLayerFirstTime(
                currObj, qvec, ef_l_[qvecID], qvecID,
                qvec_score_l,
                qvec_visited_point_l,
                qvec_visited_out_neighbor_l);
        };

        void saveIndex(const std::string& filename) {
            cout << "Saving index to " << filename.c_str() << "\n";
            std::ofstream output(filename, std::ios::binary);
            streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, size_links_per_element_);

            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, maxM_);
            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, efConstruction_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, revSize_);
            writeBinaryPOD(output, maxlevel_);

            writeBinaryPOD(output, enterpoint_node);
            writeBinaryPOD(output, dist_calc);
            writeBinaryPOD(output, size_links_level0_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, offsetLevel0_);

            output.write(data_level0_memory_, maxelements_ * size_data_per_element_);

            for (size_t i = 0; i < maxelements_; i++) {
                unsigned int linkListSize = elementLevels[i] > 0 ? size_links_per_element_ * elementLevels[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void LoadIndex(const string &location, SpaceInterface<dist_t> *s) {
            cout << "Loading index from " << location << "\n";
            std::ifstream input(location, std::ios::binary);
            streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, cur_element_count);
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, size_links_per_element_);

            readBinaryPOD(input, M_);
            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, efConstruction_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, revSize_);
            readBinaryPOD(input, maxlevel_);

            readBinaryPOD(input, enterpoint_node);
            readBinaryPOD(input, dist_calc);
            readBinaryPOD(input, size_links_level0_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, offsetLevel0_);

            data_size_ = s->get_data_size();
            label_offset_ = size_links_level0_ + data_size_;
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            data_level0_memory_ = (char *) malloc(maxelements_ * size_data_per_element_);
            input.read(data_level0_memory_, maxelements_ * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            visitedlistpool = new VisitedListPool(1, maxelements_);


            linkLists_ = (char **) malloc(sizeof(void *) * maxelements_);
            elementLevels = vector<int>(maxelements_);
            revSize_ = 1.0 / mult_;
            for (size_t i = 0; i < maxelements_; i++) {
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    elementLevels[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    elementLevels[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    input.read(linkLists_[i], linkListSize);
                }
            }

            input.close();
        }

    };
}
