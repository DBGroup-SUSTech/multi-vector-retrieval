//
// Created by username2 on 2023/2/16.
//

#ifndef VECTORSETSEARCH_DISTANCE_HPP
#define VECTORSETSEARCH_DISTANCE_HPP

// #include <immintrin.h>

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#include <queue>

#include "include/alg/MatrixMulBLAS.hpp"


namespace VectorSetSearch
{
    class BatchMaxSim
    {
    public:
        std::unique_ptr<float[]> score_l_; // query_n_vec * n_item * max_item_n_vec
        std::unique_ptr<float[]> max_score_l_; // query_n_vec * n_item

        uint32_t query_n_vec_;
        uint32_t n_item_, max_item_n_vec_;

        BatchMaxSim() = default;

        BatchMaxSim(const uint32_t& query_n_vec, const uint32_t& n_item,
                    const uint32_t& max_item_n_vec)
        {
            this->query_n_vec_ = query_n_vec;
            this->n_item_ = n_item;
            this->max_item_n_vec_ = max_item_n_vec;

            score_l_ = std::make_unique<float[]>(n_item_ * query_n_vec_ * max_item_n_vec_);
            max_score_l_ = std::make_unique<float[]>(n_item_ * query_n_vec_);
        }

        void compute(const float* query,
                     const float* item_l, const uint32_t* item_n_vec_l, const uint32_t* item_n_vec_accu_l,
                     const uint32_t vec_dim, const uint32_t n_filter_item,
                     std::pair<float, uint32_t>* max_sim_score_l)
        {
            assert(n_filter_item <= n_item_);
            const uint32_t n_vec = item_n_vec_l[n_filter_item - 1] + item_n_vec_accu_l[n_filter_item - 1];
            // first compute the whole matrix
            MatrixMultiply(query, item_l, query_n_vec_,
                           n_vec, vec_dim, score_l_.get());

            // then perform the max operator
            for (uint32_t qID = 0; qID < query_n_vec_; qID++)
            {
                const float* query_score_l = score_l_.get() + qID * n_vec;
                float* query_max_score_l = max_score_l_.get() + qID * n_item_;

                for (uint32_t itemID = 0; itemID < n_item_; itemID++)
                {
                    const uint32_t item_n_vec = item_n_vec_l[itemID];
                    const uint32_t item_offset = item_n_vec_accu_l[itemID];
                    const float* query_item_score_l = query_score_l + item_offset;
                    auto max_ele = std::max_element(query_item_score_l, query_item_score_l + item_n_vec);
                    const float max_score = *max_ele;
                    query_max_score_l[itemID] = max_score;
                }
            }

            // then perform the sum operator
            for (uint32_t itemID = 0; itemID < n_item_; itemID++)
            {
                max_sim_score_l[itemID].first = 0.0f;
            }
            for (uint32_t qID = 0; qID < query_n_vec_; qID++)
            {
                const float* query_max_score_l = max_score_l_.get() + qID * n_item_;
                for (uint32_t itemID = 0; itemID < n_item_; itemID++)
                {
                    max_sim_score_l[itemID].first += query_max_score_l[itemID];
                }
            }
        }


        void compute(const float* query, const uint32_t query_n_vec_actual,
                     const float* item_l, const uint32_t* item_n_vec_l, const uint32_t* item_n_vec_accu_l,
                     const uint32_t vec_dim, const uint32_t n_filter_item,
                     std::pair<float, uint32_t>* max_sim_score_l)
        {
            assert(n_filter_item <= n_item_);
            const uint32_t n_vec = item_n_vec_l[n_filter_item - 1] + item_n_vec_accu_l[n_filter_item - 1];
            // first compute the whole matrix
            MatrixMultiply(query, item_l, query_n_vec_actual,
                           n_vec, vec_dim, score_l_.get());

            // then perform the max operator
            for (uint32_t qID = 0; qID < query_n_vec_actual; qID++)
            {
                const float* query_score_l = score_l_.get() + qID * n_vec;
                float* query_max_score_l = max_score_l_.get() + qID * n_item_;

                for (uint32_t itemID = 0; itemID < n_item_; itemID++)
                {
                    const uint32_t item_n_vec = item_n_vec_l[itemID];
                    const uint32_t item_offset = item_n_vec_accu_l[itemID];
                    const float* query_item_score_l = query_score_l + item_offset;
                    auto max_ele = std::max_element(query_item_score_l, query_item_score_l + item_n_vec);
                    const float max_score = *max_ele;
                    query_max_score_l[itemID] = max_score;
                }
            }

            // then perform the sum operator
            for (uint32_t itemID = 0; itemID < n_item_; itemID++)
            {
                max_sim_score_l[itemID].first = 0.0f;
            }
            for (uint32_t qID = 0; qID < query_n_vec_actual; qID++)
            {
                const float* query_max_score_l = max_score_l_.get() + qID * n_item_;
                for (uint32_t itemID = 0; itemID < n_item_; itemID++)
                {
                    max_sim_score_l[itemID].first += query_max_score_l[itemID];
                }
            }
        }
    };

    class BatchMaxSimNeighborhoodFetch
    {
    public:
        std::unique_ptr<float[]> score_l_; // query_n_vec * max_item_n_vec
        std::vector<float> max_score_l_; // query_n_vec, store at most query_n_vec value
        std::unique_ptr<float[]> query_refine_l_; // query_n_vec * vec_dim, store at most query_n_vec

        uint32_t query_n_vec_, n_item_;
        uint32_t max_item_n_vec_, vec_dim_;

        BatchMaxSimNeighborhoodFetch() = default;

        BatchMaxSimNeighborhoodFetch(const uint32_t query_n_vec, const uint32_t n_item,
                                     const uint32_t max_item_n_vec, const uint32_t vec_dim)
        {
            this->query_n_vec_ = query_n_vec;
            this->n_item_ = n_item;
            this->max_item_n_vec_ = max_item_n_vec;
            this->vec_dim_ = vec_dim;

            score_l_ = std::make_unique<float[]>(query_n_vec_ * max_item_n_vec_);
            max_score_l_ = std::vector<float>(query_n_vec_);
            query_refine_l_ = std::make_unique<float[]>(query_n_vec_ * vec_dim_);
        }

        void compute(const float* query,
                     const float* item_l, const uint32_t* item_n_vec_l, const uint32_t* item_n_vec_accu_l,
                     const char* refine_qvec_l,
                     const uint32_t n_filter_item,
                     std::pair<float, uint32_t>* item_score_l)
        {
            for (uint32_t candID = 0; candID < n_filter_item; candID++)
            {
                uint32_t n_refine_qvec = 0;
                const uint32_t itemID = item_score_l[candID].second;
                for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
                {
                    if (refine_qvec_l[qvecID * n_item_ + itemID])
                    {
                        std::memcpy(query_refine_l_.get() + n_refine_qvec * vec_dim_,
                                    query + qvecID * vec_dim_, sizeof(float) * vec_dim_);
                        n_refine_qvec++;
                    }
                }
                const uint32_t item_n_vec = item_n_vec_l[candID];
                const uint32_t item_offset = item_n_vec_accu_l[candID];
                const float* item = item_l + item_offset * vec_dim_;
                MatrixMultiply(query_refine_l_.get(), item, n_refine_qvec, item_n_vec,
                               vec_dim_, score_l_.get());

                float item_score = 0.0f;
                for (uint32_t qvecID = 0; qvecID < n_refine_qvec; qvecID++)
                {
                    const float* query_item_score_l = score_l_.get() + qvecID * item_n_vec;
                    auto max_ele = std::max_element(query_item_score_l, query_item_score_l + item_n_vec);
                    const float max_score = *max_ele;
                    item_score += max_score;
                }

                item_score_l[candID].first = item_score;
            }
        }
    };

    float vectorSetDistance(const float* query_ptr, const uint32_t& query_n_vecs,
                            const float* item_ptr, const uint32_t& item_n_vecs,
                            const uint32_t& vec_dim)
    {
        float total_score = 0.0f;
        for (uint32_t queryID = 0; queryID < query_n_vecs; queryID++)
        {
            const float* query_vecs = query_ptr + queryID * vec_dim;
            float max_score = -std::numeric_limits<float>::max();
            for (uint32_t itemID = 0; itemID < item_n_vecs; itemID++)
            {
                const float* item_vecs = item_ptr + itemID * vec_dim;
                const float ip = std::inner_product(query_vecs, query_vecs + vec_dim, item_vecs, 0.0f);
                max_score = max_score > ip ? max_score : ip;
            }
            total_score += max_score;
        }
        return total_score;
    }

    float euclideanDistance(const float* query_vecs, const float* item_vecs, const uint32_t& vec_dim)
    {
        float total_score = 0;
        for (uint32_t dim = 0; dim < vec_dim; dim++)
        {
            const float diff = query_vecs[dim] - item_vecs[dim];
            total_score += diff * diff;
        }
        total_score = sqrt(total_score);
        return total_score;
    }

    float hausdorffDistance(const float* item_vecs, const uint32_t& item_n_vecs,
                            const float* represent_vecs, const uint32_t& represent_n_vecs,
                            const uint32_t& vec_dim)
    {
        float hau_dist = -std::numeric_limits<float>::max();
        for (uint32_t itemID = 0; itemID < item_n_vecs; itemID++)
        {
            const float* item_vec = item_vecs + itemID * vec_dim;
            float min_score = std::numeric_limits<float>::max();
            for (uint32_t repreID = 0; repreID < represent_n_vecs; repreID++)
            {
                const float* repre_vec = represent_vecs + repreID * vec_dim;
                const float euc_dist = euclideanDistance(item_vec, repre_vec, vec_dim);
                min_score = std::min(min_score, euc_dist);
            }
            hau_dist = hau_dist > min_score ? hau_dist : min_score;
        }
        return hau_dist;
    }


    template <class T>
    void removeDuplicates(std::vector<T>& v)
    {
        std::sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
    }
}
#endif //VECTORSETSEARCH_DISTANCE_HPP
