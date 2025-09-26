//
// Created by bianzheng on 2024/8/14.
//

#ifndef VECTORSETSEARCH_TRANSFORMIP_HPP
#define VECTORSETSEARCH_TRANSFORMIP_HPP

#include "include/struct/TypeDef.hpp"
#include "include/alg/MatrixMulBLAS.hpp"

#include "include/util/util.hpp"

namespace VectorSetSearch
{
    pyarray_float
    AssignClusterVector(const pyarray_uint8& vec_cluster_bit_l_py, const pyarray_float& item_vecs_l_chunk_py,
                        const pyarray_uint32& item_n_vec_l_chunk_py,
                        const uint32_t batch_n_vec, const uint32_t batch_n_item,
                        const uint32_t r_reps,
                        const uint32_t k_sim, const uint32_t vec_dim)
    {
        // the input is partition bit of each vector, output is the code of each vector
        // partition_bit_l_py: batch_n_vec * k_sim, item_vecs_l_chunk_py: batch_n_vec * vec_dim
        // itemlen_l_chunk_py: batch_n_item

        const uint8_t* vec_cluster_bit_l = vec_cluster_bit_l_py.data();
        assert(vec_cluster_bit_l_py.ndim() == 3);
        assert(vec_cluster_bit_l_py.shape(0) == batch_n_vec);
        assert(vec_cluster_bit_l_py.shape(1) == r_reps);
        assert(vec_cluster_bit_l_py.shape(2) == k_sim);

        const float* item_vecs_l_chunk = item_vecs_l_chunk_py.data();
        assert(item_vecs_l_chunk_py.ndim() == 2);
        assert(item_vecs_l_chunk_py.shape(0) == batch_n_vec);
        assert(item_vecs_l_chunk_py.shape(1) == vec_dim);

        const uint32_t* item_n_vec_l = item_n_vec_l_chunk_py.data();
        assert(item_n_vec_l_chunk_py.ndim() == 1);
        assert(item_n_vec_l_chunk_py.shape(0) == batch_n_item);

        std::vector<uint32_t> item_n_vec_accu_l(batch_n_item);
        item_n_vec_accu_l[0] = 0;
        for (uint32_t itemID = 1; itemID < batch_n_item; itemID++)
        {
            item_n_vec_accu_l[itemID] = item_n_vec_accu_l[itemID - 1] + item_n_vec_l[itemID - 1];
        }

        const uint32_t n_cluster = 1 << k_sim;
        float* cluster_vec_l = new float[(size_t)batch_n_item * r_reps * n_cluster * vec_dim];

#pragma omp parallel for default(none) shared(batch_n_item, r_reps, item_n_vec_accu_l, item_n_vec_l, k_sim, vec_cluster_bit_l, item_vecs_l_chunk, vec_dim, n_cluster, cluster_vec_l)
        for (uint32_t itemID = 0; itemID < batch_n_item; itemID++)
        {
            // compute the cluster ID of each vector
            const uint32_t start_vecID = item_n_vec_accu_l[itemID];
            const uint32_t item_n_vec = item_n_vec_l[itemID];

            const uint8_t* item_cluster_bit = vec_cluster_bit_l + start_vecID * r_reps * k_sim;
            const float* raw_vec_l = item_vecs_l_chunk + start_vecID * vec_dim;
            for (uint32_t repID = 0; repID < r_reps; repID++)
            {
                std::vector<uint32_t> assign_cluster_number(n_cluster, 0);
                std::vector<float> cluster_vec(n_cluster * vec_dim, 0.0f);
                for (uint32_t item_vecID = 0; item_vecID < item_n_vec; item_vecID++)
                {
                    uint32_t clusterID = 0;
                    for (uint32_t ksimID = 0; ksimID < k_sim; ksimID++)
                    {
                        assert(item_cluster_bit[item_vecID * r_reps * k_sim + repID * k_sim + ksimID] == 0 ||
                            item_cluster_bit[item_vecID * r_reps * k_sim + repID * k_sim + ksimID] == 1);
                        const uint32_t bit_val =
                            item_cluster_bit[item_vecID * r_reps * k_sim + repID * k_sim + ksimID]
                                ? 1
                                : 0;
                        clusterID += (bit_val << ksimID);
                    }

                    for (uint32_t dimID = 0; dimID < vec_dim; dimID++)
                    {
                        cluster_vec[clusterID * vec_dim + dimID] += raw_vec_l[item_vecID * vec_dim + dimID];
                    }
                    assign_cluster_number[clusterID]++;
                }

                // copy the item vector to its corresponding cluster
                for (uint32_t clusterID = 0; clusterID < n_cluster; clusterID++)
                {
                    if (assign_cluster_number[clusterID] == 0)
                    {
                        continue;
                    }
                    const uint32_t n_vec_cluster = assign_cluster_number[clusterID];
                    for (uint32_t dimID = 0; dimID < vec_dim; dimID++)
                    {
                        cluster_vec[clusterID * vec_dim + dimID] /= (float) n_vec_cluster;
                    }
                }


                for (uint32_t clusterID = 0; clusterID < n_cluster; clusterID++)
                {
                    // check who is not assigned
                    if (assign_cluster_number[clusterID] == 0)
                    {
                        // for each not assigned cluster, assign and copy the vector with the minimum l1 distance
                        uint32_t min_l1_distance = std::numeric_limits<uint32_t>::max();
                        uint32_t min_dist_clusterID = n_cluster + 100;
                        for (uint32_t iter_clsID = 0; iter_clsID < n_cluster; iter_clsID++)
                        {
                            if (assign_cluster_number[iter_clsID] > 0)
                            {
                                uint32_t l1_distance = std::popcount(iter_clsID ^ clusterID);
                                if (l1_distance < min_l1_distance)
                                {
                                    min_l1_distance = l1_distance;
                                    min_dist_clusterID = iter_clsID;
                                }
                            }
                        }
                        assert(min_l1_distance != std::numeric_limits<uint32_t>::max());
                        assert(min_dist_clusterID != n_cluster + 100);
                        assert(assign_cluster_number[min_dist_clusterID] > 0);

                        std::memcpy(cluster_vec.data() + clusterID * vec_dim,
                                    cluster_vec.data() + min_dist_clusterID * vec_dim,
                                    sizeof(float) * vec_dim);
                        // assign_cluster_number[clusterID] = assign_cluster_number[min_dist_clusterID];
                    }
                }

#ifndef NDEBUG
                // for (uint32_t clusterID = 0; clusterID < n_cluster; clusterID++)
                // {
                //     assert(assign_cluster_number[clusterID] > 0);
                // }
#endif

                std::memcpy(
                    cluster_vec_l + (size_t)itemID * r_reps * n_cluster * vec_dim + repID * n_cluster * vec_dim,
                    cluster_vec.data(),
                    sizeof(float) * n_cluster * vec_dim);
            }
        }

        py::capsule handle_cluster_vec_l_ptr(cluster_vec_l, Method::PtrDelete<float>);

        return py::array_t<float>(
            {batch_n_item, r_reps, n_cluster, vec_dim},
            {
                r_reps * n_cluster * vec_dim * sizeof(float), n_cluster * vec_dim * sizeof(float),
                vec_dim * sizeof(float), sizeof(float)
            },
            cluster_vec_l, handle_cluster_vec_l_ptr
        );
    }


    class TransformQuery
    {
        uint32_t _k_sim, _n_cluster, _d_proj, _r_reps;
        uint32_t _vec_dim;
        const float* _partition_vec_l; // r_reps * k_sim * vec_dim
        const float* _random_matrix_l; // r_reps * d_proj * vec_dim

        uint32_t _query_n_vec;

        std::vector<float> _partition_code_l; // query_n_vec * _r_reps * _k_sim
        std::vector<uint32_t> _query_code_l; // query_n_vec * _r_reps

        // transformation used cache
        std::vector<float> _query_cluster_vec_l; // r_reps * n_cluster * vec_dim
        std::vector<float> _query_ip_vector_l; // r_reps * n_cluster * d_proj

    public:
        inline TransformQuery() = default;

        inline TransformQuery(const float* partition_vec_l, const float* random_matrix_l,
                              const uint32_t k_sim, const uint32_t n_cluster, const uint32_t d_proj,
                              const uint32_t r_reps,
                              const uint32_t vec_dim)
        {
            this->_k_sim = k_sim;
            this->_n_cluster = n_cluster;
            this->_d_proj = d_proj;
            this->_r_reps = r_reps;
            this->_vec_dim = vec_dim;

            this->_partition_vec_l = partition_vec_l;
            this->_random_matrix_l = random_matrix_l;

            _query_cluster_vec_l.resize(_r_reps * _n_cluster * _vec_dim);
            _query_ip_vector_l.resize(_r_reps * _n_cluster * _d_proj);
        }

        void initQueryInfo(const uint32_t query_n_vec)
        {
            this->_query_n_vec = query_n_vec;

            _partition_code_l.resize(_query_n_vec * _r_reps * _k_sim);
            _query_code_l.resize(_query_n_vec * _r_reps);
        }

        float* transformQuery2IP(const float* query)
        {
            _query_cluster_vec_l.assign(_r_reps * _n_cluster * _vec_dim, 0.0f);

            // (query_n_vec, vec_dim) * (r_reps * k_sim, vec_dim) -> (query_n_vec, r_reps * k_sim)
            // compute the cluster code of each query vector
            MatrixMultiply(query, _partition_vec_l,
                           _query_n_vec, _r_reps * _k_sim, _vec_dim, _partition_code_l.data());

            for (uint32_t qvecID = 0; qvecID < _query_n_vec; qvecID++)
            {
                for (uint32_t repID = 0; repID < _r_reps; repID++)
                {
                    const float* partition_value =
                        _partition_code_l.data() + qvecID * _r_reps * _k_sim + repID * _k_sim;
                    uint32_t query_code = 0;
                    for (uint32_t simID = 0; simID < _k_sim; simID++)
                    {
                        const uint32_t code = partition_value[simID] > 0 ? 1 : 0;
                        query_code += (code << simID);
                    }

                    assert(query_code < _n_cluster);
                    _query_code_l[qvecID * _r_reps + repID] = query_code;
                }
            }

            // compute the assign of the cluster vector
            for (uint32_t qvecID = 0; qvecID < _query_n_vec; qvecID++)
            {
                for (uint32_t repID = 0; repID < _r_reps; repID++)
                {
                    const uint32_t clusterID = _query_code_l[qvecID * _r_reps + repID];
                    const float* query_vec = query + qvecID * _vec_dim;
                    for (uint32_t dim = 0; dim < _vec_dim; dim++)
                    {
                        _query_cluster_vec_l[repID * _n_cluster * _vec_dim +
                            clusterID * _vec_dim + dim] += query_vec[dim];
                    }
                }
            }

            // (r_reps, n_cluster, vec_dim) * (r_reps, d_proj, vec_dim) -> (r_reps, n_cluster, d_proj)
            // compute the projection of the vector
            for (uint32_t repID = 0; repID < _r_reps; repID++)
            {
                MatrixMultiply(_query_cluster_vec_l.data() + repID * _n_cluster * _vec_dim,
                               _random_matrix_l + repID * _d_proj * _vec_dim,
                               _n_cluster, _d_proj, _vec_dim, _query_ip_vector_l.data() + repID * _n_cluster * _d_proj);
            }
            //            MatrixMultiply(_query_cluster_vec_l.data(), _random_matrix_l,
            //                           _n_cluster, _r_reps * _d_proj, _vec_dim, _query_ip_vector_l.data());

            return _query_ip_vector_l.data();
        }
    };
}
#endif //VECTORSETSEARCH_TRANSFORMIP_HPP
