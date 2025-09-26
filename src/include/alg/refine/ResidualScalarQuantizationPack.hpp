//
// Created by Administrator on 2025/6/10.
//

#ifndef RESIDUALSCALARQUANTIZATIONPACK_HPP
#define RESIDUALSCALARQUANTIZATIONPACK_HPP

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>

#include "include/struct/TypeDef.hpp"

namespace VectorSetSearch
{
    py::tuple ComputeQuantizedScalarPack(const pyarray_float& item_vec_l_py, const pyarray_float& centroid_l_py,
                                         const pyarray_uint32& code_l_py, const uint32_t& n_bit)
    {
        const size_t n_vec = item_vec_l_py.shape(0);
        const uint32_t vec_dim = item_vec_l_py.shape(1);

        assert(vec_dim == centroid_l_py.shape(1));
        assert(code_l_py.ndim() == 1 && code_l_py.shape(0) == n_vec);

        const float* item_vec_l = item_vec_l_py.data();
        const float* centroid_l = centroid_l_py.data();
        const uint32_t* code_l = code_l_py.data();

        std::vector<float> residual_error_l((size_t)n_vec * vec_dim);
#pragma omp parallel for default(none) shared(n_vec, code_l, item_vec_l, vec_dim, centroid_l, residual_error_l, centroid_l_py)
        for (size_t vecID = 0; vecID < n_vec; vecID++)
        {
            const uint32_t code = code_l[vecID];
            assert(code < centroid_l_py.shape(0));
            const float* vec = item_vec_l + (size_t)vecID * vec_dim;

            std::vector<float> centroid(centroid_l + code * vec_dim, centroid_l + (code + 1) * vec_dim);

            float* vec_error_l = residual_error_l.data() + (size_t)vecID * vec_dim;
            for (uint32_t dim = 0; dim < vec_dim; dim++)
            {
                vec_error_l[dim] = vec[dim] - centroid[dim];
            }
        }

        const uint32_t n_quantile = 1 << n_bit;
        std::vector<double> quantile_l(n_quantile);
        for (uint32_t quanID = 0; quanID < n_quantile; quanID++)
        {
            const double quantile = 1.0 * quanID / n_quantile;
            quantile_l[quanID] = quantile;
        }

        const uint32_t n_cutoff = n_quantile - 1;
        std::vector<float> cutoff_l(n_cutoff);
        for (uint32_t cutID = 0; cutID < n_cutoff; cutID++)
        {
            const double quantile = quantile_l[cutID + 1];
            assert(0 < quantile && quantile < 1.0);
            const size_t quantile_idx = std::floor(quantile * (double)residual_error_l.size());
            std::nth_element(residual_error_l.begin(), residual_error_l.begin() + quantile_idx, residual_error_l.end());

            const float weight = residual_error_l[quantile_idx];
            cutoff_l[cutID] = weight;
        }

        const uint32_t n_weight = n_quantile;
        std::vector<float> weight_l(n_weight);
        for (uint32_t wID = 0; wID < n_weight; wID++)
        {
            const double quantile = quantile_l[wID] + 0.5 / n_quantile;
            assert(0 < quantile && quantile < 1.0);
            const size_t quantile_idx = std::floor(quantile * (double)residual_error_l.size());
            std::nth_element(residual_error_l.begin(), residual_error_l.begin() + quantile_idx, residual_error_l.end());

            const float weight = residual_error_l[quantile_idx];
            weight_l[wID] = weight;
        }

#ifndef NDEBUG
        assert(cutoff_l.size() + 1 == weight_l.size());
        for (uint32_t cutID = 0; cutID < n_cutoff - 1; cutID++)
        {
            assert(cutoff_l[cutID] < cutoff_l[cutID + 1]);
        }
        for (uint32_t wID = 0; wID < n_weight - 1; wID++)
        {
            assert(weight_l[wID] < weight_l[wID + 1]);
        }
        for (uint32_t ID = 0; ID < n_cutoff; ID++)
        {
            assert(weight_l[ID] < cutoff_l[ID]);
        }
        assert(cutoff_l[n_cutoff - 1] < weight_l[n_weight - 1]);
#endif

        return py::make_tuple(cutoff_l, weight_l);
    }

    class CompressResidualCode
    {
    public:
        uint32_t n_centroid_, vec_dim_;
        std::vector<float> centroid_l_; // n_centroid * vec_dim

        uint32_t n_cutoff_;
        std::vector<float> cutoff_l_;

        uint32_t n_weight_;
        std::vector<float> weight_l_;

        uint32_t n_bit_;
        uint32_t n_val_per_byte_; // n_bit_per_byte / n_bit_;
        uint32_t n_uint8_per_vec_; // (vec_dim_ + n_val_per_byte_ - 1) / n_val_per_byte_;

        CompressResidualCode() = default;

        CompressResidualCode(const pyarray_float& centroid_l_py,
                             const pyarray_float& cutoff_l_py,
                             const pyarray_float& weight_l_py,
                             const uint32_t n_bit)
        {
            assert(centroid_l_py.ndim() == 2);
            this->n_centroid_ = centroid_l_py.shape(0);
            this->vec_dim_ = centroid_l_py.shape(1);
            this->centroid_l_.resize(n_centroid_ * vec_dim_);
            std::memcpy(centroid_l_.data(), centroid_l_py.data(), n_centroid_ * vec_dim_ * sizeof(float));

            assert(cutoff_l_py.ndim() == 1);
            this->n_cutoff_ = cutoff_l_py.shape(0);
            this->cutoff_l_.resize(n_cutoff_);
            assert(n_cutoff_ == (1 << n_bit) - 1);
            std::memcpy(cutoff_l_.data(), cutoff_l_py.data(), n_cutoff_ * sizeof(float));

            assert(weight_l_py.ndim() == 1);
            this->n_weight_ = weight_l_py.shape(0);
            this->weight_l_.resize(n_weight_);
            assert(n_weight_ == (1 << n_bit));
            std::memcpy(weight_l_.data(), weight_l_py.data(), n_weight_ * sizeof(float));

            this->n_bit_ = n_bit;
            if ((n_bit_ != 1) && (n_bit_ != 2) && (n_bit_ != 4) && (n_bit_ != 8))
            {
                spdlog::error("n_bit is not in range {1, 2, 4, 8}, program exit");
                exit(-1);
            }
            constexpr uint32_t n_bit_per_byte = 8;
            this->n_val_per_byte_ = n_bit_per_byte / n_bit_;
            assert(n_bit_per_byte % n_bit_ == 0);
            this->n_uint8_per_vec_ = (vec_dim_ + n_val_per_byte_ - 1) / n_val_per_byte_;
        }

        py::tuple compute_residual_code(const pyarray_float& vec_l_py, const pyarray_uint32& code_l_py) const
        {
            const size_t n_vec = vec_l_py.shape(0);
            assert(n_vec == code_l_py.shape(0));

            const float* vec_l = vec_l_py.data();
            const uint32_t* code_l = code_l_py.data();
            spdlog::info("ComputeResidualCode");

            std::vector<uint8_t> residual_code_l((size_t)n_vec * n_uint8_per_vec_);
            // #pragma omp parallel for default(none) shared(n_vec, code_l, vec_l, vec_dim_, centroid_l, residual_code_l, n_weight, weight_l, cutoff_l, n_centroid, n_cutoff, residual_norm_l)
            for (size_t vecID = 0; vecID < n_vec; vecID++)
            {
                const uint32_t code = code_l[vecID];
                assert(code < n_centroid_);
                const float* vec = vec_l + (size_t)vecID * vec_dim_;

                std::vector<float> centroid(centroid_l_.data() + code * vec_dim_,
                                            centroid_l_.data() + (code + 1) * vec_dim_);

                uint8_t* residual_code = residual_code_l.data() + vecID * n_uint8_per_vec_;
                for (uint32_t packID = 0; packID < n_uint8_per_vec_; packID++)
                {
                    const uint32_t start_dim = packID * n_val_per_byte_;
                    std::vector<uint8_t> sq_code_l(n_val_per_byte_);
                    for (uint32_t valID = 0; valID < n_val_per_byte_; valID++)
                    {
                        const uint32_t dim = start_dim + valID;
                        uint8_t sq_code;
                        if (dim < vec_dim_)
                        {
                            const float error = vec[dim] - centroid[dim];
                            const float* ptr = std::upper_bound(cutoff_l_.data(), cutoff_l_.data() + n_cutoff_, error,
                                                                [](const float& ele, const float& error)
                                                                {
                                                                    return ele < error;
                                                                });
                            sq_code = ptr - cutoff_l_.data();
                            assert(sq_code < n_weight_);
                            assert(sq_code <= n_cutoff_);
                            if (!((sq_code == 0 && error <= cutoff_l_[sq_code]) ||
                                (0 < sq_code && sq_code < n_cutoff_ && cutoff_l_[sq_code - 1] <= error && error < cutoff_l_[sq_code]) ||
                                (sq_code == n_cutoff_ && cutoff_l_[sq_code - 1] <= error)))
                            {
                                printf("error %.3f, sq_code %d\n", error, sq_code);
                                printf("weight: ");
                                for (uint32_t i = 0; i < n_cutoff_; i++)
                                {
                                    printf("%.3f ", cutoff_l_[i]);
                                }
                                printf("\n");
                            }
                            assert((sq_code == 0 && error <= cutoff_l_[sq_code]) ||
                                (0 < sq_code && sq_code < n_cutoff_ && cutoff_l_[sq_code - 1] <= error && error < cutoff_l_[sq_code]) ||
                                (sq_code == n_cutoff_ && cutoff_l_[sq_code - 1] <= error));
                            assert(n_weight_ == (1 << n_bit_));
                        }
                        else
                        {
                            sq_code = 0;
                        }
                        sq_code_l[valID] = sq_code;
                    }
                    const uint8_t packed_code = compress(sq_code_l.data());
                    residual_code[packID] = packed_code;
                }
            }

            return py::make_tuple(residual_code_l);
        }


        uint8_t compress(const uint8_t* sq_code_l) const
        {
            uint8_t result = 0;
            const uint8_t mask = (1 << n_bit_) - 1; // n_bit位掩码

            for (int valID = 0; valID < n_val_per_byte_; ++valID)
            {
                result = (result << n_bit_) | (sq_code_l[valID] & mask);
            }
            return result;
        }
    };

    class ResidualCodePack
    {
    public:
        const uint8_t* residual_code_l_;
        const float* weight_l_;
        uint32_t n_weight_;

        const uint32_t* item_n_vec_l_;
        const size_t* item_n_vec_accu_l_;
        uint32_t n_item_;
        uint32_t vec_dim_;

        uint32_t n_bit_, n_val_per_byte_, n_packed_val_per_vec_;

        std::vector<float> decompression_table_; // 256 * n_val_per_byte_
        // store the weight of each decompress code

        const float* centroid_l_;
        const uint32_t* vq_code_l_;

        ResidualCodePack() = default;

        ResidualCodePack(const uint8_t* residual_code_l, const float* weight_l,
                         const uint32_t* item_n_vec_l, const size_t* item_n_vec_accu_l,
                         const float* centroid_l, const uint32_t* vq_code_l,
                         const uint32_t n_weight, const uint32_t n_item, const size_t n_vec,
                         const uint32_t vec_dim, const uint32_t n_bit)
        {
            residual_code_l_ = residual_code_l;
            weight_l_ = weight_l;
            n_weight_ = n_weight;

            item_n_vec_l_ = item_n_vec_l;
            item_n_vec_accu_l_ = item_n_vec_accu_l;
            n_item_ = n_item;
            vec_dim_ = vec_dim;
            n_bit_ = n_bit;

            constexpr uint32_t n_bit_per_byte = 8;
            n_val_per_byte_ = n_bit_per_byte / n_bit_;
            assert(n_bit_per_byte % n_bit_ == 0);
            n_packed_val_per_vec_ = (vec_dim_ + n_val_per_byte_ - 1) / n_val_per_byte_;

            decompression_table_.resize(256 * n_val_per_byte_);
            // generate decompression table
            for (uint32_t code = 0; code < 256; code++)
            {
                std::vector<uint8_t> decompress_sq_code_l = decompressSQPackedCode((int)n_bit_, (uint8_t)code);
                assert(decompress_sq_code_l.size() == n_val_per_byte_);
                float* code_decompress_table = decompression_table_.data() + code * n_val_per_byte_;
                for (uint32_t valID = 0; valID < n_val_per_byte_; valID++)
                {
                    const uint8_t decompressed_sq_code = decompress_sq_code_l[valID];
                    const float weight = weight_l_[decompressed_sq_code];
                    code_decompress_table[valID] = weight;
                }
            }

            // for(uint32_t code=0;code < 256;code++) {
            //     printf("code: ");
            //     for(uint32_t valID=0;valID < n_val_per_byte_;valID++) {
            //         printf("%f ",decompression_table_[code * n_val_per_byte_ + valID]);
            //     }
            //     printf("\n");
            // }

            centroid_l_ = centroid_l;
            vq_code_l_ = vq_code_l;
        }

        std::vector<uint8_t> decompressSQPackedCode(int n_bit, uint8_t value)
        {
            if (n_bit != 1 && n_bit != 2 && n_bit != 4 && n_bit != 8)
            {
                throw std::invalid_argument("n_bit must be 1, 2, 4, or 8");
            }

            const int count = 8 / n_bit;
            std::vector<uint8_t> result;
            result.reserve(count);
            const uint8_t mask = (1 << n_bit) - 1;

            for (int i = 0; i < count; ++i)
            {
                const int shift = 8 - n_bit * (i + 1);
                uint8_t extracted = (value >> shift) & mask;
                result.push_back(extracted);
            }
            return result;
        }

        void Decode(const uint32_t& itemID, float* item_vec) const
        {
            assert(itemID < n_item_);
            const size_t item_offset = item_n_vec_accu_l_[itemID];
            const uint32_t item_n_vec = item_n_vec_l_[itemID];

            const uint32_t* item_vq_code = vq_code_l_ + item_offset;
            //            const uint8_t *item_residual_code = residual_code_l_.data() + item_offset * vec_dim_;
            const uint8_t* item_residual_code = residual_code_l_ + item_offset * n_packed_val_per_vec_;

            for (uint32_t vecID = 0; vecID < item_n_vec; vecID++)
            {
                const uint32_t vq_code = item_vq_code[vecID];
                std::memcpy(item_vec + (size_t)vecID * vec_dim_, centroid_l_ + (size_t)vq_code * vec_dim_,
                            sizeof(float) * vec_dim_);
                const uint8_t* packed_residual_code_l = item_residual_code + vecID * n_packed_val_per_vec_;

                for (uint32_t packID = 0; packID < n_packed_val_per_vec_; packID++)
                {
                    const uint32_t start_dim = packID * n_val_per_byte_;
                    const uint32_t end_dim = std::min((packID + 1) * n_val_per_byte_, vec_dim_);

                    const uint8_t packed_sq_code = packed_residual_code_l[packID];
                    const float* decompress_weight_l = decompression_table_.data() + packed_sq_code * n_val_per_byte_;
                    for (uint32_t dim = start_dim; dim < end_dim; dim++)
                    {
                        const uint32_t offset = dim - start_dim;
                        const float weight = decompress_weight_l[offset];
                        item_vec[vecID * vec_dim_ + dim] += weight;
                    }
                }

                // for (uint32_t dim = 0; dim < vec_dim_; dim++)
                // {
                //     const uint8_t residual_code = item_residual_code[vecID * vec_dim_ + dim];
                //     assert(residual_code < n_weight_);
                //     const float weight = weight_l_[residual_code];
                //     item_vec[vecID * vec_dim_ + dim] += weight;
                // }

                // normalize
                //                const float ip = std::inner_product(item_vec + vecID * vec_dim_, item_vec + (vecID + 1) * vec_dim_,
                //                                                    item_vec + vecID * vec_dim_, 0.0f);
                //                const float norm = std::sqrt(ip);
                //                for (uint32_t dim = 0; dim < vec_dim_; dim++) {
                //                    item_vec[vecID * vec_dim_ + dim] /= norm;
                //                }
            }
        }
    };
};
#endif //RESIDUALSCALARQUANTIZATIONPACK_HPP
