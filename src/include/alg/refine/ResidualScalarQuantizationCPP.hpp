//
// Created by Administrator on 2025/5/19.
//

#ifndef RESIDUALSCALARQUANTIZATIONCPP_HPP
#define RESIDUALSCALARQUANTIZATIONCPP_HPP

namespace VectorSetSearch
{

    class ResidualCodeCPP
    {
    public:
        const uint8_t* residual_code_l_;
        std::vector<float> weight_l_;
        uint32_t n_weight_;

        const uint32_t* item_n_vec_l_;
        const size_t* item_n_vec_accu_l_;
        uint32_t n_item_;
        uint32_t vec_dim_;

        const float* centroid_l_;
        const uint32_t* vq_code_l_;

        ResidualCodeCPP() = default;

        ResidualCodeCPP(const uint8_t* residual_code_l,
                        const float* weight_l, const uint32_t n_weight,
                        const uint32_t* item_n_vec_l, const size_t* item_n_vec_accu_l,
                        const float* centroid_l, const uint32_t* vq_code_l,
                        const uint32_t& n_item, const size_t& n_vec,
                        const uint32_t& vec_dim)
        {
            residual_code_l_ = residual_code_l;

            weight_l_.resize(n_weight);
            std::memcpy(weight_l_.data(), weight_l, sizeof(float) * n_weight);
            n_weight_ = n_weight;

            item_n_vec_l_ = item_n_vec_l;
            item_n_vec_accu_l_ = item_n_vec_accu_l;
            n_item_ = n_item;
            vec_dim_ = vec_dim;

            centroid_l_ = centroid_l;
            vq_code_l_ = vq_code_l;
        }

        void Decode(const uint32_t& itemID, float* item_vec) const
        {
            assert(itemID < n_item_);
            const size_t item_offset = item_n_vec_accu_l_[itemID];
            const uint32_t item_n_vec = item_n_vec_l_[itemID];

            const uint32_t* item_vq_code = vq_code_l_ + item_offset;
            //            const uint8_t *item_residual_code = residual_code_l_.data() + item_offset * vec_dim_;
            const uint8_t* item_residual_code = residual_code_l_ + item_offset * vec_dim_;

            for (uint32_t vecID = 0; vecID < item_n_vec; vecID++)
            {
                const uint32_t vq_code = item_vq_code[vecID];
                std::memcpy(item_vec + (size_t)vecID * vec_dim_, centroid_l_ + (size_t)vq_code * vec_dim_,
                            sizeof(float) * vec_dim_);

                for (uint32_t dim = 0; dim < vec_dim_; dim++)
                {
                    const uint8_t residual_code = item_residual_code[vecID * vec_dim_ + dim];
                    assert(residual_code < n_weight_);
                    const float weight = weight_l_[residual_code];
                    item_vec[vecID * vec_dim_ + dim] += weight;
                }

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
#endif //RESIDUALSCALARQUANTIZATIONCPP_HPP
