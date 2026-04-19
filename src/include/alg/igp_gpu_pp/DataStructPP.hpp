//
// Created by Administrator on 2025/9/5.
//

#ifndef DATASTRUCTPP_HPP
#define DATASTRUCTPP_HPP

#include <cublas_v2.h>
#include <cstdint>
#include <cfloat>
#include <cstddef>
#include <vector>
#include <raft/core/device_resources.hpp>

namespace VectorSetSearch {
struct ItemVecInfoPP {
  uint32_t n_item_{}, max_item_n_vec_{}, min_item_n_vec_{};
  uint32_t vec_dim_{}, vec_dim_float4_{};
  size_t n_vec_{};
  const uint32_t *item_n_vec_l_{}; // n_item
  uint32_t *item_n_vec_l_gpu_{}; // n_item
  const size_t *item_n_vec_offset_l_{}; // n_item
  size_t *item_n_vec_offset_l_gpu_{}; // n_item

  ItemVecInfoPP() = default;

  ItemVecInfoPP(const uint32_t n_item, const uint32_t max_item_n_vec, const uint32_t min_item_n_vec,
                const uint32_t vec_dim,
                const size_t n_vec,
                const uint32_t *item_n_vec_l, const size_t *item_n_vec_offset_l) {
    this->n_item_ = n_item;
    this->max_item_n_vec_ = max_item_n_vec;
    this->min_item_n_vec_ = min_item_n_vec;
    this->vec_dim_ = vec_dim;
    this->vec_dim_float4_ = vec_dim_ / 4;
    if (vec_dim_ % 4 != 0) {
      throw "Vector dimension should be multiple of 4";
    }
    this->n_vec_ = n_vec;
    this->item_n_vec_l_ = item_n_vec_l;
    this->item_n_vec_offset_l_ = item_n_vec_offset_l;

    CHECK(cudaMalloc(&item_n_vec_l_gpu_, n_item_ * sizeof(uint32_t)));
    CHECK(cudaMemcpy(item_n_vec_l_gpu_, item_n_vec_l_, n_item_ * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&item_n_vec_offset_l_gpu_, n_item_ * sizeof(size_t)));
    CHECK(cudaMemcpy(item_n_vec_offset_l_gpu_, item_n_vec_offset_l_, n_item_ * sizeof(size_t),
                     cudaMemcpyHostToDevice));
  }

  void FinishCompute() {
    cudaMemFreeMarco(item_n_vec_l_gpu_);
    cudaMemFreeMarco(item_n_vec_offset_l_gpu_);
  }
};

struct VQInfoPP {
  uint32_t n_centroid_{}, vec_dim_{};
  size_t n_vec_{};
  const float *centroid_l_{}; // n_centroid * vec_dim
  float *centroid_l_gpu_{}; // n_centroid * vec_dim
  const uint32_t *vq_code_l_{}; // n_vec

  VQInfoPP() = default;

  VQInfoPP(const uint32_t n_centroid, const uint32_t vec_dim,
           const size_t n_vec,
           const float *centroid_l, const uint32_t *vq_code_l) {
    this->n_centroid_ = n_centroid;
    this->vec_dim_ = vec_dim;
    this->n_vec_ = n_vec;
    this->centroid_l_ = centroid_l;
    this->vq_code_l_ = vq_code_l;

    CHECK(cudaMalloc(&centroid_l_gpu_, n_centroid_ * vec_dim_ * sizeof(float)));
    CHECK(cudaMemcpy(centroid_l_gpu_, centroid_l_, n_centroid_ * vec_dim_ * sizeof(float),
                     cudaMemcpyHostToDevice));

  }

  void FinishCompute() {
    cudaMemFreeMarco(centroid_l_gpu_);
  }
};

struct IVFInfoPP {
  uint32_t n_centroid_{}, max_n_ele_ivf_{};
  size_t n_ele_ivf_{};
  const size_t *ivf_offset_l_; // n_centroid
  size_t *ivf_offset_l_gpu_{}; // n_centroid
  const uint32_t *ivf_size_l_; // n_centroid
  uint32_t *ivf_size_l_gpu_{}; // n_centroid
  const uint32_t *ivf_; // n_ele_ivf_
  uint32_t *ivf_gpu_; // n_ele_ivf_

  IVFInfoPP() = default;

  IVFInfoPP(const uint32_t n_centroid, const size_t n_ele_ivf,
            const size_t *ivf_offset_l, const uint32_t *ivf_size_l, const uint32_t *ivf) {
    this->n_centroid_ = n_centroid;
    this->n_ele_ivf_ = n_ele_ivf;
    this->ivf_offset_l_ = ivf_offset_l;
    this->ivf_size_l_ = ivf_size_l;
    this->ivf_ = ivf;

    CHECK(cudaMalloc(&ivf_offset_l_gpu_, n_centroid_ * sizeof(size_t)));
    CHECK(cudaMemcpy(ivf_offset_l_gpu_, ivf_offset_l_, n_centroid_ * sizeof(size_t), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&ivf_size_l_gpu_, n_centroid_ * sizeof(uint32_t)));
    CHECK(cudaMemcpy(ivf_size_l_gpu_, ivf_size_l_, n_centroid_ * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&ivf_gpu_, n_ele_ivf_ * sizeof(uint32_t)));
    CHECK(cudaMemcpy(ivf_gpu_, ivf_, n_ele_ivf_ * sizeof(uint32_t), cudaMemcpyHostToDevice));

    this->max_n_ele_ivf_ = *std::max_element(ivf_size_l_, ivf_size_l_ + n_centroid_);
  }

  void FinishCompute() {
    cudaMemFreeMarco(ivf_offset_l_gpu_);
    cudaMemFreeMarco(ivf_size_l_gpu_);
    cudaMemFreeMarco(ivf_gpu_);
  }
};

struct SQInfoPP {
  size_t n_vec_{}, vec_dim_{};

  uint32_t n_bit_, n_val_per_byte_, n_packed_val_per_vec_;

  uint32_t n_weight_{};
  const float *weight_l_{}; // n_weight
  float *weight_l_gpu_{}; // n_weight

  constexpr static uint32_t pack_code_range_ = 256;
  std::vector<float> pack_code2weight_l_; // pack_code_range_ * n_val_per_byte_
  // store the weight of each decompress code
  float *pack_code2weight_l_gpu_; // pack_code_range_ * n_val_per_byte_

  const uint8_t *residual_code_l_{}; // n_vec * n_packed_val_per_vec_

  SQInfoPP() = default;

  SQInfoPP(const size_t n_vec, const size_t vec_dim,
           const uint32_t n_bit, const uint32_t n_val_per_byte, const uint32_t n_packed_val_per_vec,
           const uint32_t n_weight, const float *weight_l,
           const uint8_t *residual_code_l) {
    this->n_vec_ = n_vec;
    this->vec_dim_ = vec_dim;

    this->n_bit_ = n_bit;
    this->n_val_per_byte_ = n_val_per_byte;
    this->n_packed_val_per_vec_ = n_packed_val_per_vec;

    this->n_weight_ = n_weight;
    this->weight_l_ = weight_l;

    pack_code2weight_l_.resize(pack_code_range_ * n_val_per_byte_);
    // generate decompression table
    for (uint32_t code = 0; code < pack_code_range_; code++) {
      std::vector<uint8_t> decompress_sq_code_l = decompressSQPackedCode((int) n_bit_, (uint8_t) code);
      assert(decompress_sq_code_l.size() == n_val_per_byte_);
      float *code_decompress_table = pack_code2weight_l_.data() + code * n_val_per_byte_;
      for (uint32_t valID = 0; valID < n_val_per_byte_; valID++) {
        const uint8_t decompressed_sq_code = decompress_sq_code_l[valID];
        const float weight = weight_l_[decompressed_sq_code];
        code_decompress_table[valID] = weight;
      }
    }

    CHECK(cudaMalloc(&pack_code2weight_l_gpu_, pack_code_range_ * n_val_per_byte_ * sizeof(float)));
    CHECK(cudaMemcpy(pack_code2weight_l_gpu_, pack_code2weight_l_.data(),
                     pack_code_range_ * n_val_per_byte_ * sizeof(float), cudaMemcpyHostToDevice));

    this->residual_code_l_ = residual_code_l;

    CHECK(cudaMalloc(&weight_l_gpu_, n_weight_ * sizeof(float)));
    CHECK(cudaMemcpy(weight_l_gpu_, weight_l_, n_weight_ * sizeof(float), cudaMemcpyHostToDevice));
  }

  std::vector<uint8_t> decompressSQPackedCode(int n_bit, uint8_t value) {
    if (n_bit != 1 && n_bit != 2 && n_bit != 4 && n_bit != 8) {
      throw std::invalid_argument("n_bit must be 1, 2, 4, or 8");
    }

    const int count = 8 / n_bit;
    std::vector<uint8_t> result;
    result.reserve(count);
    const uint8_t mask = (1 << n_bit) - 1;

    for (int i = 0; i < count; ++i) {
      const int shift = 8 - n_bit * (i + 1);
      uint8_t extracted = (value >> shift) & mask;
      result.push_back(extracted);
    }
    return result;
  }

  void FinishCompute() {
    cudaMemFreeMarco(pack_code2weight_l_gpu_);
    cudaMemFreeMarco(weight_l_gpu_);
  }
};

struct VQScoreItemPairPP {
  uint32_t qvecID_;
  uint32_t itemID_;
  float score_;

  __host__ __device__ VQScoreItemPairPP() {
    qvecID_ = std::uint32_t(-1);
    itemID_ = std::uint32_t(-1);
    score_ = FLT_MAX;
  }

  __host__ __device__ VQScoreItemPairPP(
      const uint32_t qvecID,
      const uint32_t itemID,
      const float score) {
    this->qvecID_ = qvecID;
    this->itemID_ = itemID;
    this->score_ = score;
  }
};

struct VecOffsetPP {
  uint32_t item_offset_, item_vecID_, item_n_vec_, probeID_;

  __host__ __device__ VecOffsetPP() {
    item_offset_ = std::uint32_t(-1);
    item_vecID_ = std::uint32_t(-1);
    item_n_vec_ = std::uint32_t(-1);
    probeID_ = std::uint32_t(-1);
  }

  __host__ __device__ VecOffsetPP(
      const uint32_t item_offset,
      const uint32_t item_vecID,
      const uint32_t item_n_vec,
      const uint32_t probeID) {
    this->item_offset_ = item_offset;
    this->item_vecID_ = item_vecID;
    this->item_n_vec_ = item_n_vec;
    this->probeID_ = probeID;
  }
};

struct ResourcePP {

  mutable std::atomic<bool> compute_available_{true};
  mutable std::atomic<bool> trans_available_{true};

  uint32_t n_cpu_thread_;
  std::vector<cudaStream_t> stream_l_; // n_cpu_thread
  std::vector<cublasHandle_t> handle_l_; // n_cpu_thread
  std::vector<raft::device_resources> handle_raft_l_; // n_cpu_thread

  ResourcePP() = default;

  ResourcePP(const uint32_t n_cpu_thread) {

    compute_available_ = true;
    trans_available_ = true;

    this->n_cpu_thread_ = n_cpu_thread;
    stream_l_.resize(n_cpu_thread_);
    for (uint32_t threadID = 0; threadID < n_cpu_thread_; threadID++) {
      CHECK(cudaStreamCreate(&stream_l_[threadID]));
    }
    handle_l_.resize(n_cpu_thread_);
    for (uint32_t threadID = 0; threadID < n_cpu_thread_; threadID++) {
      cublasCheckErrors(cublasCreate(&handle_l_[threadID]));
    }
    handle_raft_l_.resize(n_cpu_thread_);
    for (uint32_t threadID = 0; threadID < n_cpu_thread_; threadID++) {
      raft::resource::set_cuda_stream(handle_raft_l_[threadID],
                                      stream_l_[threadID]);
    }

  }

  void AcquireCompResource(const uint32_t threadID) const {
    while (true) {
      bool expected = true;
      if (compute_available_.compare_exchange_weak(expected, false,
                                                   std::memory_order_acquire)) {
        break;
      }
      std::this_thread::yield();
    }
  }

  void ReleaseCompResource(const uint32_t threadID) const {
    compute_available_.store(true, std::memory_order_release);
  }

  void AcquireTransResource(const uint32_t threadID) const {
    while (true) {
      bool expected = true;
      if (trans_available_.compare_exchange_weak(expected, false,
                                                 std::memory_order_acquire)) {
        break;
      }
      std::this_thread::yield();
    }
  }

  void ReleaseTransResource(const uint32_t threadID) const {
    trans_available_.store(true, std::memory_order_release);
  }

  void FinishCompute() {
    for (uint32_t threadID = 0; threadID < n_cpu_thread_; threadID++) {
      cublasCheckErrors(cublasDestroy(handle_l_[threadID]));
      CHECK(cudaStreamDestroy(stream_l_[threadID]));
    }

  }

};
}
#endif //DATASTRUCTPP_HPP
