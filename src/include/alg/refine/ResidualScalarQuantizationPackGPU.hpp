//
// Created by Administrator on 7/2/2025.
//

#ifndef RESIDUALSCALARQUANTIZATIONPACKGPU_HPP
#define RESIDUALSCALARQUANTIZATIONPACKGPU_HPP

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>

#include "include/struct/TypeDef.hpp"
#include "include/util/CUDAUtil.hpp"

namespace VectorSetSearch {

__device__  uint8_t compress_gpu(const uint8_t *sq_code_l,
                                 const uint8_t mask,
                                 const uint32_t n_bit,
                                 const uint32_t n_val_per_byte) {
  uint8_t result = 0;

  for (int valID = 0; valID < n_val_per_byte; ++valID) {
    result = (result << n_bit) | (sq_code_l[valID] & mask);
  }
  return result;
}

__global__ void compressSQCode(const float *centroid_l,
                               const float *cutoff_l,
                               const float *vec_l,
                               const uint32_t *vq_code_l,
                               const uint32_t n_centroid, const uint32_t vec_dim,
                               const uint32_t n_cutoff, const uint32_t n_vec,
                               const uint32_t n_thread_per_vec,
                               const uint32_t n_bit, const uint32_t n_val_per_byte,
                               const uint32_t n_packed_val_per_vec,
                               uint8_t *sq_code_buffer_l,
                               uint8_t *residual_code_l) {

  // threadID = n_vec * n_thread_per_vec
  const uint32_t threadID = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= n_vec * n_thread_per_vec) return;
  const uint32_t vecID = threadID / n_thread_per_vec;
  const uint32_t vec_threadID = threadID % n_thread_per_vec;
  const uint32_t vq_code = vq_code_l[vecID];
  const float *centroid = centroid_l + vq_code * vec_dim;
  const float *vec = vec_l + vecID * vec_dim;
  uint8_t *vec_residual_code = residual_code_l + vecID * n_packed_val_per_vec;
  uint8_t *sq_code_l = sq_code_buffer_l + threadID * n_val_per_byte;

  const uint8_t mask = (1 << n_bit) - 1;

  for (uint32_t packed_codeID = vec_threadID; packed_codeID < n_packed_val_per_vec;
       packed_codeID += n_thread_per_vec) {
    const uint32_t start_dim = packed_codeID * n_val_per_byte;

    for (uint32_t valID = 0; valID < n_val_per_byte; valID++) {
      const uint32_t dim = start_dim + valID;
      assert(dim < vec_dim);
      uint8_t sq_code = n_cutoff;
      if (dim < vec_dim) {
        const float error = vec[dim] - centroid[dim];

        for (uint32_t cutoffID = 0; cutoffID < n_cutoff; ++cutoffID) {
          if (cutoff_l[cutoffID] > error) {
            sq_code = cutoffID;
            break;
          }
        }
      } else {
        sq_code = 0;
      }
      sq_code_l[valID] = sq_code;
    }
    const uint8_t packed_code = compress_gpu(sq_code_l, mask, n_bit, n_val_per_byte);
    vec_residual_code[packed_codeID] = packed_code;
  }
}

class CompressResidualCodeGPU {
 public:
  uint32_t n_centroid_, vec_dim_;
  std::vector<float> centroid_l_; // n_centroid * vec_dim

  uint32_t n_cutoff_;
  std::vector<float> cutoff_l_; // n_cutoff

  uint32_t n_weight_;
  std::vector<float> weight_l_; // n_weight

  uint32_t n_bit_;
  uint32_t n_val_per_byte_; // n_bit_per_byte / n_bit_;
  uint32_t n_packed_val_per_vec_; // (vec_dim_ + n_val_per_byte_ - 1) / n_val_per_byte_;

  float *centroid_l_gpu_{}; // n_centroid * vec_dim
  float *cutoff_l_gpu_{}; // n_cutoff
  float *weight_l_gpu_{}; // n_weight

  CompressResidualCodeGPU() = default;

  CompressResidualCodeGPU(const pyarray_float &centroid_l_py,
                          const pyarray_float &cutoff_l_py,
                          const pyarray_float &weight_l_py,
                          const uint32_t n_bit) {
    spdlog::info("CompressResidualCodeGPU------------------------");
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
    if ((n_bit_ != 1) && (n_bit_ != 2) && (n_bit_ != 4) && (n_bit_ != 8)) {
      spdlog::error("n_bit is not in range {1, 2, 4, 8}, program exit");
      exit(-1);
    }
    constexpr uint32_t n_bit_per_byte = 8;
    this->n_val_per_byte_ = n_bit_per_byte / n_bit_;
    assert(n_bit_per_byte % n_bit_ == 0);
    this->n_packed_val_per_vec_ = (vec_dim_ + n_val_per_byte_ - 1) / n_val_per_byte_;

    CHECK(cudaMalloc(&centroid_l_gpu_, n_centroid_ * vec_dim_ * sizeof(float)));
    CHECK(cudaMemcpy(centroid_l_gpu_, centroid_l_.data(), n_centroid_ * vec_dim_ * sizeof(float),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&cutoff_l_gpu_, n_cutoff_ * sizeof(float)));
    CHECK(cudaMemcpy(cutoff_l_gpu_, cutoff_l_.data(), n_cutoff_ * sizeof(float),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&weight_l_gpu_, n_weight_ * sizeof(float)));
    CHECK(cudaMemcpy(weight_l_gpu_, weight_l_.data(), n_weight_ * sizeof(float),
                     cudaMemcpyHostToDevice));
  }

  std::vector<uint8_t> compute_residual_code_cpu(const pyarray_float &vec_l_py, const pyarray_uint32 &code_l_py) const {
    const size_t n_vec = vec_l_py.shape(0);
    assert(n_vec == code_l_py.shape(0));

    const float *vec_l = vec_l_py.data();
    const uint32_t *code_l = code_l_py.data();
    spdlog::info("ComputeResidualCode");

    std::vector<uint8_t> residual_code_l((size_t) n_vec * n_packed_val_per_vec_);
    // #pragma omp parallel for default(none) shared(n_vec, code_l, vec_l, vec_dim_, centroid_l, residual_code_l, n_weight, weight_l, cutoff_l, n_centroid, n_cutoff, residual_norm_l)
    for (size_t vecID = 0; vecID < n_vec; vecID++) {
      const uint32_t code = code_l[vecID];
      assert(code < n_centroid_);
      const float *vec = vec_l + (size_t) vecID * vec_dim_;

      std::vector<float> centroid(centroid_l_.data() + code * vec_dim_,
                                  centroid_l_.data() + (code + 1) * vec_dim_);

      uint8_t *residual_code = residual_code_l.data() + vecID * n_packed_val_per_vec_;
      for (uint32_t packID = 0; packID < n_packed_val_per_vec_; packID++) {
        const uint32_t start_dim = packID * n_val_per_byte_;
        std::vector<uint8_t> sq_code_l(n_val_per_byte_);
        for (uint32_t valID = 0; valID < n_val_per_byte_; valID++) {
          const uint32_t dim = start_dim + valID;
          uint8_t sq_code;
          if (dim < vec_dim_) {
            const float error = vec[dim] - centroid[dim];
            const float *ptr = std::upper_bound(cutoff_l_.data(), cutoff_l_.data() + n_cutoff_, error,
                                                [](const float &ele, const float &error) {
                                                  return ele < error;
                                                });
            sq_code = ptr - cutoff_l_.data();
            assert(sq_code < n_weight_);
            assert(sq_code <= n_cutoff_);
            if (!((sq_code == 0 && error <= cutoff_l_[sq_code]) ||
                (0 < sq_code && sq_code < n_cutoff_ && cutoff_l_[sq_code - 1] <= error && error < cutoff_l_[sq_code]) ||
                (sq_code == n_cutoff_ && cutoff_l_[sq_code - 1] <= error))) {
              printf("error %.3f, sq_code %d\n", error, sq_code);
              printf("weight: ");
              for (uint32_t i = 0; i < n_cutoff_; i++) {
                printf("%.3f ", cutoff_l_[i]);
              }
              printf("\n");
            }
            assert((sq_code == 0 && error <= cutoff_l_[sq_code]) ||
                (0 < sq_code && sq_code < n_cutoff_ && cutoff_l_[sq_code - 1] <= error && error < cutoff_l_[sq_code]) ||
                (sq_code == n_cutoff_ && cutoff_l_[sq_code - 1] <= error));
            assert(n_weight_ == (1 << n_bit_));
          } else {
            sq_code = 0;
          }
          sq_code_l[valID] = sq_code;
        }
        const uint8_t packed_code = compress(sq_code_l.data());
        residual_code[packID] = packed_code;
      }
    }

    return residual_code_l;
  }

  uint8_t compress(const uint8_t *sq_code_l) const {
    uint8_t result = 0;
    const uint8_t mask = (1 << n_bit_) - 1;

    for (int valID = 0; valID < n_val_per_byte_; ++valID) {
      result = (result << n_bit_) | (sq_code_l[valID] & mask);
    }
    return result;
  }

  py::tuple compute_residual_code_gpu(const pyarray_float &vec_l_py, const pyarray_uint32 &code_l_py) const {
    const size_t n_vec = vec_l_py.shape(0);
    assert(n_vec == code_l_py.shape(0));
    assert(vec_dim_ == vec_l_py.shape(1));

    const float *vec_l = vec_l_py.data();
    const uint32_t *code_l = code_l_py.data();
    spdlog::info("ComputeResidualCode");

    float *vec_l_gpu{}; // n_vec * vec_dim
    uint32_t *vq_code_l_gpu{}; // n_vec
    uint8_t *residual_code_l_gpu{}; // n_vec * n_packed_val_per_vec_

    CHECK(cudaMalloc(&vec_l_gpu, n_vec * vec_dim_ * sizeof(float)));
    CHECK(cudaMemcpy(vec_l_gpu, vec_l, n_vec * vec_dim_ * sizeof(float),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&vq_code_l_gpu, n_vec * sizeof(uint32_t)));
    CHECK(cudaMemcpy(vq_code_l_gpu, code_l, n_vec * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&residual_code_l_gpu, n_vec * n_packed_val_per_vec_ * sizeof(uint8_t)));

    const uint32_t block_size = 1024;
    const uint32_t n_thread_per_vec = std::min(16u, n_packed_val_per_vec_);
    const uint32_t grid_size = (n_vec * n_thread_per_vec + block_size - 1) / block_size;
    uint8_t *sq_code_buffer_l_gpu{}; // n_vec * n_thread_per_vec * n_val_per_byte_
    CHECK(cudaMalloc(&sq_code_buffer_l_gpu,
                     n_vec * n_thread_per_vec * n_val_per_byte_ * sizeof(uint8_t)));

    compressSQCode<<<grid_size, block_size>>>(centroid_l_gpu_, cutoff_l_gpu_,
                                              vec_l_gpu, vq_code_l_gpu,
                                              n_centroid_, vec_dim_,
                                              n_cutoff_, n_vec, n_thread_per_vec,
                                              n_bit_, n_val_per_byte_, n_packed_val_per_vec_,
                                              sq_code_buffer_l_gpu, residual_code_l_gpu);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    std::vector<uint8_t> residual_code_l((size_t) n_vec * n_packed_val_per_vec_);
    CHECK(cudaMemcpy(residual_code_l.data(), residual_code_l_gpu,
                     n_vec * n_packed_val_per_vec_ * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    cudaMemFreeMarco(vec_l_gpu);
    cudaMemFreeMarco(vq_code_l_gpu);
    cudaMemFreeMarco(residual_code_l_gpu);
    cudaMemFreeMarco(sq_code_buffer_l_gpu);

    return py::make_tuple(residual_code_l);
  }

};

};
#endif //RESIDUALSCALARQUANTIZATIONPACKGPU_HPP
