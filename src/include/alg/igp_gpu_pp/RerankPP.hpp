//
// Created by Administrator on 2025/9/5.
//

#ifndef RERANKPP_HPP
#define RERANKPP_HPP

#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <raft/linalg/reduce.cuh>
#include <execution>

namespace VectorSetSearch {
__global__ void decompressAndAddResidual(
    const uint32_t *__restrict__ vq_code_l,
    const uint8_t *__restrict__ sq_code_l,
    const float *__restrict__ centroid_l,
    const float *__restrict__ pack_code2weight_l,
    // pack_code_range_ * n_val_per_byte_

    const uint32_t n_rerank_vec,
    const uint32_t n_thread_per_vec,
    const uint32_t vec_dim,
    const uint32_t n_packed_val_per_vec,
    const uint32_t pack_code_range,
    const uint32_t n_val_per_byte,

    float *__restrict__ cand_vec_l) {
  extern __shared__ float pack_code2weight_l_shared[];
  for (uint32_t i = threadIdx.x; i < pack_code_range * n_val_per_byte; i += blockDim.x) {
    pack_code2weight_l_shared[i] = pack_code2weight_l[i];
  }
  __syncthreads();

  const uint32_t threadID = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= n_rerank_vec * n_thread_per_vec) return;

  const uint32_t stack_vecID = threadID / n_thread_per_vec;
  assert(stack_vecID < n_rerank_vec);
  const uint32_t vec_threadID = threadID % n_thread_per_vec;
  assert(vec_threadID < n_thread_per_vec);

  const uint8_t *residual_code_ptr = sq_code_l + stack_vecID * n_packed_val_per_vec;
  const uint32_t vq_code = vq_code_l[stack_vecID];
  const float *centroid = centroid_l + vq_code * vec_dim;

  float *output_vec = cand_vec_l + stack_vecID * vec_dim;

  for (uint32_t dim = vec_threadID; dim < vec_dim; dim += n_thread_per_vec) {
    float val = __ldg(&centroid[dim]);

    const uint32_t packed_codeID = dim / n_val_per_byte;
    const uint32_t valID_in_pack = dim % n_val_per_byte;

    const uint8_t packed_code = residual_code_ptr[packed_codeID];
    const float weight = pack_code2weight_l_shared[packed_code * n_val_per_byte + valID_in_pack];

    val += weight;

    output_vec[dim] = val;
  }
}

__global__ void reduceMax(const float *__restrict__ score_table,
                          const uint32_t *__restrict__ item_n_vec_l,
                          const uint32_t *__restrict__ item_n_vec_offset_l,
                          const uint32_t query_n_vec,
                          const uint32_t probe_topk,
                          const uint32_t n_rerank_vec,
                          const uint32_t n_thread_per_vec,
                          float *__restrict__ max_score_l) {
  extern __shared__ float s_scores[];

  const uint64_t global_threadID = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  // global_threadID = qvecID * probe_topk * n_thread_per_vec + item_candID * n_thread_per_vec + group_threadID
  const uint64_t groupID = global_threadID / n_thread_per_vec;
  const uint32_t group_threadID = global_threadID % n_thread_per_vec;

  const uint64_t total_tasks = (uint64_t) query_n_vec * probe_topk;
  if (groupID >= total_tasks) {
    return;
  }

  const uint32_t query_vecID = groupID / probe_topk;
  const uint32_t item_candID = groupID % probe_topk;
  const uint32_t item_n_vec = item_n_vec_l[item_candID];
  const uint32_t item_offset = item_n_vec_offset_l[item_candID];
  const float *query_score_row = score_table + (uint64_t) query_vecID * n_rerank_vec + item_offset;

  float thread_max = -FLT_MAX;
  for (uint32_t i = group_threadID; i < item_n_vec; i += n_thread_per_vec) {
    thread_max = fmaxf(thread_max, query_score_row[i]);
  }

  s_scores[threadIdx.x] = thread_max;
  __syncthreads();

  for (unsigned int stride = n_thread_per_vec / 2; stride > 0; stride >>= 1) {
    if (group_threadID < stride) {
      s_scores[threadIdx.x] = fmaxf(s_scores[threadIdx.x], s_scores[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (group_threadID == 0) {
    max_score_l[groupID] = s_scores[threadIdx.x];
  }

  // max_score_l[groupID] = thread_max;
}

class RerankPP {
 public:
  const ItemVecInfoPP *item_vec_info_;
  const VQInfoPP *vq_info_;
  const SQInfoPP *sq_info_;
  const ResourcePP *resource_;

  uint32_t threadID_;
  uint32_t query_n_vec_, probe_topk_;

  std::vector<uint32_t> itemID_l_; // probe_topk
  std::vector<uint32_t> item_n_vec_l_; // probe_topk
  std::vector<uint32_t> item_n_vec_offset_l_; // probe_topk
  uint32_t *vq_code_l_; // probe_topk * max_item_n_vec
  uint8_t *sq_code_l_; // (probe_topk * max_item_n_vec * n_packed_val_per_vec), stack storage

  uint32_t n_rerank_vec_{};
  // transmit by cpu:
  uint32_t *vq_code_l_gpu_; // probe_topk * max_item_n_vec
  uint8_t *sq_code_l_gpu_; // probe_topk * max_item_n_vec * n_packed_val_per_vec, stack storage
  uint32_t *item_n_vec_l_gpu_; // probe_topk
  // computed by GPU
  uint32_t *item_n_vec_offset_l_gpu_; // probe_topk
  float *vec_l_gpu_; // probe_topk * max_item_n_vec * vec_dim, stack storage
  float *score_table_gpu_; // query_n_vec * probe_topk * max_item_n_vec
  float *max_score_l_gpu_; // query_n_vec * probe_topk
  float *score_l_gpu_; // probe_topk

  RerankPP() = default;

  RerankPP(const ItemVecInfoPP *item_vec_info,
           const VQInfoPP *vq_info,
           const SQInfoPP *sq_info,
           const ResourcePP *resource,
           const uint32_t threadID,
           const uint32_t query_n_vec,
           const uint32_t probe_topk) {
    this->item_vec_info_ = item_vec_info;
    this->vq_info_ = vq_info;
    this->sq_info_ = sq_info;
    this->resource_ = resource;

    this->threadID_ = threadID;
    this->query_n_vec_ = query_n_vec;
    this->probe_topk_ = probe_topk;

    itemID_l_.resize(probe_topk_);
    item_n_vec_l_.resize(probe_topk_);
    item_n_vec_offset_l_.resize(probe_topk_);
    CHECK(cudaMallocHost((void **) &vq_code_l_,
                         probe_topk_ * item_vec_info_->max_item_n_vec_ * sizeof(uint32_t)));
    CHECK(cudaMallocHost((void **) &sq_code_l_,
                         probe_topk_ * item_vec_info_->max_item_n_vec_ * sq_info_->n_packed_val_per_vec_ *
                         sizeof(uint8_t)));

    n_rerank_vec_ = probe_topk_ * item_vec_info_->max_item_n_vec_;
    CHECK(cudaMalloc(&sq_code_l_gpu_,
                     probe_topk_ * item_vec_info_->max_item_n_vec_ * sq_info_->n_packed_val_per_vec_ *
                     sizeof(uint8_t)));
    CHECK(cudaMalloc(&item_n_vec_l_gpu_, probe_topk_ * sizeof(uint32_t)));

    CHECK(cudaMalloc(&vq_code_l_gpu_, probe_topk_ * item_vec_info_->max_item_n_vec_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&item_n_vec_offset_l_gpu_, probe_topk_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&vec_l_gpu_,
                     probe_topk_ * item_vec_info_->max_item_n_vec_ * item_vec_info_->vec_dim_ * sizeof(float)));
    CHECK(cudaMalloc(&score_table_gpu_,
                     query_n_vec_ * probe_topk_ * item_vec_info_->max_item_n_vec_ * sizeof(float)));
    CHECK(cudaMalloc(&max_score_l_gpu_, query_n_vec_ * probe_topk_ * sizeof(float)));
    CHECK(cudaMalloc(&score_l_gpu_, probe_topk_ * sizeof(float)));
  }

  void reset() {
    CHECK(cudaMemsetAsync(vec_l_gpu_, 0,
                          n_rerank_vec_ * item_vec_info_->vec_dim_ * sizeof(float),
                          resource_->stream_l_[threadID_]));
    CHECK(cudaMemsetAsync(score_table_gpu_, 0,
                          query_n_vec_ * n_rerank_vec_ * sizeof(float),
                          resource_->stream_l_[threadID_]));
    n_rerank_vec_ = probe_topk_ * item_vec_info_->max_item_n_vec_;
  }

  // output is score_l_gpu_
  void rerank(const float *query_gpu,
              const uint32_t *filter_itemID_l_gpu,
              const uint32_t queryID) {

    transmitSQCode(filter_itemID_l_gpu, n_rerank_vec_);
    // in cpu, gather the item_n_vec, the offset, and the residual code

    RerankFunc(query_gpu, filter_itemID_l_gpu, n_rerank_vec_);
  }

  void transmitSQCode(const uint32_t *filter_itemID_l_gpu, uint32_t &n_rerank_vec) {
    CHECK(cudaMemcpy(itemID_l_.data(), filter_itemID_l_gpu,
                     probe_topk_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    nvtxRangePushA("rerank-transfer-CPU");
#pragma omp parallel for default(none) num_threads(16)
    for (uint32_t candID = 0; candID < probe_topk_; candID++) {
      const uint32_t itemID = itemID_l_[candID];
      if (candID + 1 < probe_topk_) {
        const uint32_t next_itemID = itemID_l_[candID + 1];
        __builtin_prefetch(&item_vec_info_->item_n_vec_l_[next_itemID], 0, 1);
      }

      assert(itemID < item_vec_info_->n_item_);
      const uint32_t item_n_vec = item_vec_info_->item_n_vec_l_[itemID];
      item_n_vec_l_[candID] = item_n_vec;
    }
    std::exclusive_scan(std::execution::par,
                        item_n_vec_l_.begin(),
                        item_n_vec_l_.end(),
                        item_n_vec_offset_l_.data(),
                        0u);
    n_rerank_vec = item_n_vec_l_[probe_topk_ - 1] + item_n_vec_offset_l_[probe_topk_ - 1];
#ifdef DEBUG
    assert(n_rerank_vec == std::accumulate(item_n_vec_l_.data(), item_n_vec_l_.data() + probe_topk_, 0u));
#endif

    // printf("item_n_vec_l_: ");
    // for(uint32_t candID = 0; candID < probe_topk_; candID++) {
    //   printf("%d ", item_n_vec_l_[candID]);
    // }
    // printf("\n");
    // printf("item_n_vec_offset_l_: ");
    // for(uint32_t candID = 0; candID < probe_topk_; candID++) {
    //   printf("%d ", item_n_vec_offset_l_[candID]);
    // }
    // printf("\n");

    // transfer vq code
#pragma omp parallel for default(none) shared(itemID_l_, item_vec_info_, vq_info_, sq_info_, item_n_vec_l_, item_n_vec_offset_l_, probe_topk_, vq_code_l_, sq_code_l_) schedule(dynamic, 4)
    for (uint32_t candID = 0; candID < probe_topk_; candID++) {
      const uint32_t itemID = itemID_l_[candID];
      const size_t global_item_offset = item_vec_info_->item_n_vec_offset_l_[itemID];
      const uint32_t buffer_item_offset = item_n_vec_offset_l_[candID];
      const uint32_t item_n_vec = item_n_vec_l_[candID];

      std::memcpy(vq_code_l_ + buffer_item_offset,
                  vq_info_->vq_code_l_ + global_item_offset,
                  item_n_vec * sizeof(uint32_t));

      const uint8_t *item_residual_code = sq_info_->residual_code_l_ +
                                          global_item_offset * sq_info_->n_packed_val_per_vec_;
      std::memcpy(sq_code_l_ + buffer_item_offset * sq_info_->n_packed_val_per_vec_,
                  item_residual_code,
                  item_n_vec * sq_info_->n_packed_val_per_vec_ * sizeof(uint8_t));
    }

    assert(n_rerank_vec <= probe_topk_ * item_vec_info_->max_item_n_vec_);
    nvtxRangePop();

    resource_->AcquireTransResource(threadID_);
    nvtxRangePushA("rerank-transfer-CPU->GPU");
    CHECK(cudaMemcpyAsync(item_n_vec_l_gpu_, item_n_vec_l_.data(),
                          probe_topk_ * sizeof(uint32_t), cudaMemcpyHostToDevice,
                          resource_->stream_l_[threadID_]));
    CHECK(cudaMemcpyAsync(vq_code_l_gpu_, vq_code_l_,
                          n_rerank_vec * sizeof(uint32_t),
                          cudaMemcpyHostToDevice,
                          resource_->stream_l_[threadID_]));
    CHECK(cudaMemcpyAsync(sq_code_l_gpu_, sq_code_l_,
                          n_rerank_vec * sq_info_->n_packed_val_per_vec_ * sizeof(uint8_t),
                          cudaMemcpyHostToDevice,
                          resource_->stream_l_[threadID_]));
    CHECK(cudaStreamSynchronize(resource_->stream_l_[threadID_]));
    nvtxRangePop();
    resource_->ReleaseTransResource(threadID_);
  }

  // all of them are done in GPU
  void RerankFunc(const float *query_gpu,
                  const uint32_t *filter_itemID_l_gpu,
                  const uint32_t n_rerank_vec) {
    // compute the offset of item_n_vec
    resource_->AcquireCompResource(threadID_);
    nvtxRangePushA("rerank-compute");
    auto exec_policy = thrust::cuda::par.on(resource_->stream_l_[threadID_]);
    thrust::exclusive_scan(exec_policy,
                           item_n_vec_l_gpu_,
                           item_n_vec_l_gpu_ + probe_topk_,
                           item_n_vec_offset_l_gpu_);

    uint32_t block_size = 1024;
    uint32_t n_thread_per_vec = 16;
    uint32_t grid_size = (n_rerank_vec * n_thread_per_vec + block_size - 1) / block_size;
    assert(n_rerank_vec < probe_topk_ * item_vec_info_->max_item_n_vec_);
    decompressAndAddResidual<<<grid_size, block_size,
    VectorSetSearch::SQInfoPP::pack_code_range_ * sq_info_->n_val_per_byte_ * sizeof(float),
    resource_->stream_l_[threadID_]>>>(
        vq_code_l_gpu_,
        sq_code_l_gpu_,
        vq_info_->centroid_l_gpu_,
        sq_info_->pack_code2weight_l_gpu_,

        n_rerank_vec,
        n_thread_per_vec,
        item_vec_info_->vec_dim_,
        sq_info_->n_packed_val_per_vec_,
        VectorSetSearch::SQInfoPP::pack_code_range_,
        sq_info_->n_val_per_byte_,
        vec_l_gpu_);

    // compute vec score
    // (query_n_vec, vec_dim) * (n_rerank_vec, vec_dim) -> (query_n_vec, n_rerank_vec)
    MatrixMultiply(resource_->handle_l_[threadID_],
                   query_gpu,
                   vec_l_gpu_,
                   query_n_vec_,
                   n_rerank_vec,
                   vq_info_->vec_dim_,
                   score_table_gpu_);

    // replace this reduce code
    n_thread_per_vec = 1;
    block_size = 256;
    const uint64_t total_tasks = (uint64_t) query_n_vec_ * probe_topk_;
    const uint64_t total_threads_needed = total_tasks * n_thread_per_vec;
    grid_size = (total_threads_needed + block_size - 1) / block_size;
    const size_t shared_mem_size = block_size * sizeof(float);
    reduceMax<<<grid_size, block_size, shared_mem_size, resource_->stream_l_[threadID_]>>>(
        score_table_gpu_,
        item_n_vec_l_gpu_,
        item_n_vec_offset_l_gpu_,
        query_n_vec_,
        probe_topk_,
        n_rerank_vec,
        n_thread_per_vec,
        max_score_l_gpu_
    );

    // // reduce vec score
    // raft::linalg::reduce(
    //     max_score_l_gpu_,
    //     score_table_gpu_,
    //     item_vec_info_.max_item_n_vec_,
    //     query_n_vec_ * probe_topk_,
    //     std::numeric_limits<float>::lowest(),
    //     true,
    //     true,
    //     resource_->stream_l_[threadID_],
    //     false,
    //     raft::identity_op(),
    //     raft::max_op(),
    //     raft::identity_op()
    // );

    raft::linalg::reduce(
        score_l_gpu_,
        max_score_l_gpu_,
        probe_topk_,
        query_n_vec_,
        0.0f,
        true,
        false,
        resource_->stream_l_[threadID_],
        false,
        raft::identity_op(),
        raft::add_op(),
        raft::identity_op()
    );
    CHECK(cudaStreamSynchronize(resource_->stream_l_[threadID_]));
    nvtxRangePop();
    resource_->ReleaseCompResource(threadID_);
  }

  void finishCompute() {
    if (vq_code_l_ != nullptr) {
      CHECK(cudaFreeHost(vq_code_l_));
      vq_code_l_ = nullptr;
    }

    if (sq_code_l_ != nullptr) {
      CHECK(cudaFreeHost(sq_code_l_));
      sq_code_l_ = nullptr;
    }

    cudaMemFreeMarco(this->sq_code_l_gpu_);
    cudaMemFreeMarco(this->item_n_vec_l_gpu_);
    cudaMemFreeMarco(this->item_n_vec_offset_l_gpu_);
    cudaMemFreeMarco(this->vec_l_gpu_);
    cudaMemFreeMarco(this->score_table_gpu_);
    cudaMemFreeMarco(this->max_score_l_gpu_);
    cudaMemFreeMarco(this->score_l_gpu_);
  }
};
}
#endif //RERANKPP_HPP
