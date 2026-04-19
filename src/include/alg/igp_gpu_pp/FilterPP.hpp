//
// Created by Administrator on 2025/9/5.
//

#ifndef FILTERPP_HPP
#define FILTERPP_HPP

#include <raft/core/device_resources.hpp>
#include <raft/matrix/detail/select_k-inl.cuh>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <raft/linalg/reduce.cuh>

#include <cooperative_groups.h>

#include "include/alg/igp_gpu_pp/DataStructPP.hpp"

namespace VectorSetSearch
{

__global__ void gather_ivf_data(
    const uint32_t* __restrict__ unique_centID_l,
    const size_t* __restrict__ ivf_offset_l,
    const uint32_t* __restrict__ ivf_size_l,
    const uint32_t* __restrict__ ivf,
    const uint32_t n_unique_centroid,
    const uint32_t max_n_ele_ivf,

    uint32_t* __restrict__ uniqueCentID2itemID_l,
    uint32_t* __restrict__ uniqueCentID2_n_item_l) {

  const uint32_t cand_centID = blockIdx.x;
  if (cand_centID >= n_unique_centroid) {
    return;
  }

  __shared__ uint32_t ivf_size_s;
  __shared__ const uint32_t* centroid_ivf_s;
  __shared__ uint32_t* itemID_l_s;

  if (threadIdx.x == 0) {
    const uint32_t centID = unique_centID_l[cand_centID];
    const size_t ivf_offset = ivf_offset_l[centID];
    const uint32_t ivf_size = ivf_size_l[centID];

    ivf_size_s = ivf_size;
    centroid_ivf_s = ivf + ivf_offset;
    itemID_l_s = uniqueCentID2itemID_l + (size_t)cand_centID * max_n_ele_ivf;

    uniqueCentID2_n_item_l[cand_centID] = ivf_size;
  }

  __syncthreads();

  const uint32_t ivf_size = ivf_size_s;
  if (ivf_size == 0) {
    return;
  }

  const uint32_t* centroid_ivf = centroid_ivf_s;
  uint32_t* itemID_l = itemID_l_s;

  const uint32_t threadID = threadIdx.x;
  const uint32_t n_thread_per_block = blockDim.x;

  for (uint32_t i = threadID; i < ivf_size; i += n_thread_per_block) {
    itemID_l[i] = centroid_ivf[i];
  }
}

__global__ void find_candidate_item(const uint32_t* __restrict__ uniqueCentID2_n_item_l,
                                    const uint32_t* __restrict__ uniqueCentID2itemID_l,
                                    const uint32_t max_n_ele_ivf,
                                    const uint32_t n_unique_centroid, const uint32_t ivf_n_thread,
                                    uint32_t* __restrict__ need_comp_item_score_l)
{
  // const uint64_t threadID = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t unique_cent_candID = blockIdx.x;
  const uint32_t ivf_threadID = threadIdx.x;
  assert(blockIdx.x < n_unique_centroid);

  __shared__ uint32_t ivf_n_item_s;
  __shared__ const uint32_t* ivf_itemID_l_s;

  if (threadIdx.x == 0) {

    ivf_n_item_s = uniqueCentID2_n_item_l[unique_cent_candID];
    ivf_itemID_l_s = uniqueCentID2itemID_l + unique_cent_candID * max_n_ele_ivf;;
  }
  __syncthreads();

  const uint32_t ivf_n_item = ivf_n_item_s;
  const uint32_t* ivf_itemID_l = ivf_itemID_l_s;
  for (uint32_t ivf_candID = ivf_threadID; ivf_candID < ivf_n_item; ivf_candID += ivf_n_thread)
  {
    const uint32_t itemID = ivf_itemID_l[ivf_candID];
    need_comp_item_score_l[itemID] = 1; // means true
  }
}

__global__ void max_qvec_doc_score(const uint32_t* __restrict__ filter_centID_l,
                                   const float* __restrict__ filter_score_l,
                                   const uint32_t* __restrict__ unique_centID_l,
                                   const uint32_t* __restrict__ centID_filter2unique_l,
                                   const uint32_t* __restrict__ uniqueCentID2_n_item_l,
                                   const uint32_t* __restrict__ uniqueCentID2itemID_l,
                                   const uint32_t* __restrict__ itemID2candID_l,
                                   const uint32_t max_n_ele_ivf,
                                   const uint32_t query_n_vec, const uint32_t nprobe, const uint32_t ivf_n_thread,
                                   float* __restrict__ qvec_max_score_l)
{
  // threadID = query_n_vec_ * nprobe_ * ivf_n_thread
  // threadID = qvecID * nprobe * ivf_n_thread + probeID * ivf_n_thread + ivf_threadID
  // const uint64_t threadID = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t pairID = (uint64_t)blockIdx.x;
  const uint32_t qvecID = pairID / nprobe;
  const uint32_t probeID = pairID % nprobe;
  // const uint32_t ivf_threadID = threadID - qvecID * nprobe * ivf_n_thread - probeID * ivf_n_thread;
  const uint32_t ivf_threadID = threadIdx.x;
  assert(blockIdx.x < query_n_vec * nprobe);

  __shared__ float cent_score_s;
  __shared__ uint32_t ivf_n_item_s;
  __shared__ const uint32_t* ivf_itemID_l_s;

  if (threadIdx.x == 0) {
    cent_score_s = filter_score_l[qvecID * nprobe + probeID];
    const uint32_t unique_cent_candID = centID_filter2unique_l[qvecID * nprobe + probeID];
    assert(filter_centID_l[qvecID * nprobe + probeID] == unique_centID_l[unique_cent_candID]);
    ivf_n_item_s = uniqueCentID2_n_item_l[unique_cent_candID];
    ivf_itemID_l_s = uniqueCentID2itemID_l + unique_cent_candID * max_n_ele_ivf;
  }
  __syncthreads();

  const float cent_score = cent_score_s;
  const uint32_t ivf_n_item = ivf_n_item_s;
  const uint32_t* ivf_itemID_l = ivf_itemID_l_s;
  for (uint32_t ivf_candID = ivf_threadID; ivf_candID < ivf_n_item; ivf_candID += ivf_n_thread)
  {
    const uint32_t itemID = ivf_itemID_l[ivf_candID];
    const uint32_t candID = itemID2candID_l[itemID] - 1;
    atomicMax(&qvec_max_score_l[candID * query_n_vec + qvecID], cent_score);
  }
}

__global__ void add_score(const float* __restrict__ qvec_max_score_l,
                          const uint32_t n_item_cand, const uint32_t n_item,
                          const uint32_t query_n_vec,
                          float* __restrict__ item_score_l)
{
  const uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= n_item_cand) return;

  const uint32_t item_candID = threadID;
  const float* max_score_l = qvec_max_score_l + item_candID * query_n_vec;

  float item_score = 0.0f;
  for (uint32_t qvecID = 0; qvecID < query_n_vec; qvecID++)
  {
    item_score += max_score_l[qvecID];
  }
  item_score_l[item_candID] = item_score;
}

class FilterPP
{
 public:
  const ItemVecInfoPP* item_vec_info_;
  const VQInfoPP* vq_info_;
  const IVFInfoPP* ivf_info_;
  const ResourcePP* resource_;

  uint32_t threadID_;
  uint32_t query_n_vec_, nprobe_, probe_topk_;

  uint32_t* unique_centID_l_gpu_; // query_n_vec * nprobe, not full
  uint32_t* centID_filter2unique_l_gpu_; // query_n_vec * nprobe
  std::vector<uint32_t> unique_centID_l_; // query_n_vec * nprobe, not full
  std::vector<uint32_t> uniqueCentID2itemID_l_; // query_n_vec * nprobe * max_n_ele_ivf, padding
  uint32_t* uniqueCentID2_n_item_l_gpu_; // query_n_vec * nprobe_
  uint32_t* uniqueCentID2itemID_l_gpu_; // query_n_vec * nprobe * max_n_ele_ivf

  uint32_t n_item_cand_{};
  uint32_t max_n_item_cand_{}; // n_item / 5 * 4
  uint32_t* need_comp_item_score_l_gpu_{}; // n_item, need reset to 0, i.e., false
  uint32_t* itemID2candID_cache_l_gpu_{}; // n_item
  uint32_t* itemID2candID_l_gpu_{}; // n_item, store the mapping from itemID to (candID + 1)
  float* qvec_max_score_l_gpu_{}; // n_item * query_n_vec, need reset to 0

  uint32_t* item_cand_cache_l_gpu_{}; // n_item, value [0, 1, ..., n_item - 1]
  uint32_t* item_cand_l_gpu_{}; // n_item

  float* item_score_l_gpu_{}; // n_item
  float* filter_score_l_gpu_{}; // probe_topk
  uint32_t* filter_itemID_l_gpu_{}; // probe_topk

  FilterPP() = default;

  FilterPP(const ItemVecInfoPP* item_vec_info, const VQInfoPP* vq_info,
              const IVFInfoPP* ivf_info, const ResourcePP* resource,
              const uint32_t threadID,
              const uint32_t query_n_vec, const uint32_t nprobe, const uint32_t probe_topk)
  {
    this->item_vec_info_ = item_vec_info;
    this->vq_info_ = vq_info;
    this->ivf_info_ = ivf_info;
    this->resource_ = resource;

    this->threadID_ = threadID;
    this->query_n_vec_ = query_n_vec;
    this->nprobe_ = nprobe;
    this->probe_topk_ = probe_topk;

    spdlog::info("after modify the filter");

    CHECK(cudaMalloc(&unique_centID_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&centID_filter2unique_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
    unique_centID_l_.resize(query_n_vec_ * nprobe_);
    uniqueCentID2itemID_l_.resize(query_n_vec_ * nprobe_ * ivf_info_->max_n_ele_ivf_);
    CHECK(cudaMalloc(&uniqueCentID2_n_item_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&uniqueCentID2itemID_l_gpu_, query_n_vec_ * nprobe_ * ivf_info_->max_n_ele_ivf_ *
                                                  sizeof(uint32_t)));

    max_n_item_cand_ = (item_vec_info_->n_item_ > 21000000 ? item_vec_info_->n_item_ / 3 * 2 : item_vec_info_->n_item_);
    n_item_cand_ = max_n_item_cand_;
    CHECK(cudaMalloc(&need_comp_item_score_l_gpu_, item_vec_info_->n_item_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&itemID2candID_cache_l_gpu_, item_vec_info_->n_item_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&itemID2candID_l_gpu_, item_vec_info_->n_item_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&qvec_max_score_l_gpu_, max_n_item_cand_ * query_n_vec_ * sizeof(float)));

    CHECK(cudaMalloc(&item_cand_cache_l_gpu_, item_vec_info_->n_item_ * sizeof(uint32_t)));
    thrust::sequence(thrust::device, item_cand_cache_l_gpu_,
                     item_cand_cache_l_gpu_ + item_vec_info_->n_item_, 0);
    CHECK(cudaMalloc(&item_cand_l_gpu_, item_vec_info_->n_item_ * sizeof(uint32_t)));

    CHECK(cudaMalloc(&item_score_l_gpu_, item_vec_info_->n_item_ * sizeof(float)));
    CHECK(cudaMalloc(&filter_score_l_gpu_, probe_topk_ * sizeof(float)));
    CHECK(cudaMalloc(&filter_itemID_l_gpu_, probe_topk_ * sizeof(uint32_t)));


    size_t free_mem, total_mem;
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    double free_mem_mb = free_mem / (1024.0 * 1024.0);
    double total_mem_mb = total_mem / (1024.0 * 1024.0);
    double used_mem_mb = total_mem_mb - free_mem_mb;
    double used_percentage = (used_mem_mb / total_mem_mb) * 100.0;

    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "[" << "after FilterPPGAO" << "] "
              << "used: " << used_mem_mb << " MB (" << used_percentage << "%), "
              << "free: " << free_mem_mb << " MB, "
              << "total: " << total_mem_mb << " MB" << std::endl;
  }

  void reset()
  {
    CHECK(cudaMemsetAsync(qvec_max_score_l_gpu_, 0,
                          n_item_cand_ * query_n_vec_ * sizeof(float),
                          resource_->stream_l_[threadID_]));
    CHECK(cudaMemsetAsync(need_comp_item_score_l_gpu_, 0,
                          item_vec_info_->n_item_ * sizeof(uint32_t),
                          resource_->stream_l_[threadID_]));
    n_item_cand_ = max_n_item_cand_;
  }

  // output: filter_itemID_l_gpu_
  void filter(const float* filter_cent_score_l_gpu, const uint32_t* filter_centID_l_gpu)
  {
    // compute the unique centID
    /*
     * input: filter_centID_l_gpu
     * output: (1) unique_centID_l_gpu_, centID_filter2unique_l_gpu_, call unique function
     * (2) uniqueCentID2_n_item_l_gpu_, compute the offset value
     * (3) uniqueCentID2itemID_l_gpu_, transform cpu ivf element to gpu
     */
    nvtxRangePushA("generate candidate");
    CHECK(cudaMemcpyAsync(unique_centID_l_gpu_, filter_centID_l_gpu,
                          query_n_vec_ * nprobe_ * sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                          resource_->stream_l_[threadID_]));
    auto exec_policy = thrust::cuda::par.on(resource_->stream_l_[threadID_]);
    thrust::sort(exec_policy, unique_centID_l_gpu_, unique_centID_l_gpu_ + query_n_vec_ * nprobe_);
    auto unique_end_ptr = thrust::unique(exec_policy,
                                         unique_centID_l_gpu_, unique_centID_l_gpu_ + query_n_vec_ * nprobe_);
    uint32_t n_unique_centroid = unique_end_ptr - unique_centID_l_gpu_;
    // transmit to cpu to get the inverted file data, at the same time compute centID_filter2unique_l_gpu_ and uniqueCentID2_n_item_l_gpu_
    thrust::lower_bound(exec_policy,
                        unique_centID_l_gpu_, unique_centID_l_gpu_ + n_unique_centroid,
                        filter_centID_l_gpu, filter_centID_l_gpu + query_n_vec_ * nprobe_,
                        centID_filter2unique_l_gpu_);

    uint64_t blockSize = 256;
    uint32_t gridSize = n_unique_centroid;
    uint64_t shared_memory_size = sizeof(uint32_t) + sizeof(const uint32_t*) + sizeof(uint32_t*);
    gather_ivf_data<<<gridSize, blockSize,
        shared_memory_size, resource_->stream_l_[threadID_]>>>
    (unique_centID_l_gpu_, ivf_info_->ivf_offset_l_gpu_,
        ivf_info_->ivf_size_l_gpu_, ivf_info_->ivf_gpu_,
        n_unique_centroid, ivf_info_->max_n_ele_ivf_,

        uniqueCentID2itemID_l_gpu_,
        uniqueCentID2_n_item_l_gpu_);

    // compute the max score
    constexpr uint32_t n_thread_warp = 32;
    // integer multiple of n_thread_warp
    uint64_t ivf_n_thread = (ivf_info_->max_n_ele_ivf_ + n_thread_warp - 1) / n_thread_warp;
    blockSize = ivf_n_thread;
    gridSize = n_unique_centroid;
    shared_memory_size = sizeof(uint32_t) + sizeof(const uint32_t*);
    find_candidate_item<<<gridSize, blockSize, 0, resource_->stream_l_[threadID_]>>>
    (uniqueCentID2_n_item_l_gpu_, uniqueCentID2itemID_l_gpu_,
        ivf_info_->max_n_ele_ivf_, n_unique_centroid, ivf_n_thread,
        need_comp_item_score_l_gpu_);

    thrust::exclusive_scan(exec_policy, need_comp_item_score_l_gpu_,
                           need_comp_item_score_l_gpu_ + item_vec_info_->n_item_,
                           itemID2candID_cache_l_gpu_);
    thrust::transform(exec_policy,
                      thrust::make_zip_iterator(
                          thrust::make_tuple(need_comp_item_score_l_gpu_, itemID2candID_cache_l_gpu_)),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(need_comp_item_score_l_gpu_ + item_vec_info_->n_item_,
                                             itemID2candID_cache_l_gpu_ + item_vec_info_->n_item_)),
                      itemID2candID_l_gpu_,
                      [] __host__ __device__ (const thrust::tuple<uint32_t, uint32_t>& t) -> uint32_t
    {
      uint32_t flag = thrust::get<0>(t);
      uint32_t prefix = thrust::get<1>(t);
      return flag ? (prefix + 1) : 0;
    }
    );// for itemID2candID_l_gpu, if it is a candidate, then set it as the candID + 1, else it is not a candidate and we set it as 0

    blockSize = ivf_n_thread;
    gridSize = query_n_vec_ * nprobe_;
    shared_memory_size = sizeof(float) + sizeof(uint32_t) + sizeof(const uint32_t*);
    max_qvec_doc_score<<<gridSize, blockSize, shared_memory_size, resource_->stream_l_[threadID_]>>>
    (filter_centID_l_gpu, filter_cent_score_l_gpu,
        unique_centID_l_gpu_, centID_filter2unique_l_gpu_,
        uniqueCentID2_n_item_l_gpu_, uniqueCentID2itemID_l_gpu_,
        itemID2candID_l_gpu_,
        ivf_info_->max_n_ele_ivf_,
        query_n_vec_, nprobe_, ivf_n_thread,
        qvec_max_score_l_gpu_);

    // count the index of the item candidates in need_comp_item_score_l_gpu_, store in item_cand_l_gpu_
    uint32_t* need_comp_item_score_l_gpu = this->need_comp_item_score_l_gpu_;
    auto end_iter = thrust::copy_if(
        exec_policy,
        item_cand_cache_l_gpu_,
        item_cand_cache_l_gpu_ + item_vec_info_->n_item_,
        item_cand_l_gpu_,
    [need_comp_item_score_l_gpu] __device__ (uint32_t index)
    {
      return need_comp_item_score_l_gpu[index] == true;
    }
    );
    n_item_cand_ = end_iter - item_cand_l_gpu_;
    n_item_cand_ = std::min(max_n_item_cand_, n_item_cand_);

    //    spdlog::info("start filter-add_score");
    blockSize = 1024;
    gridSize = (n_item_cand_ + blockSize - 1) / blockSize;
    raft::linalg::reduce(
        item_score_l_gpu_,
        qvec_max_score_l_gpu_,
        query_n_vec_,
        n_item_cand_,
        0.0f,
        true,
        true,
        resource_->stream_l_[threadID_],
        false,
        raft::identity_op(),
        raft::add_op(),
        raft::identity_op()
    );
    // add_score<<<gridSize, blockSize>>>(qvec_max_score_l_gpu_,
    // n_item_cand,
    // item_vec_info_.n_item_, query_n_vec_,
    // item_score_l_gpu_);

    bool select_min = false;
    raft::matrix::detail::select_k(resource_->handle_raft_l_[threadID_],
                                   item_score_l_gpu_,
                                   item_cand_l_gpu_,
                                   1,
                                   n_item_cand_,
                                   probe_topk_,
                                   filter_score_l_gpu_,
                                   filter_itemID_l_gpu_,
                                   select_min);
    CHECK(cudaStreamSynchronize(resource_->stream_l_[threadID_]));
    nvtxRangePop();
  }

  void finishCompute()
  {
    cudaMemFreeMarco(this->unique_centID_l_gpu_);
    cudaMemFreeMarco(this->centID_filter2unique_l_gpu_);
    cudaMemFreeMarco(this->uniqueCentID2_n_item_l_gpu_);
    cudaMemFreeMarco(this->uniqueCentID2itemID_l_gpu_);

    cudaMemFreeMarco(this->need_comp_item_score_l_gpu_);
    cudaMemFreeMarco(this->itemID2candID_cache_l_gpu_);
    cudaMemFreeMarco(this->itemID2candID_l_gpu_);
    cudaMemFreeMarco(this->qvec_max_score_l_gpu_);

    cudaMemFreeMarco(this->item_cand_cache_l_gpu_);
    cudaMemFreeMarco(this->item_cand_l_gpu_);

    cudaMemFreeMarco(this->item_score_l_gpu_);
    cudaMemFreeMarco(this->filter_score_l_gpu_);
    cudaMemFreeMarco(this->filter_itemID_l_gpu_);
  }
};
}
#endif //FILTERPP_HPP
