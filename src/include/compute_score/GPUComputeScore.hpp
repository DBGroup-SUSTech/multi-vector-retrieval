//
// Created by bianzheng on 2023/7/15.
//

#ifndef VECTORSETSEARCH_GPUCOMPUTESCORE_HPP
#define VECTORSETSEARCH_GPUCOMPUTESCORE_HPP

#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

namespace VectorSetSearch {

// error check macros
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// for CUBLAS V2 API
#define cublasCheckErrors(fn) \
    do { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

    struct select_max_functor {

        const float *item_vecs_score_l_device_ptr_; // n_total_item_vecs

        const uint32_t *item_n_vecs_l_device_ptr_; // n_item_
        const uint32_t *item_vecs_offset_l_device_ptr_; // n_item_

        float *item_max_score_l_device_ptr_; // n_item_

        select_max_functor(const float *item_vecs_score_l_device_ptr,
                           const uint32_t *item_n_vecs_l_device_ptr,
                           const uint32_t *item_vecs_offset_l_device_ptr,
                           float *item_max_score_l_device_ptr
        ) {
            this->item_vecs_score_l_device_ptr_ = item_vecs_score_l_device_ptr;
            this->item_n_vecs_l_device_ptr_ = item_n_vecs_l_device_ptr;
            this->item_vecs_offset_l_device_ptr_ = item_vecs_offset_l_device_ptr;
            this->item_max_score_l_device_ptr_ = item_max_score_l_device_ptr;
        }

        __host__ __device__
        void operator()(const int itemID) const {
            const uint32_t start_searchID = item_vecs_offset_l_device_ptr_[itemID];
            const uint32_t end_searchID = start_searchID + item_n_vecs_l_device_ptr_[itemID];
            float max_score = item_vecs_score_l_device_ptr_[start_searchID];
            for (uint32_t searchID = start_searchID; searchID < end_searchID; searchID++) {
                max_score = max_score > item_vecs_score_l_device_ptr_[searchID] ? max_score
                                                                                : item_vecs_score_l_device_ptr_[searchID];
            }
            item_max_score_l_device_ptr_[itemID] = max_score;

        }
    };

    class GPUComputeScore {

        uint64_t n_query_, query_n_vecs_, vec_dim_;
        uint64_t n_item_, n_total_item_vecs_;

        float *query_vecs_l_device_ptr_ = nullptr; // n_query * query_n_vecs * vec_dim
        uint32_t *item_n_vecs_l_device_ptr_ = nullptr; // n_item_
        uint32_t *item_vecs_offset_l_device_ptr_ = nullptr; // n_item_
        float *item_vecs_l_device_ptr_ = nullptr; // n_total_item_vecs * vec_dim

        float *item_vecs_score_l_device_ptr_ = nullptr; // query_n_vecs * n_total_item_vecs
        float *item_max_score_l_device_ptr_ = nullptr; // n_item
        uint32_t *for_each_device_ptr_ = nullptr; // n_item

        float *query_score_device_ptr_ = nullptr; // n_query * n_item

        cublasHandle_t handle_;

    public:


        GPUComputeScore() = default;

        inline GPUComputeScore(const float *query_vecs_l,
                               const uint32_t &n_query, const uint32_t &query_n_vecs, const uint32_t &vec_dim) {
            this->n_query_ = n_query;
            this->vec_dim_ = vec_dim;
            this->query_n_vecs_ = query_n_vecs;

            CHECK(cudaMalloc((void **) &query_vecs_l_device_ptr_, sizeof(float) * n_query_ * query_n_vecs_ * vec_dim_));

            cudaCheckErrors("cuda malloc fail");

            cudaMemcpy(query_vecs_l_device_ptr_, query_vecs_l,
                       sizeof(float) * n_query_ * query_n_vecs_ * vec_dim_,
                       cudaMemcpyHostToDevice);

            cudaCheckErrors("cuda memcpy fail");

            cublasCheckErrors(cublasCreate(&handle_));

        }

        inline void addItem(const float **item_vecs_l, const uint32_t *item_n_vecs_l,
                            const uint32_t &n_item
        ) {
            destroyItem();
            this->n_item_ = n_item;

            std::vector<uint32_t> item_vecs_offset_l(n_item_);
            item_vecs_offset_l[0] = 0;
            for (uint32_t itemID = 1; itemID < n_item_; itemID++) {
                item_vecs_offset_l[itemID] = item_vecs_offset_l[itemID - 1] + item_n_vecs_l[itemID - 1];
            }

            uint32_t n_total_item_vecs = 0;
            for (uint32_t itemID = 0; itemID < n_item_; itemID++) {
                n_total_item_vecs += item_n_vecs_l[itemID];
            }
            this->n_total_item_vecs_ = n_total_item_vecs;

            CHECK(cudaMalloc((void **) &item_n_vecs_l_device_ptr_, sizeof(uint32_t) * n_item_));
            CHECK(cudaMalloc((void **) &item_vecs_offset_l_device_ptr_, sizeof(uint32_t) * n_item_));
            CHECK(cudaMalloc((void **) &item_vecs_l_device_ptr_, sizeof(float) * n_total_item_vecs * vec_dim_));

            CHECK(cudaMalloc((void **) &item_vecs_score_l_device_ptr_,
                             sizeof(float) * query_n_vecs_ * n_total_item_vecs));
            CHECK(cudaMalloc((void **) &item_max_score_l_device_ptr_, sizeof(float) * n_item_));
            CHECK(cudaMalloc((void **) &for_each_device_ptr_, sizeof(uint32_t) * n_item_));

            CHECK(cudaMalloc((void **) &query_score_device_ptr_, sizeof(float) * n_query_ * n_item_));

            cudaCheckErrors("cuda malloc fail");

            cudaMemcpy(item_n_vecs_l_device_ptr_, item_n_vecs_l,
                       sizeof(uint32_t) * n_item_,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(item_vecs_offset_l_device_ptr_, item_vecs_offset_l.data(),
                       sizeof(uint32_t) * n_item_,
                       cudaMemcpyHostToDevice);

            for (uint32_t itemID = 0; itemID < n_item_; itemID++) {
                cudaMemcpy(item_vecs_l_device_ptr_ + (int64_t) item_vecs_offset_l[itemID] * vec_dim_,
                           item_vecs_l[itemID],
                           sizeof(float) * item_n_vecs_l[itemID] * vec_dim_,
                           cudaMemcpyHostToDevice);
            }

            std::vector<uint32_t> seq_l(n_item_);
            std::iota(seq_l.begin(), seq_l.end(), 0);
            CHECK(cudaMemcpy(for_each_device_ptr_,
                             seq_l.data(),
                             sizeof(uint32_t) * n_item_,
                             cudaMemcpyHostToDevice));

            cudaCheckErrors("cuda memcpy fail");

        }

        void computeItemScore(const float **item_vecs_l, const uint32_t *item_n_vecs_l, const uint32_t &n_item,
                              float *const distance_l) {

            addItem(item_vecs_l, item_n_vecs_l, n_item);

            // compute the score between single query vector and the whole item vectors
            // select the maximum score for each query
            // then add the maximum score to a cache
            // then compute the next query vector
            // finally return the score of all item and query

            TimeRecord record;
            double batch_ip_time = 0;
            double batch_max_time = 0;
            double batch_sum_time = 0;

            for (uint32_t queryID = 0; queryID < n_query_; queryID++) {
                float *this_query_score_device_ptr = query_score_device_ptr_ + (int64_t) queryID * n_item_;
                thrust::fill(thrust::device, this_query_score_device_ptr, this_query_score_device_ptr + n_item_, 0.0f);

                const float *this_query_vecs_device_ptr =
                        query_vecs_l_device_ptr_ + (int64_t) queryID * query_n_vecs_ * vec_dim_;
                const float alpha = 1.0;
                const float beta = 0.0;
                cublasCheckErrors(
                        cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                    n_total_item_vecs_, query_n_vecs_, vec_dim_,
                                    &alpha, item_vecs_l_device_ptr_, vec_dim_, this_query_vecs_device_ptr, vec_dim_,
                                    &beta,
                                    item_vecs_score_l_device_ptr_, n_total_item_vecs_
                        ));
                CHECK(cudaDeviceSynchronize());
                batch_ip_time += record.get_elapsed_time_second();

                for (uint32_t query_vecsID = 0; query_vecsID < query_n_vecs_; query_vecsID++) {

//                    std::vector<float> res(2);
//                    cudaMemcpy(res.data(), item_vecs_score_l_device_ptr_,
//                               sizeof(float) * 2,
//                               cudaMemcpyDeviceToHost);
//                    printf("%.3f %.3f\n", res[0], res[1]);

                    record.reset();
                    select_max_functor f(item_vecs_score_l_device_ptr_ + (int64_t) query_vecsID * n_total_item_vecs_,
                                         item_n_vecs_l_device_ptr_, item_vecs_offset_l_device_ptr_,
                                         item_max_score_l_device_ptr_
                    );
                    thrust::for_each(thrust::device,
                                     for_each_device_ptr_, for_each_device_ptr_ + n_item_, f);
                    CHECK(cudaDeviceSynchronize());
                    batch_max_time += record.get_elapsed_time_second();

//                    std::vector<float> res(2);
//                    cudaMemcpy(res.data(), item_max_score_l_device_ptr_,
//                               sizeof(float) * 2,
//                               cudaMemcpyDeviceToHost);
//                    printf("%.3f %.3f\n", res[0], res[1]);

                    record.reset();
                    thrust::transform(thrust::device, item_max_score_l_device_ptr_,
                                      item_max_score_l_device_ptr_ + n_item_,
                                      this_query_score_device_ptr, this_query_score_device_ptr, thrust::plus<float>());
                    CHECK(cudaDeviceSynchronize());
                    batch_sum_time += record.get_elapsed_time_second();

//                    std::vector<float> res(2);
//                    cudaMemcpy(res.data(), this_query_score_device_ptr,
//                               sizeof(float) * 2,
//                               cudaMemcpyDeviceToHost);
//                    printf("%.3f %.3f\n", res[0], res[1]);

                }
            }

            record.reset();
            CHECK(cudaMemcpy(distance_l,
                             query_score_device_ptr_,
                             sizeof(float) * n_query_ * n_item_,
                             cudaMemcpyDeviceToHost));
            const double batch_transfer_time = record.get_elapsed_time_second();
            spdlog::info(
                    "batch ip time {:.3f}s, batch max time {:.3f}s, batch sum time {:.3f}s, batch transfer time {:.3f}s",
                    batch_ip_time, batch_max_time, batch_sum_time, batch_transfer_time);


//            CHECK(cudaDeviceSynchronize());

//            spdlog::info("compute batch time {}s, sort memcpy time {}s", compute_time, sort_memcpy_time);

        }

        void destroyQuery() {
            if (query_vecs_l_device_ptr_ != nullptr) {
                cudaFree(query_vecs_l_device_ptr_);
                query_vecs_l_device_ptr_ = nullptr;
            }

            cublasCheckErrors(cublasDestroy(handle_));
//            cudaDeviceReset();
        }

        void destroyItem() {
            if (item_n_vecs_l_device_ptr_ != nullptr) {
                cudaFree(item_n_vecs_l_device_ptr_);
                item_n_vecs_l_device_ptr_ = nullptr;
            }
            if (item_vecs_offset_l_device_ptr_ != nullptr) {
                cudaFree(item_vecs_offset_l_device_ptr_);
                item_vecs_offset_l_device_ptr_ = nullptr;
            }
            if (item_vecs_l_device_ptr_ != nullptr) {
                cudaFree(item_vecs_l_device_ptr_);
                item_vecs_l_device_ptr_ = nullptr;
            }
            if (item_vecs_score_l_device_ptr_ != nullptr) {
                cudaFree(item_vecs_score_l_device_ptr_);
                item_vecs_score_l_device_ptr_ = nullptr;
            }
            if (item_max_score_l_device_ptr_ != nullptr) {
                cudaFree(item_max_score_l_device_ptr_);
                item_max_score_l_device_ptr_ = nullptr;
            }
            if (for_each_device_ptr_ != nullptr) {
                cudaFree(for_each_device_ptr_);
                for_each_device_ptr_ = nullptr;
            }
            if (query_score_device_ptr_ != nullptr) {
                cudaFree(query_score_device_ptr_);
                query_score_device_ptr_ = nullptr;
            }
        }

        void finishCompute() {
            destroyQuery();
            destroyItem();
        }
    };

}
#endif //VECTORSETSEARCH_GPUCOMPUTESCORE_HPP
