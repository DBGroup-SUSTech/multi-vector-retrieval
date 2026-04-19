//
// Created by bianzheng on 2024/2/20.
//

#ifndef VECTORSETSEARCH_CUDAUTIL_HPP
#define VECTORSETSEARCH_CUDAUTIL_HPP
#include <cublas_v2.h>
#include <cuda.h>

namespace VectorSetSearch {

#define cudaMemFreeMarco(var) \
    if (var != nullptr) {\
      cudaFree(var);\
      var = nullptr;\
      }
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

void printGPUMemoryInfo() {
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);

    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failed to get GPU memory info: %s\n",
                cudaGetErrorString(err));
        return;
    }

    const double free_gb = free / 1024.0 / 1024.0 / 1024.0;
    const double total_gb = total / 1024.0 / 1024.0 / 1024.0;

    printf("GPU Memory Status:\n");
    printf("  Free  : %.2f GB (%.0f MB)\n", free_gb, free / 1024.0 / 1024.0);
    printf("  Total : %.2f GB (%.0f MB)\n", total_gb, total / 1024.0 / 1024.0);
    printf("  Used  : %.2f%%\n", (1.0 - (double)free / total) * 100.0);
}

class TimeRecordCUDA {
    cudaEvent_t start_e_, stop_e_;
public:
    TimeRecordCUDA() {
        CHECK(cudaEventCreate(&start_e_));
        CHECK(cudaEventCreate(&stop_e_));
    }

    void start_record() {
        CHECK(cudaEventRecord(start_e_));
    }

    double get_time_second() {
        CHECK(cudaEventRecord(stop_e_));
        CHECK(cudaEventSynchronize(stop_e_));
        float time_ms;
        CHECK(cudaEventElapsedTime(&time_ms, start_e_, stop_e_));
        double time_second = time_ms * 1e-3;
        return time_second;
    }

    void destroy() {
        CHECK(cudaEventDestroy(start_e_));
        CHECK(cudaEventDestroy(stop_e_));
    }

};

}
#endif //VECTORSETSEARCH_CUDAUTIL_HPP
