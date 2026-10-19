#!/usr/bin/env python3
"""CUDA regression for FilterPP's launch sizing, without the RAFT build.

Extract the two unchanged kernel bodies and the launch-size expression from
the selected header. A tiny atomicMax shim replaces the RAFT-provided overload;
this test exercises nonnegative exact-representable scores, not RAFT numerics.
Requires Python 3 and nvcc. Run from any directory:
  python3 tests/test_long_posting_launch.py --nvcc /path/to/nvcc
Use --header to test an unpatched header: it must exit nonzero on long lists.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


PREAMBLE = r'''
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)
__device__ float atomicMax(float* p, float v) {
  unsigned int* u = reinterpret_cast<unsigned int*>(p);
  unsigned int old = *u, assumed;
  do { assumed = old; old = atomicCAS(u, assumed,
      __float_as_uint(fmaxf(v, __uint_as_float(assumed)))); } while (old != assumed);
  return __uint_as_float(old);
}
template<class T> T* device(const std::vector<T>& v) {
  T* p; CHECK(cudaMalloc(&p, v.size()*sizeof(T)));
  CHECK(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); return p;
}
'''

MAIN = r'''
int main() {
  cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s maxThreadsPerBlock=%d\n", prop.name, prop.maxThreadsPerBlock);
  for (uint32_t length : {1u,31u,32u,33u,32768u,32769u,404381u,1048577u}) {
    struct { uint32_t max_n_ele_ivf_; } info{length}; auto* ivf_info_ = &info;
    constexpr uint32_t n_thread_warp = 32;
    LAUNCH_EXPRESSION
    std::printf("length=%u threads=%llu ", length, (unsigned long long)ivf_n_thread);
    std::fflush(stdout);
    // Two posting lists: even / odd IDs. Two query terms and two probes each.
    const uint32_t ndoc = 2*length+1;
    std::vector<uint32_t> posting(2*length), counts{length,length}, flags(ndoc,0), map(ndoc);
    for(uint32_t i=0;i<length;++i) {posting[i]=2*i; posting[length+i]=2*i+1;}
    for(uint32_t i=0;i<ndoc;++i) map[i]=i+1;
    auto* dp=device(posting); auto* dn=device(counts); auto* df=device(flags); auto* dm=device(map);
    find_candidate_item<<<2,ivf_n_thread>>>(dn,dp,length,2,ivf_n_thread,df);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(flags.data(),df,ndoc*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    for(uint32_t i=0;i<ndoc;++i) if(flags[i] != (i<2*length ? 1u:0u)) return 2;
    std::vector<uint32_t> cent{0,1}, probe{0,1,0,1};
    std::vector<float> values{1,2,3,4}, scores(ndoc*2,0);
    auto* dc=device(cent);auto* di=device(probe);auto* dv=device(values);auto* ds=device(scores);
    max_qvec_doc_score<<<4,ivf_n_thread>>>(di,dv,dc,di,dn,dp,dm,length,2,2,ivf_n_thread,ds);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(scores.data(),ds,scores.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(uint32_t i=0;i<ndoc;++i) for(uint32_t q=0;q<2;++q) {
      float want=i<2*length ? float(1+(i%2)+2*q) : 0;
      if(scores[i*2+q]!=want) return 3;
    }
    CHECK(cudaFree(dp));CHECK(cudaFree(dn));CHECK(cudaFree(df));CHECK(cudaFree(dm));
    CHECK(cudaFree(dc));CHECK(cudaFree(di));CHECK(cudaFree(dv));CHECK(cudaFree(ds));
    std::puts("PASS candidates and scores");
  }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header', type=Path, default=Path(__file__).resolve().parents[1] /
                        'src/include/alg/igp_gpu_pp/FilterPP.hpp')
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', default='native')
    parser.add_argument('--ccbin', help='Optional CUDA-compatible host C++ compiler')
    args = parser.parse_args()
    source = args.header.read_text()
    kernels = source[source.index('__global__ void find_candidate_item'):
                     source.index('__global__ void add_score')]
    launch = re.search(r'uint64_t ivf_n_thread\s*=.*?;', source, re.S)
    if launch is None:
        raise RuntimeError('FilterPP launch expression not found')
    with tempfile.TemporaryDirectory(prefix='gigp-launch-') as tmp:
        cu = Path(tmp) / 'test.cu'
        exe = Path(tmp) / 'test'
        cu.write_text(PREAMBLE + kernels + MAIN.replace('LAUNCH_EXPRESSION', launch.group()))
        compiler = ['-ccbin', args.ccbin] if args.ccbin else []
        subprocess.run([args.nvcc, *compiler, '-std=c++17', '-O2', '-arch='+args.arch, str(cu), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
