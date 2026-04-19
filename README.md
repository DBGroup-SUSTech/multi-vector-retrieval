# Multi-Vector Retrieval

## Introduction

This repository shows the implementation of `IGP: Efficient Multi-Vector Retrieval via Proximity Graph Index` and `GIGP+: A GPU-based Solution for Multi-Vector Retrieval`, a CPU-based and GPU-based multi-vector retrieval solution based on ColBERT.

## How to build and run

Note that the python version must be 3.8

1. download the ColBERT pretrain model in https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
2. decompress the ColBERT model and move it into the file `{project_file}/multi-vector-retrieval-data/RawData/colbert-pretrain`
3. `mv {project_file}/multi-vector-retrieval-data /home/{username}/Dataset/multi-vector-retrieval`
4. set username in the following scripts described

Note that you should move the project path as `/home/{username}/`

Install the necessary library when the system reports an error

## Important scripts

- `script/data/build_index.py`, build the index of plaid, it also generates the embedding
- `script/data/build_index_by_sample.py`, if you want to test the program in a small dataset, use this one
- `script/evaluation/eval_igp.py`, implementation of IGP. This implementation uses the single-threaded version from the GIGP+ paper, which is faster than the original IGP implementation while producing identical results.
- `script/evaluation/eval_igp_gpu_pp.py`, implementation of GIGP+

## Parameter description

- `n_probe` stands for $\phi_{pb}$ in the paper
- `probe_topk` stands for $\phi_{ref}$ in the paper

## Some tips for library installation:

1. pybind: build file, then make install

2. eigen: sudo apt install libeigen3-dev

3. to install spdlog, you should install the spdlog-1.11.0
   when build and install, run `cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && make -j` to build the spdlog

4. when encountering the error: no CUDA runtime found, using CUDA_HOME = 'xxxx'

It may because of the incompatible version between CUDA and pytorch, the correct answer is to reinstall the pytorch and not do anything for the CUDA
A command that has worked before:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

5. When using the colbert to connect to huggingface. If it report the error show that huggingface cannot connect, then please add the proxy for downloading.
   The detail is to add the proxy configuration at the line 'loadedConfig  = AutoConfig.from_pretrained(name_or_path)' in ColBERT/colbert/modeling/hf_colbert.py
   You can learn how to use the proxychains for climbing the "wall"

6. to let PLAID run in the CPU, you can set the following

os.environ["CUDA_VISIBLE_DEVICES"] = ""

export CUDA_VISIBLE_DEVICES=""

in bash, setting CUDA_VISIBLE_DEVICES=0 to enable

export CUDA_VISIBLE_DEVICES="0"

7. For Colbert, when want to use GPU for build index / retrieval, set total_visible_gpus in ColBERT/colbert/infra/config/settings, the total_visible_gpus variable

cmake -DCMAKE_BUILD_TYPE=Debug ..

8. if you find the error `AttributeError: module 'faiss' has no attribute 'StandardGpuResources'`, or find the error about the matrix multiplication for calling the openblas, then you need to uninstall faiss-gpu in pip version and install the faiss-gpu using conda command, `pip install faiss-gpu-cu12`
