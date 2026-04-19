import numpy as np
import os
from typing import Callable
import sys
import time
import json
import pandas as pd

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import evaluation_pipeline
from script.data import util
from script.evaluation.technique import vq_sq_pack

def resolve_performance_file(performance_single_result_fname: str):
    with open(performance_single_result_fname) as f:
        data = {}
        for line in f:
            key, value = line.strip().split('\t')
            try:
                data[key] = float(value) if '.' in value else int(value)
            except ValueError:
                data[key] = value
    return data


def approximate_solution_retrieval(username: str, dataset: str,
                                   method_name: str, build_index_config: dict,
                                   retrieval_config: dict, topk: int,
                                   retrieval_cpp_func: Callable):
    nprobe = retrieval_config['nprobe']
    probe_topk = retrieval_config['probe_topk']
    print(f"retrieval: nprobe {nprobe}, probe_topk {probe_topk}")

    n_bit = build_index_config['n_bit']
    n_centroid = build_index_config['n_centroid']
    build_index_suffix = f'n_centroid_{n_centroid}-n_bit_{n_bit}'

    is_profile = False if retrieval_cpp_func.__name__ == 'retrieval_no_profile_func' else True
    command = retrieval_cpp_func(username=username, dataset=dataset,
                                 method_name=method_name, topk=topk,
                                 n_centroid=n_centroid, n_bit=n_bit,
                                 nprobe=nprobe, probe_topk=probe_topk,
                                 is_profile=is_profile)
    print(command)

    os.system(command)
    # os.system(f"compute-sanitizer --tool memcheck /home/{username}/multi-vector-retrieval/build/{method_name} "
    #           f"-username {username} -dataset {dataset} -method-name {method_name} -build-index-suffix {build_index_suffix} "
    #           f"-topk {topk} -nprobe {nprobe} -probe_topk {probe_topk}")
    retrieval_suffix = f'nprobe_{nprobe}-probe_topk_{probe_topk}'
    retrieval_suffix = f'{retrieval_suffix}-profile' if is_profile else retrieval_suffix
    performance_fname = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer/' \
                        f'{dataset}-{method_name}-performance-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'
    performance_df = pd.read_csv(performance_fname, delimiter='\t')

    performance_single_result_fname = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer/' \
                                      f'{dataset}-{method_name}-performance-top{topk}-{build_index_suffix}-{retrieval_suffix}.txt'
    single_result_m = resolve_performance_file(performance_single_result_fname=performance_single_result_fname)

    retrieval_time_batch_query_ms = single_result_m['batch_query_time(s)'] / single_result_m['n_query'] * 1e3
    search_time_m = {
        "retrieval_time_batch_query(ms)": '{:.3f}'.format(retrieval_time_batch_query_ms),

        "retrieval_time_single_query_p5(ms)": '{:.3f}'.format(np.percentile(performance_df['search_time'], 5) * 1e3),
        "retrieval_time_single_query_p50(ms)": '{:.3f}'.format(np.percentile(performance_df['search_time'], 50) * 1e3),
        "retrieval_time_single_query_p95(ms)": '{:.3f}'.format(np.percentile(performance_df['search_time'], 95) * 1e3),
        "retrieval_time_single_query_average(ms)": '{:.3f}'.format(
            1.0 * np.average(performance_df['search_time']) * 1e3),
    }
    retrieval_time_ms_l = np.around(performance_df['search_time'] * 1e3, 3)

    return retrieval_suffix, search_time_m, retrieval_time_ms_l


def approximate_solution_build_index(username: str, dataset: str,
                                     module: object,
                                     method_name: str,
                                     build_index_config: dict, build_index_suffix: str):
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/{method_name}-{build_index_suffix}'
    if os.path.exists(index_path):
        if not os.path.exists(os.path.join(index_path, 'ivf_group.index')):
            embedding_dir = os.path.expanduser(f'~/Dataset/multi-vector-retrieval/Embedding/{dataset}')
            item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
            n_item = len(item_n_vec_l)
            n_vecs = np.sum(item_n_vec_l)
            n_centroid = build_index_config['n_centroid_f'](n_vecs)

            index_path = os.path.expanduser(f'~/Dataset/multi-vector-retrieval/Index/{dataset}/{method_name}-{build_index_suffix}')
            vq_code_l = np.load(os.path.join(index_path, 'vq_code_l.npy')).astype(np.int32)
            ivf_index = module.IVFIndex(vq_code_l=vq_code_l, item_n_vec_l=item_n_vec_l,
                                        n_vec=n_vecs, n_item=n_item, n_centroid=n_centroid)
            ivf_index.build()
            ivf_index.saveGroupIndex(os.path.join(index_path, f'ivf_group.index'))
            print("save ivf_group.index")

        print("exist index, skip building")
        return
    print(f"start insert item")
    start_time = time.time()

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
    n_item = len(item_n_vec_l)
    n_vecs = np.sum(item_n_vec_l)
    n_centroid = build_index_config['n_centroid_f'](n_vecs)
    n_bit = build_index_config['n_bit']

    centroid_l, vq_code_l, weight_l, residual_code_l = vq_sq_pack.vq_sq_ivf(username=username, dataset=dataset,
                                                                            module=module,
                                                                            n_centroid=n_centroid, n_bit=n_bit)
    print("weight_l", weight_l)

    constructor_build_index = {'centroid_l': centroid_l, 'vq_code_l': vq_code_l,
                               'weight_l': weight_l, 'residual_code_l': residual_code_l}
    print(f"n_centroid {n_centroid}, total_n_vec {len(vq_code_l)}")

    ivf_index = module.IVFIndex(vq_code_l=vq_code_l, item_n_vec_l=item_n_vec_l,
                                n_vec=n_vecs, n_item=n_item, n_centroid=n_centroid)
    ivf_index.build()

    end_time = time.time()
    build_index_time_sec = end_time - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")

    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/{method_name}-{build_index_suffix}'
    os.makedirs(index_path, exist_ok=True)
    np.save(os.path.join(index_path, 'item_n_vec_l.npy'), item_n_vec_l.astype(np.uint32))
    np.save(os.path.join(index_path, 'centroid_l.npy'), centroid_l)
    np.save(os.path.join(index_path, 'vq_code_l.npy'), vq_code_l.astype(np.int32))
    np.save(os.path.join(index_path, 'weight_l.npy'), weight_l)
    np.save(os.path.join(index_path, 'residual_code_l.npy'), residual_code_l)
    ivf_index.saveIVFIndex(os.path.join(index_path, f'ivf.index'))
    ivf_index.saveGroupIndex(os.path.join(index_path, f'ivf_group.index'))

    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    build_index_performance_filename = os.path.join(result_performance_path,
                                                    f'{dataset}-build_index-{method_name}-{build_index_suffix}.json')

    with open(build_index_performance_filename, 'w') as f:
        json.dump({'build_index_time (s)': build_index_time_sec}, f)


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for nprobe in grid_search_para['nprobe']:
        for probe_topk in grid_search_para['probe_topk']:
            parameter_l.append(
                {"nprobe": nprobe, "probe_topk": probe_topk})
    return parameter_l


def retrieval_compute_sanitizer_func(username: str, dataset: str,
                                     method_name: str, topk: int,
                                     n_centroid: int, n_bit: int,
                                     nprobe: int, probe_topk: int,
                                     is_profile: bool):
    is_profile_str = 'true' if is_profile else 'false'
    command = f"compute-sanitizer --tool memcheck /home/{username}/multi-vector-retrieval/build/{method_name} " \
              f"-username {username} -dataset {dataset} -method-name {method_name} " \
              f"-num-centroid {n_centroid} -num-bit {n_bit} " \
              f"-topk {topk} -nprobe {nprobe} -probe_topk {probe_topk} " \
              f"-is_profile {is_profile_str}"
    return command


def retrieval_nsight_system_func(username: str, dataset: str,
                                 method_name: str, topk: int,
                                 n_centroid: int, n_bit: int,
                                 nprobe: int, probe_topk: int,
                                 is_profile: bool):
    is_profile_str = 'true' if is_profile else 'false'
    command = f"sudo nsys profile --force-overwrite true --gpu-metrics-devices=all --gpu-metrics-set=ga10x-gfxt " \
              f"--gpuctxsw=true --stats=true -o {method_name}-{dataset}-nsys-report " \
              f"/home/{username}/multi-vector-retrieval/build/{method_name} " \
              f"-username {username} -dataset {dataset} -method-name {method_name} " \
              f"-num-centroid {n_centroid} -num-bit {n_bit} " \
              f"-topk {topk} -nprobe {nprobe} -probe_topk {probe_topk} " \
              f"-is_profile {is_profile_str}"
    return command


def retrieval_nsight_compute_func(username: str, dataset: str,
                                  method_name: str, topk: int,
                                  n_centroid: int, n_bit: int,
                                  nprobe: int, probe_topk: int,
                                  is_profile: bool):
    is_profile_str = 'true' if is_profile else 'false'
    command = f"sudo /home/{username}/software/anaconda3/envs/billion_MVR/bin/ncu --nvtx " \
              f"--nvtx-include \"filter-compute_score/\" -o {method_name}-{dataset}-ncu-report " \
              f"/home/{username}/multi-vector-retrieval/build/{method_name} " \
              f"-username {username} -dataset {dataset} -method-name {method_name} " \
              f"-num-centroid {n_centroid} -num-bit {n_bit} " \
              f"-topk {topk} -nprobe {nprobe} -probe_topk {probe_topk} " \
              f"-is_profile {is_profile_str}"
    return command


def retrieval_no_profile_func(username: str, dataset: str,
                              method_name: str, topk: int,
                              n_centroid: int, n_bit: int,
                              nprobe: int, probe_topk: int,
                              is_profile: bool):
    is_profile_str = 'true' if is_profile else 'false'
    command = f"/home/{username}/multi-vector-retrieval/build/{method_name} " \
              f"-username {username} -dataset {dataset} -method-name {method_name} " \
              f"-num-centroid {n_centroid} -num-bit {n_bit} " \
              f"-topk {topk} -nprobe {nprobe} -probe_topk {probe_topk} " \
              f"-is_profile {is_profile_str}"
    return command


if __name__ == '__main__':
    # default value {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x))))},
    # TODO, preallocate the resources in cublas and rmm
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            # 'dataset_l': ['lotte', 'msmacro', 'hotpotqa', 'dpr-nq'],
            # 'topk_l': [10, 100],
            'dataset_l': ['msmacro'],
            'topk_l': [100],
            'is_debug': False,
            'retrieval_cpp_func': retrieval_no_profile_func,
            'build_index_parameter_l': [
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(8 * np.sqrt(x)))), 'n_bit': 2},
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 2},
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(32 * np.sqrt(x)))), 'n_bit': 2},
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(64 * np.sqrt(x)))), 'n_bit': 2},
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 8},
            ],
            'retrieval_parameter_l': [
                # {'nprobe': 4, 'probe_topk': 200},
                # {'nprobe': 12, 'probe_topk': 1000},
                {'nprobe': 16, 'probe_topk': 1400},
            ],
            'grid_search': False,
            'grid_search_para': {
                10: {
                    'nprobe': [1, 2, 4, 8, 16, 32],
                    'probe_topk': [20, 40, 60, 80, 100, 200, 300, 400, 600],
                },
                100: {
                    # 'nprobe': [1, 2, 4, 8, 16, 32, 64],
                    'nprobe': [1, 2, 4, 8, 16, 32],
                    'probe_topk': [100, 200, 500, 1000, 1200, 1400, 1600, 1800, 2000],
                },
                1000: {
                    'nprobe': [1, 2, 4, 8, 16, 32, 64],
                    'probe_topk': [1000, 2000, 3000, 4000],
                }
            }
        },
        'local': {
            'username': 'bianzheng',
            # 'dataset_l': ['fake-normal', 'lotte-500-gnd'],
            'dataset_l': ['lotte-500-gnd'],
            # 'dataset_l': ['fake-normal'],
            # 'topk_l': [10, 50],
            'topk_l': [10],
            'is_debug': True,
            'retrieval_cpp_func': retrieval_no_profile_func,
            'build_index_parameter_l': [
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 2}],
            'retrieval_parameter_l': [
                {'nprobe': 20, 'probe_topk': 20},
                # {'nprobe': 64, 'probe_topk': 120},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'nprobe': [32, 64],
                    'probe_topk': [20, 50],
                },
                50: {
                    'nprobe': [10, 9, 8, 7],
                    'probe_topk': [100],
                },
            }
        }
    }
    host_name = 'local'
    config = config_l[host_name]

    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']
    is_debug = config['is_debug']
    retrieval_cpp_func = config['retrieval_cpp_func']
    build_index_parameter_l = config['build_index_parameter_l']

    method_name = 'IGPGPUPP'
    build_index_module_name = 'GPGPUBuildIndexPackGPU'
    move_path = 'evaluation'

    util.compile_file(username=username, module_name=build_index_module_name, is_debug=is_debug, move_path=move_path)
    for dataset in dataset_l:
        for build_index_config in build_index_parameter_l:
            embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
            vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
            if vec_dim % 4 != 0:
                raise Exception(
                    "the vector dimension must be divisible by 4, i.e., each vector size should be the multiple of 128 bit")
            n_item = np.load(os.path.join(embedding_dir, f'doclens.npy')).shape[0]
            item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
            n_vecs = np.sum(item_n_vec_l)

            n_centroid = build_index_config['n_centroid_f'](n_vecs)
            n_bit = build_index_config['n_bit']
            build_index_config['n_centroid'] = n_centroid

            build_index_suffix = f'n_centroid_{n_centroid}-n_bit_{n_bit}'

            module = evaluation_pipeline.approximate_solution_compile_load(
                username=username, dataset=dataset,
                module_name=build_index_module_name, compile_file=False,
                is_debug=is_debug, move_path=move_path)

            approximate_solution_build_index(
                username=username, dataset=dataset,
                module=module, method_name=method_name,
                build_index_config=build_index_config, build_index_suffix=build_index_suffix)

            for topk in topk_l:
                grid_search = config['grid_search']
                if grid_search:
                    retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
                else:
                    retrieval_parameter_l = config['retrieval_parameter_l']

                evaluation_pipeline.cpp_retrieval_outter(
                    username=username, dataset=dataset,
                    method_name=method_name,
                    build_index_suffix=build_index_suffix, build_index_config=build_index_config,
                    topk=topk,
                    retrieval_parameter_l=retrieval_parameter_l,
                    retrieval_python_func=approximate_solution_retrieval, retrieval_cpp_func=retrieval_cpp_func
                )
