import numpy as np
import os
from typing import Dict
import sys
import time
import json

os.environ["OPENBLAS_NUM_THREADS"] = "1"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric, evaluation_pipeline
from script.data import util
from script.evaluation.technique import vq_sq_pack


def approximate_solution_retrieval(index: object, retrieval_config: dict, query_l: np.ndarray, topk: int):
    nprobe = retrieval_config['nprobe']
    probe_topk = retrieval_config['probe_topk']
    print(f"retrieval: nprobe {nprobe}, probe_topk {probe_topk}")

    (est_dist_l, est_id_l, retrieval_time_l,
     filter_time_l, decode_time_l, refine_time_l,
     n_sorted_ele_l, n_seen_item_l, n_refine_item_l,
     incremental_graph_n_compute_l,
     n_vq_score_refine_l, n_vq_score_linear_scan_l) = index.search(
        query_l=query_l, topk=topk,
        nprobe=nprobe, probe_topk=probe_topk)

    search_time_m = {
        "retrieval_time_single_query_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5) * 1e3),
        "retrieval_time_single_query_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50) * 1e3),
        "retrieval_time_single_query_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95) * 1e3),
        "retrieval_time_single_query_average(ms)": '{:.3f}'.format(1.0 * np.average(retrieval_time_l) * 1e3),

        "filter_time_average(ms)": '{:.3f}'.format(1.0 * np.average(filter_time_l) * 1e3),
        "decode_time_average(ms)": '{:.3f}'.format(1.0 * np.average(decode_time_l) * 1e3),
        "refine_time_average(ms)": '{:.3f}'.format(1.0 * np.average(refine_time_l) * 1e3),

        "n_sorted_ele_average": '{:.3f}'.format(1.0 * np.average(n_sorted_ele_l)),
        "n_seen_item_average": '{:.3f}'.format(1.0 * np.average(n_seen_item_l)),
        "n_refine_item_average": '{:.3f}'.format(1.0 * np.average(n_refine_item_l)),
        "incremental_graph_n_compute_l_average": '{:.3f}'.format(1.0 * np.average(incremental_graph_n_compute_l)),
        "n_vq_score_refine_average": '{:.3f}'.format(1.0 * np.average(n_vq_score_refine_l)),
        "n_vq_score_linear_scan_average": '{:.3f}'.format(1.0 * np.average(n_vq_score_linear_scan_l)),
    }
    retrieval_suffix = f'nprobe_{nprobe}-probe_topk_{probe_topk}'
    retrieval_time_ms_l = np.around(retrieval_time_l * 1e3, 3)

    return est_dist_l, est_id_l, retrieval_suffix, search_time_m, retrieval_time_ms_l


def approximate_solution_build_index(username: str, dataset: str,
                                     constructor_insert_item: dict, module: object,
                                     module_name: str,
                                     build_index_config: dict, build_index_suffix: str,
                                     save_index: bool):
    index = module.DocRetrieval(**constructor_insert_item)
    print(f"start insert item")
    start_time = time.time()

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
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
    index.load_quantization_index(**constructor_build_index)
    index.build_graph_index()
    end_time = time.time()
    build_index_time_sec = end_time - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")

    if save_index:
        index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/{module_name}-{build_index_suffix}'
        if os.path.exists(index_path):
            print("index path exists, not save the index")
        else:
            print(f"save the index into {index_path}")
            os.makedirs(index_path, exist_ok=False)
            np.save(os.path.join(index_path, 'centroid_l.npy'), centroid_l)
            np.save(os.path.join(index_path, 'vq_code_l.npy'), vq_code_l)
            np.save(os.path.join(index_path, 'weight_l.npy'), weight_l)
            np.save(os.path.join(index_path, 'residual_code_l.npy'), residual_code_l)
            with open(os.path.join(index_path, 'constructor_insert_item.json'), 'w') as f:
                json.dump(constructor_insert_item, f)
            index.save_graph_index(os.path.join(index_path, 'index.hnsw'))

    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    build_index_performance_filename = os.path.join(result_performance_path,
                                                    f'{dataset}-build_index-{module_name}-{build_index_suffix}.json')
    with open(build_index_performance_filename, 'w') as f:
        json.dump({'build_index_time (s)': build_index_time_sec}, f)

    return index


def load_index(index_path: str, module: object):
    centroid_l = np.load(os.path.join(index_path, 'centroid_l.npy'))
    vq_code_l = np.load(os.path.join(index_path, 'vq_code_l.npy'))
    weight_l = np.load(os.path.join(index_path, 'weight_l.npy'))
    residual_code_l = np.load(os.path.join(index_path, 'residual_code_l.npy'))
    with open(os.path.join(index_path, 'constructor_insert_item.json'), 'r') as f:
        constructor_insert_item = json.load(f)

    index = module.DocRetrieval(**constructor_insert_item)
    constructor_build_index = {'centroid_l': centroid_l, 'vq_code_l': vq_code_l,
                               'weight_l': weight_l, 'residual_code_l': residual_code_l}
    n_centroid = centroid_l.shape[0]
    print(f"n_centroid {n_centroid}, total_n_vec {len(vq_code_l)}")
    index.load_quantization_index(**constructor_build_index)
    index.load_graph_index(os.path.join(index_path, 'index.hnsw'))
    return index


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for nprobe in grid_search_para['nprobe']:
        for probe_topk in grid_search_para['probe_topk']:
            parameter_l.append(
                {"nprobe": nprobe, "probe_topk": probe_topk})
    return parameter_l


if __name__ == '__main__':
    # default value {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x))))},
    config_l = {
        'dbg': {
            'username': 'username1',
            # 'dataset_l': ['quora', 'lotte', 'msmacro', 'hotpotqa'],
            'dataset_l': ['msmacro', 'lotte', 'hotpotqa'],
            'topk_l': [10, 100],
            'is_debug': False,
            'build_index_parameter_l': [
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(8 * np.sqrt(x)))), 'n_bit': 2},
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 2},
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(32 * np.sqrt(x)))), 'n_bit': 2},
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(64 * np.sqrt(x)))), 'n_bit': 2},
                # {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 8},
            ],
            'retrieval_parameter_l': [
                {'nprobe': 4, 'probe_topk': 100},
                {'nprobe': 4, 'probe_topk': 200},
                {'nprobe': 4, 'probe_topk': 400},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'nprobe': [1, 2, 4, 8, 16, 32],
                    'probe_topk': [100, 200, 300, 400, 600],
                    # 'nprobe': [1, 2, 4, 8, 16, 32, 64],
                    # 'probe_topk': [20, 50, 100, 200, 300, 400, 600],
                },
                100: {
                    # 'nprobe': [1, 2, 4, 8, 16, 32, 64],
                    'nprobe': [1, 2, 4, 8, 16, 32],
                    'probe_topk': [100, 200, 500, 700, 1000],
                },
                1000: {
                    'nprobe': [1, 2, 4, 8, 16, 32, 64],
                    'probe_topk': [1000, 2000, 3000, 4000],
                }
            }
        },
        'local': {
            'username': 'username2',
            'dataset_l': ['lotte-500-gnd'],
            # 'topk_l': [10, 50],
            'topk_l': [10],
            'is_debug': True,
            'build_index_parameter_l': [
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 2}],
            'retrieval_parameter_l': [
                # {'nprobe': 1, 'probe_topk': 20},
                # {'nprobe': 2, 'probe_topk': 20},
                # {'nprobe': 4, 'probe_topk': 20},
                {'nprobe': 8, 'probe_topk': 20},
                {'nprobe': 16, 'probe_topk': 20},
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
    # only use the nprobe, no query coreset, no maintain the lower bound

    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']
    is_debug = config['is_debug']
    build_index_parameter_l = config['build_index_parameter_l']

    module_name = 'IGP'
    save_index = True
    move_path = 'evaluation'

    util.compile_file(username=username, module_name=module_name, is_debug=is_debug, move_path=move_path)
    for dataset in dataset_l:
        for build_index_config in build_index_parameter_l:
            embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
            vec_dim = 128
            n_item = np.load(os.path.join(embedding_dir, f'doclens.npy')).shape[0]
            item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
            n_vecs = np.sum(item_n_vec_l)

            n_centroid = build_index_config['n_centroid_f'](n_vecs)
            n_bit = build_index_config['n_bit']

            constructor_insert_item = {'item_n_vec_l': item_n_vec_l.tolist(),
                                       'n_item': n_item, 'vec_dim': vec_dim,
                                       'n_centroid': n_centroid, 'n_bit': n_bit}

            build_index_suffix = f'n_centroid_{n_centroid}-n_bit_{n_bit}'

            module = evaluation_pipeline.approximate_solution_compile_load(
                username=username, dataset=dataset,
                module_name=module_name, compile_file=False,
                is_debug=is_debug, move_path=move_path)

            index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/{module_name}-{build_index_suffix}'
            if os.path.exists(index_path):
                index = load_index(index_path=index_path, module=module)
            else:
                index = approximate_solution_build_index(
                    username=username, dataset=dataset,
                    constructor_insert_item=constructor_insert_item,
                    module=module, module_name=module_name,
                    build_index_config=build_index_config, build_index_suffix=build_index_suffix,
                    save_index=save_index)

            for topk in topk_l:
                grid_search = config['grid_search']
                if grid_search:
                    retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
                else:
                    retrieval_parameter_l = config['retrieval_parameter_l']

                evaluation_pipeline.approximate_solution_retrieval_outter(
                    username=username, dataset=dataset,
                    index=index, module_name=module_name,
                    build_index_suffix=build_index_suffix,
                    constructor_insert_item=constructor_insert_item,
                    topk=topk,
                    retrieval_parameter_l=retrieval_parameter_l,
                    retrieval_f=approximate_solution_retrieval
                )
