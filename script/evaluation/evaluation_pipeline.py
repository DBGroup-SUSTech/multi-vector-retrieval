import copy

import numpy as np
import os
import time
import importlib
from typing import Dict, Callable
import sys
import tqdm
import json
import pandas as pd

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric
from script.data import util


def save_retrieval_result(est_dist_l: np.ndarray, est_id_l: np.ndarray,
                          queryID_l: list,
                          retrieval_result_filename: str):
    assert len(est_dist_l) == len(est_id_l)
    n_query = len(est_dist_l)
    with open(retrieval_result_filename, "w") as f:
        for local_queryID, single_dist_l, single_id_l in zip(np.arange(n_query), est_dist_l, est_id_l):
            queryID = queryID_l[local_queryID]
            for rank, (dist, passageID) in enumerate(zip(single_dist_l, single_id_l)):
                f.write(f"{queryID}\t{passageID}\t{rank + 1}\t{dist}\n")


def approximate_solution_compile_load(username: str, dataset: str, compile_file: bool,
                                      module_name: str, is_debug: bool, move_path: str):
    # compile and import module
    if compile_file:
        util.compile_file(username=username, module_name=module_name, is_debug=is_debug, move_path=move_path)

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'

    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')
    vec_dim = 128
    n_item = np.load(os.path.join(embedding_dir, f'doclens.npy')).shape[0]
    print(f"dataset {dataset}, n_item {n_item}, vec_dim {vec_dim}")
    module = importlib.import_module(module_name)
    return module


def approximate_solution_build_index(username: str, dataset: str,
                                     constructor_insert_item: dict, module: object,
                                     module_name: str, build_index_suffix: str,
                                     save_index: bool, constructor_build_index: dict = {}):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')
    index = module.DocRetrieval(**constructor_insert_item)
    print(f"start insert item")
    start_time = time.time()
    n_chunk = util.get_n_chunk(base_embedding_dir)
    accu_itemID = 0
    for chunkID in tqdm.trange(n_chunk):
        itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        n_item_chunk = itemlen_l_chunk.shape[0]

        item_l_chunk = []
        itemID_l_chunk = []
        for itemID_chunk in range(n_item_chunk):
            item = util.item_vecs_in_chunk(vecs_l=item_vecs_l_chunk, itemlen_l=itemlen_l_chunk, itemID=itemID_chunk)
            item_l_chunk.append(item)
            itemID = accu_itemID + itemID_chunk
            itemID_l_chunk.append(itemID)
        index.add_item_batch(item_l=item_l_chunk, itemID_l=itemID_l_chunk)
        accu_itemID += n_item_chunk
    index.build_index(**constructor_build_index)
    end_time = time.time()
    build_index_time_sec = end_time - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")

    if save_index:
        index_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/'
        os.makedirs(os.path.join(index_dir, module_name), exist_ok=True)
        index_filename = os.path.join(index_dir, module_name,
                                      f'{dataset}-{module_name}-{build_index_suffix}.index')
        index.save(index_filename)

    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    build_index_performance_filename = os.path.join(result_performance_path,
                                                    f'{dataset}-build_index-{module_name}-{build_index_suffix}.json')
    with open(build_index_performance_filename, 'w') as f:
        json.dump({'build_index_time (s)': build_index_time_sec}, f)

    return index


def load_query(username: str, dataset: str):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    query_l = np.load(os.path.join(embedding_dir, 'query_embedding.npy'))

    rawdata_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}'

    query_text_filename = os.path.join(rawdata_path, f'document/queries.dev.tsv')
    if os.path.exists(query_text_filename):
        queryID_l = []
        with open(query_text_filename, 'r', encoding="utf-8") as f:
            for line in f:
                query_text_l = line.split('\t')
                queryID_l.append(int(query_text_l[0]))
            assert len(queryID_l) == len(query_l)
    else:
        queryID_l = np.arange(len(query_l))
    return query_l, queryID_l


def approximate_solution_retrieval_outter(username: str, dataset: str,
                                          index: object, module_name: str,
                                          build_index_suffix: str,
                                          constructor_insert_item: dict,
                                          topk: int,
                                          retrieval_parameter_l: list,
                                          retrieval_f: Callable
                                          ):
    query_l, queryID_l = load_query(username=username, dataset=dataset)

    mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                topk=topk)

    final_result_l = []
    for retrieval_config in retrieval_parameter_l:
        est_dist_l, est_id_l, retrieval_suffix, search_time_m, time_ms_l = retrieval_f(
            index=index, retrieval_config=retrieval_config,
            query_l=query_l, topk=topk)

        result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer/'
        method_ans_name = f'{dataset}-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'
        retrieval_result_filename = os.path.join(result_answer_path, method_ans_name)
        save_retrieval_result(est_dist_l=est_dist_l, est_id_l=est_id_l,
                              queryID_l=queryID_l, retrieval_result_filename=retrieval_result_filename)

        recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
            username=username, dataset=dataset, topk=topk,
            method_name=module_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
            mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)

        build_index_config = copy.deepcopy(constructor_insert_item)
        if 'item_n_vec_l' in build_index_config:
            del build_index_config['item_n_vec_l']
        if 'item_n_vecs_l' in build_index_config:
            del build_index_config['item_n_vecs_l']
        retrieval_info_m = {'n_query': len(queryID_l), 'topk': topk,
                            'build_index': build_index_config, 'retrieval': retrieval_config,
                            'search_time': search_time_m, 'search_accuracy': search_accuracy_m}
        method_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
        result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
        performance_filename = os.path.join(result_performance_path, method_performance_name)
        with open(performance_filename, "w") as f:
            json.dump(retrieval_info_m, f)

        df = pd.DataFrame({'time(ms)': time_ms_l, 'recall': recall_l})
        df.index.name = 'local_queryID'
        if mrr_l:
            df['mrr'] = mrr_l
        if success_l:
            df['success'] = success_l
        single_query_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.csv'
        result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/single_query_performance'
        single_query_performance_filename = os.path.join(result_performance_path, single_query_performance_name)
        df.to_csv(single_query_performance_filename, index=True)

        print("#############final result###############")
        print("filename", method_performance_name)
        print("search time", retrieval_info_m['search_time'])
        print("search accuracy", retrieval_info_m['search_accuracy'])
        print("########################################")

        final_result_l.append({'filename': method_performance_name, 'search_time': retrieval_info_m['search_time'],
                               'search_accuracy': retrieval_info_m['search_accuracy']})

    for final_result in final_result_l:
        print("#############final result###############")
        print("filename", final_result['filename'])
        print("search time", final_result['search_time'])
        print("search accuracy", final_result['search_accuracy'])
        print("########################################")


def approximate_solution_retrieval(index: object, retrieval_config: dict, query_l: np.ndarray, topk: int):
    efSearch = retrieval_config['efSearch']
    index.set_efSearch(efSearch)
    print(f"retrieval: efSearch {efSearch}")

    est_dist_l, est_id_l, retrieval_time_l, n_dist_compute_l = index.query(query_l=query_l, topk=topk)
    retrieval_suffix = f'efSearch_{retrieval_config["efSearch"]}'

    search_time_m = {
        'total_query_time_ms': '{:.3f}'.format(sum(retrieval_time_l) * 1e3),
        "retrieval_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5) * 1e3),
        "retrieval_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50) * 1e3),
        "retrieval_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95) * 1e3),
        "retrieval_time_average(ms)": '{:.3f}'.format(1.0 * sum(retrieval_time_l) / len(retrieval_time_l) * 1e3),

        "n_dist_compute_p5": '{:.3f}'.format(np.percentile(n_dist_compute_l, 5)),
        "n_dist_compute_p50": '{:.3f}'.format(np.percentile(n_dist_compute_l, 50)),
        "n_dist_compute_p95": '{:.3f}'.format(np.percentile(n_dist_compute_l, 95)),
        "n_dist_compute_average": '{:.3f}'.format(np.average(n_dist_compute_l)),
    }
    return est_dist_l, est_id_l, retrieval_suffix, search_time_m


def approximate_solution(username: str, module_name: str, is_debug: bool, move_path: str,
                         dataset: str, topk_l: list,
                         load_index: bool,
                         constructor_insert_item: Dict,
                         constructor_load_index: Dict,
                         build_index_suffix: str,
                         retrieval_parameter_l: list):
    module = approximate_solution_compile_load(
        username=username, dataset=dataset,
        module_name=module_name, compile_file=False,
        is_debug=is_debug, move_path=move_path)

    index_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/'
    index_filename = os.path.join(index_dir, module_name,
                                  f'{dataset}-{module_name}-{build_index_suffix}.index')
    if os.path.exists(index_filename) and constructor_load_index and load_index:
        print("start load index")
        index = module.DocRetrieval(**constructor_load_index)
    else:
        index = approximate_solution_build_index(
            username=username, dataset=dataset,
            constructor_insert_item=constructor_insert_item, module=module,
            module_name=module_name, build_index_suffix=build_index_suffix,
            save_index=True)

    retrieval_suffix_m = {}
    for topk in topk_l:
        answer_suffix_l = approximate_solution_retrieval_outter(
            username=username, dataset=dataset,
            index=index, module_name=module_name,
            build_index_suffix=build_index_suffix,
            constructor_insert_item=constructor_insert_item,
            topk=topk,
            retrieval_parameter_l=retrieval_parameter_l,
            retrieval_f=approximate_solution_retrieval
        )
        retrieval_suffix_m[topk] = answer_suffix_l

    performance_metric.count_accuracy_by_baseline(username=username, dataset=dataset, topk_l=topk_l,
                                                  method_name=module_name,
                                                  build_index_suffix=build_index_suffix,
                                                  retrieval_suffix_m=retrieval_suffix_m)
