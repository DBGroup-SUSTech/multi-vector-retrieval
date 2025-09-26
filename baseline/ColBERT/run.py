import json

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# export CUDA_VISIBLE_DEVICES=""
# in bash, setting CUDA_VISIBLE_DEVICES=1 to enable
# export CUDA_VISIBLE_DEVICES="0"
from os import listdir
from os.path import isfile, join
import re
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher, IndexerGenerate
from colbert.data import Queries
import numpy as np
import torch
import time
import sys
import copy
import pandas as pd

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def get_n_chunk(base_dir: str):
    filename_l = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)


def build_index_official(username: str, dataset: str):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    delete_file_if_exist(embedding_path)
    os.makedirs(base_embedding_path, exist_ok=False)
    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'experiments'))):
        config = ColBERTConfig(
            nbits=2,
            root=colbert_project_path,
        )
        print(config)
        indexer = Indexer(checkpoint=pretrain_index_path, config=config)
        build_index_time, encode_passage_time = indexer.index(name=dataset,
                                                              collection=os.path.join(document_data_path,
                                                                                      'collection.tsv'),
                                                              embedding_filename=base_embedding_path,
                                                              overwrite=True)
    index_origin_path = os.path.join(colbert_project_path, f'experiments/{dataset}/indexes/{dataset}')
    delete_file_if_exist(index_path)
    os.makedirs(index_path, exist_ok=False)
    index_new_path = os.path.join(index_path, 'plaid')
    os.system(f'mv {index_origin_path} {index_new_path}')
    print("finish indexing, start searching")

    build_index_json = {'build_index_time (s)': build_index_time, 'encode_passage_time (s)': encode_passage_time}
    with open(os.path.join(result_performance_path, f'{dataset}-build_index-plaid-.json'), 'w') as f:
        json.dump(build_index_json, f)

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        total_encode_time_ms, n_encode_query = searcher.save_query_embedding(queries, query_embedding_filename)
        # topk = 100

        # ranking = searcher.search_all_embedding(query_embedding_filename, k=topk)
        # ranking.save(f"{dataset}_self_search_method_top{topk}.tsv")

        # ranking = searcher.search_all(queries, k=topk)
        # ranking.save(f"{dataset}_official_search_method_top{topk}.tsv")

    encode_info = {'total_encode_time_ms': total_encode_time_ms, 'n_encode_query': n_encode_query,
                   'average_encode_time_ms': total_encode_time_ms / n_encode_query}
    with open(os.path.join(result_performance_path, f'{dataset}-encode_query.json'), 'w') as f:
        json.dump(encode_info, f)

    n_chunk = get_n_chunk(base_embedding_path)
    total_doclens = []
    for chunkID in range(n_chunk):
        doclens = np.load(os.path.join(base_embedding_path, f'doclens{chunkID}.npy'))
        total_doclens = np.append(total_doclens, doclens)
    np.save(os.path.join(embedding_path, 'doclens.npy'), total_doclens)


def encode_query_cpu(username: str, dataset: str):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')
    print("finish indexing, start searching")

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        total_encode_time_ms, n_encode_query = searcher.save_query_embedding_cpu(queries, query_embedding_filename)
        # topk = 100

        # ranking = searcher.search_all_embedding(query_embedding_filename, k=topk)
        # ranking.save(f"{dataset}_self_search_method_top{topk}.tsv")

        # ranking = searcher.search_all(queries, k=topk)
        # ranking.save(f"{dataset}_official_search_method_top{topk}.tsv")

    encode_info = {'total_encode_time_ms': total_encode_time_ms, 'n_encode_query': n_encode_query,
                   "cpu": True,
                   'average_encode_time_ms': total_encode_time_ms / n_encode_query}
    with open(os.path.join(result_performance_path, f'{dataset}-encode_query.json'), 'w') as f:
        json.dump(encode_info, f)


def build_index_generate(username: str, dataset: str):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    indexer = IndexerGenerate()
    build_index_time, encode_passage_time = indexer.index(username=username, dataset=dataset)

    build_index_json = {'build_index_time (s)': build_index_time, 'encode_passage_time (s)': encode_passage_time}
    with open(os.path.join(result_performance_path, f'{dataset}-build_index-plaid-.json'), 'w') as f:
        json.dump(build_index_json, f)



def load_training_query(username: str, dataset: str, n_sample_query: int):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'

    n_gpu = torch.cuda.device_count()
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.train.tsv'))
        train_query = searcher.get_query_embedding(queries)
        del searcher, queries
    return train_query


def load_dev_query(username: str, dataset: str, n_sample_query: int):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'

    n_gpu = torch.cuda.device_count()
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        train_query = searcher.get_query_embedding(queries)
        del searcher, queries
    return train_query


def retrieval_official(username: str, dataset: str, topk: int, search_config_l: list):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer'
    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                topk=topk)

    index_new_path = os.path.join(index_path, 'plaid')
    module_name = 'plaid'

    query_emb = np.load(query_embedding_filename)
    n_query = len(query_emb)
    final_result_l = []
    for search_config in search_config_l:
        print(f"plaid topk {topk}, search config {search_config}")
        torch.set_num_threads(search_config['n_thread'])
        with Run().context(
                RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
                **colbert_retrieval_config
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_new_path,
                                config=config)

            qid_l = []
            with open(query_text_filename, 'r') as f:
                for line in f:
                    query_text_l = line.split('\t')
                    qid_l.append(int(query_text_l[0]))
                assert len(qid_l) == n_query

            ranking, retrieval_time_l, time_ivf_l, time_filter_l, time_refine_l, n_refine_ivf_l, n_refine_filter_l, n_vec_score_refine_l = searcher.search_all_embedding_by_vector(
                query_emb=query_emb,
                query_embd_filename=query_embedding_filename,
                qid_l=qid_l,
                k=topk)

        time_ms_l = np.around(retrieval_time_l, 3)

        build_index_suffix = ''
        para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
        retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                           f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
        ranking.save_absolute_path(
            os.path.join(result_answer_path,
                         f'{dataset}-plaid-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'))

        search_time_m = {
            'total_query_time_ms': '{:.3f}'.format(sum(retrieval_time_l)),
            "retrieval_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
            "retrieval_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
            "retrieval_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
            'average_query_time_ms': '{:.3f}'.format(1.0 * sum(retrieval_time_l) / n_query),
            'average_ivf_time_ms': '{:.3f}'.format(1.0 * sum(time_ivf_l) / n_query),
            'average_filter_time_ms': '{:.3f}'.format(1.0 * sum(time_filter_l) / n_query),
            'average_refine_time_ms': '{:.3f}'.format(1.0 * sum(time_refine_l) / n_query),
            'average_n_refine_ivf': '{:.3f}'.format(np.average(n_refine_ivf_l)),
            'average_n_refine_filter': '{:.3f}'.format(np.average(n_refine_filter_l)),
            'average_n_vec_score_refine': '{:.3f}'.format(np.average(n_vec_score_refine_l)),
        }
        retrieval_config = {
            'ndocs': searcher.config.ndocs,
            'ncells': searcher.config.ncells,
            'centroid_score_threshold': searcher.config.centroid_score_threshold,
            'n_thread': search_config['n_thread']
        }
        recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
            username=username, dataset=dataset, topk=topk,
            method_name=module_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
            mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)
        retrieval_info_m = {
            'n_query': n_query, 'topk': topk, 'build_index': {},
            'retrieval': retrieval_config,
            'search_time': search_time_m, 'search_accuracy': search_accuracy_m
        }

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


def retrieval_end2end_single(username: str, dataset: str, topk: int, search_config_l: list):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
    result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval/end2end/Result/answer'
    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    index_new_path = os.path.join(index_path, 'plaid')
    retrieval_suffix_l = []

    n_query = len(np.load(query_embedding_filename))
    for search_config in search_config_l:
        print(f"plaid topk {topk}, search config {search_config}")
        torch.set_num_threads(search_config['n_thread'])
        with Run().context(
                RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
                **colbert_retrieval_config
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_new_path,
                                config=config)
            queries = Queries(
                path=os.path.join(document_data_path, 'queries.dev.tsv'))
            ranking_l, encode_time_l, retrieval_time_l = searcher.search_all_single(
                queries, k=topk)

            search_result_m = {
                'time': {
                    "average_retrieval_time_ms": '{:.3f}'.format(
                        np.average(encode_time_l) + np.average(retrieval_time_l)),
                    "average_encode_time_ms": '{:.3f}'.format(np.average(encode_time_l)),
                    "search_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
                    "search_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
                    "search_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
                    'average_search_time_ms': '{:.3f}'.format(np.average(retrieval_time_l)),
                }, 'config': {
                    'ndocs': searcher.config.ndocs,
                    'ncells': searcher.config.ncells,
                    'centroid_score_threshold': searcher.config.centroid_score_threshold,
                    'n_thread': search_config['n_thread']
                }
            }
            retrieval_info_m = {'n_query': n_query, 'topk': topk, 'search_result': search_result_m}
            para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
            build_index_suffix = ''
            retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                               f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
            output_path = f'/home/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
            output_filename = os.path.join(output_path,
                                           f'{dataset}-retrieval-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}-time.json')
            with open(output_filename, 'w') as f:
                json.dump(retrieval_info_m, f)

            answer_str_l = []
            for ranking in ranking_l:
                str_l = ranking.tolist()
                for string in str_l:
                    answer_str_l.append(string)
            index_filename = os.path.join(result_answer_path,
                                          f'{dataset}-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv')

            with open(index_filename, 'w') as f:
                for items in answer_str_l:
                    line = '\t'.join(
                        map(lambda x: str(int(x) if type(x) is torch.Tensor or type(x) is bool else x), items)) + '\n'
                    f.write(line)
                print(f"#> Saved ranking of {n_query} queries and {len(answer_str_l)} lines to {f.name}")

            retrieval_suffix_l.append(retrieval_suffix)

    return retrieval_suffix_l


def retrieval_end2end_batch(username: str, dataset: str, topk: int, search_config_l: list):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
    result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval/end2end/Result/answer'
    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    index_new_path = os.path.join(index_path, 'plaid')
    retrieval_suffix_l = []

    n_query = len(np.load(query_embedding_filename))
    for search_config in search_config_l:
        print(f"plaid topk {topk}, search config {search_config}")
        torch.set_num_threads(search_config['n_thread'])
        with Run().context(
                RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
                **colbert_retrieval_config
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_new_path,
                                config=config)
            queries = Queries(
                path=os.path.join(document_data_path, 'queries.dev.tsv'))
            ranking, encode_time, retrieval_time_l = searcher.search_all_batch(
                queries, k=topk)

            search_result_m = {
                'time': {
                    "average_retrieval_time_ms": '{:.3f}'.format(
                        1.0 * encode_time / n_query + np.average(retrieval_time_l)),
                    "average_encode_time_ms": '{:.3f}'.format(1.0 * encode_time / n_query),
                    "search_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
                    "search_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
                    "search_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
                    'average_search_time_ms': '{:.3f}'.format(np.average(retrieval_time_l)),
                }, 'config': {
                    'ndocs': searcher.config.ndocs,
                    'ncells': searcher.config.ncells,
                    'centroid_score_threshold': searcher.config.centroid_score_threshold,
                    'n_thread': search_config['n_thread']
                }
            }
            retrieval_info_m = {'n_query': n_query, 'topk': topk, 'search_result': search_result_m}
            para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
            build_index_suffix = ''
            retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                               f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
            output_path = f'/home/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
            output_filename = os.path.join(output_path,
                                           f'{dataset}-retrieval-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}-time.json')
            with open(output_filename, 'w') as f:
                json.dump(retrieval_info_m, f)

            ranking.save_absolute_path(
                os.path.join(result_answer_path,
                             f'{dataset}-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'))
            retrieval_suffix_l.append(retrieval_suffix)

    return retrieval_suffix_l


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-small'
    build_index_official(username=username, dataset=dataset)
