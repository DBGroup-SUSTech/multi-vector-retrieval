import copy

import numpy as np
import os
import time
import importlib
from typing import Dict, Callable, List
import tqdm
import sys
import re
import string

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
import json
from script.data import util


def read_method_tsv(base_dir: str, dataset: str, method_name: str,
                    topk: int, build_index_suffix: str, retrieval_suffix: str):
    baseline_tsv_name = f'{dataset}-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'
    baseline_filename = os.path.join(base_dir, baseline_tsv_name)
    baseline_id_m = {}
    with open(baseline_filename, 'r', encoding="utf-8") as f:
        for line in f:
            arr = line.split('\t')
            qID = int(arr[0])
            itemID = int(arr[1])
            rank = int(arr[2])
            if qID not in baseline_id_m:
                baseline_id_m[qID] = [[itemID, rank]]
            else:
                baseline_id_m[qID].append([itemID, rank])
    return baseline_id_m


'''
used to count the mrr
'''


def read_mrr_groundtruth_jsonl(dataset: str, username: str):
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    gnd_jsonl_filename = os.path.join(raw_data_path, f'{dataset}/document/queries.gnd.jsonl')
    end2end_gnd_m = {}
    with open(gnd_jsonl_filename, 'r', encoding="utf-8") as f:
        for line in f:
            query_gnd_json = json.loads(line)
            queryID = query_gnd_json['query_id']
            passageID_l = query_gnd_json['passage_id']
            end2end_gnd_m[queryID] = passageID_l
    return end2end_gnd_m


def read_mrr_passageID_l(dataset: str, username: str):
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    collection_filename = os.path.join(raw_data_path, f'{dataset}/document/collection.tsv')
    end2end_passageID_l = []
    with open(collection_filename, 'r', encoding="utf-8") as f:
        for line in f:
            passageID = int(line.split('\t')[0])
            end2end_passageID_l.append(passageID)
    return end2end_passageID_l


def count_mrr(est_id_m: dict, end2end_gnd_m: dict, end2end_passageID_l: list):
    mrr_l = []
    for queryID in est_id_m.keys():
        assert queryID in end2end_gnd_m.keys()
        tmp_mrr = 0
        for local_passageID, rank in est_id_m[queryID]:
            assert local_passageID < len(end2end_passageID_l), \
                f'program output unexpected itemID, local itemID {local_passageID}'
            global_passageID = end2end_passageID_l[local_passageID]
            if global_passageID in end2end_gnd_m[queryID]:
                tmp_mrr = 1 / rank
                break
        mrr_l.append(tmp_mrr)
    assert len(mrr_l) == len(est_id_m.keys()) and len(mrr_l) == len(end2end_gnd_m.keys())
    return mrr_l


'''
used to count end-to-end recall
'''


def count_end2end_recall(est_id_m: dict, end2end_gnd_m: dict, end2end_passageID_l: list):
    end2end_recall_l = []
    for queryID in est_id_m.keys():
        assert queryID in end2end_gnd_m.keys()
        if len(end2end_gnd_m[queryID]) == 0:
            end2end_recall_l.append(1)
            continue
        global_passageID_l = [end2end_passageID_l[local_passageID] for local_passageID, rank in est_id_m[queryID]]
        recall = len(np.intersect1d(global_passageID_l, end2end_gnd_m[queryID])) / len(end2end_gnd_m[queryID])
        end2end_recall_l.append(recall)
    assert len(end2end_recall_l) == len(est_id_m.keys()) and len(end2end_recall_l) == len(end2end_gnd_m.keys())
    return end2end_recall_l


'''
used to count end-to-end recall
'''


def dcg(grades, n=0):
    """
    Discounted Cumulative Gain
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain

    A metric that varies directly with the average judgement of the result set, placing more weight toward the top.

    :param grades: A list of numbers indicating how relevant the corresponding document was at that position in the list
    :param n: A number indicating the maximum number of positions to consider
    :return: A number >= 0.0 indicating how the result set should be judged
    """
    if n == 0:
        n = len(grades)
    n = min(n, len(grades))
    from math import log
    dcg = 0
    for i in range(0, n):
        r = i + 1
        dcg += grades[i] / log((r + 1), 2.0)
    return dcg


def ndcg(grades, rels, n=0):
    """
    Normalized Discounted Cumulative Gain
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    A metric that considers the sort order of the rated documents against an ideal sort order with higher rated docs
    at the top and lower rated docs at the bottom.

    :param grades: A list of numbers indicating how relevant the corresponding document was at that position in the list
    :param rels: A list of numbers containing all of the relevance judgements, for computing the ideal DCG. May be just the non-zero values.
    :param n: A number indicating the maximum number of positions to consider
    :return: A number between 1.0 and 0.0 indicating how close to the ideal ordering the docs are (higher is better)
    """
    if n == 0:
        n = len(grades)
    n = min(n, len(grades))
    _dcg = dcg(grades, n=n)
    _idcg = dcg(sorted(rels, reverse=True), n=n)
    if _idcg > 0.0:
        return _dcg / _idcg
    else:
        return 0.0


def count_ndcg(est_id_m: dict, end2end_gnd_m: dict, end2end_passageID_l: list, topk: int):
    ndcg_l = []
    for queryID in est_id_m.keys():
        assert queryID in end2end_gnd_m.keys()

        est_passageID_l = [end2end_passageID_l[local_passageID] for local_passageID, rank in est_id_m[queryID]]
        grades = [1 if passageID in end2end_gnd_m[queryID] else 0 for passageID in est_passageID_l]

        rels = np.zeros(min(topk, len(grades)), dtype=np.int32)
        rels[:len(end2end_gnd_m[queryID])] = 1
        assert len(grades) == len(rels)
        ndcg_val = ndcg(grades, rels, n=topk)
        ndcg_l.append(ndcg_val)
    assert len(ndcg_l) == len(est_id_m.keys()) and len(ndcg_l) == len(end2end_gnd_m.keys())
    return ndcg_l


'''
used to count the success
'''


def read_end2end_success_gnd(dataset: str, username: str):
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    gnd_jsonl_filename = os.path.join(raw_data_path, f'{dataset}/document/queries_short_answer.gnd.jsonl')
    end2end_gnd_m = {}
    with open(gnd_jsonl_filename, 'r', encoding="utf-8") as f:
        for line in f:
            query_gnd_json = json.loads(line)
            queryID = query_gnd_json['query_id']
            passageID_l = query_gnd_json['answers']
            end2end_gnd_m[queryID] = passageID_l
    return end2end_gnd_m


def read_end2end_success_passageID_l(dataset: str, username: str):
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    collection_filename = os.path.join(raw_data_path, f'{dataset}/document/collection.tsv')
    passageID_local2global_l = []
    passage_l = []
    with open(collection_filename, 'r', encoding="utf-8") as f:
        for line in f:
            txt_l = line.split('\t')
            passageID = int(txt_l[0])
            passage = txt_l[1]
            passageID_local2global_l.append(passageID)
            passage_l.append(passage)
    return passageID_local2global_l, passage_l


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def count_success(est_id_m: dict, gnd_queryID2answer_m: dict,
                  passageID_local2global_l: list,
                  passage_l: list):
    success_l = []
    for queryID in est_id_m.keys():
        assert queryID in gnd_queryID2answer_m.keys()
        contain_golden_answer = False

        answer_l = [ans.strip().strip("\t") for ans in gnd_queryID2answer_m[queryID]]

        for local_passageID, rank in est_id_m[queryID]:
            assert local_passageID < len(passageID_local2global_l), \
                f'program output unexpected itemID, local itemID {local_passageID}'
            global_passageID = passageID_local2global_l[local_passageID]

            passage = passage_l[local_passageID]
            passage = passage.strip().strip("\t")

            for answer in answer_l:
                if answer in passage:
                    contain_golden_answer = True
                    break

            if contain_golden_answer:
                break
        success = 1 if contain_golden_answer else 0
        success_l.append(success)
    assert len(success_l) == len(est_id_m.keys()) and len(success_l) == len(gnd_queryID2answer_m.keys())
    return success_l


'''
used to count the recall
'''


def count_recall(gnd_id_m: dict, est_id_m: dict, topk: int):
    assert len(gnd_id_m.keys()) == len(est_id_m.keys())
    recall_l = []
    for qID in est_id_m.keys():
        assert qID in gnd_id_m.keys()
        est_id_rank = np.array(est_id_m[qID])
        est_id = est_id_rank[:, 0]
        gnd_id_rank = np.array(gnd_id_m[qID])
        gnd_id = gnd_id_rank[:, 0]
        # assert len(est_id_rank) <= topk and len(gnd_id_rank) == topk
        recall = len(np.intersect1d(gnd_id, est_id)) / topk
        recall_l.append(recall)

    return recall_l


def count_recall_id_l(gnd_id_m: dict, est_id_l: list, queryID_l: list, topk: int):
    assert len(gnd_id_m.keys()) == len(est_id_l)
    recall_l = []
    for i, qID in enumerate(queryID_l):
        assert qID in gnd_id_m.keys()
        est_id = np.array(est_id_l[i])
        gnd_id_rank = np.array(gnd_id_m[qID])
        gnd_id = gnd_id_rank[:, 0]
        # assert len(est_id_rank) <= topk and len(gnd_id_rank) == topk
        recall = len(np.intersect1d(gnd_id, est_id)) / topk
        recall_l.append(recall)

    return recall_l


'''
used to count the end2end and vector set search accuracy
'''


def load_groundtruth(username: str, dataset: str, topk: int):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    rawdata_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}'

    mrr_gnd_filename = os.path.join(rawdata_path, 'document', 'queries.gnd.jsonl')
    has_mrr_groundtruth = os.path.exists(mrr_gnd_filename)

    success_gnd_filename = os.path.join(rawdata_path, 'document', 'queries_short_answer.gnd.jsonl')
    has_success_groundtruth = os.path.exists(success_gnd_filename)

    # assert has_mrr_groundtruth or has_success_groundtruth

    # global mrr_groundtruth_m, mrr_passageID_l, success_gnd_m, success_passageID_local2global_l, success_passage_l
    mrr_groundtruth_m, mrr_passageID_l = None, None
    success_gnd_m, success_passageID_local2global_l, success_passage_l = None, None, None
    if has_mrr_groundtruth:
        mrr_groundtruth_m = read_mrr_groundtruth_jsonl(dataset=dataset, username=username)
        mrr_passageID_l = read_mrr_passageID_l(dataset=dataset, username=username)
    if has_success_groundtruth:
        success_gnd_m = read_end2end_success_gnd(dataset=dataset, username=username)
        success_passageID_local2global_l, success_passage_l = \
            read_end2end_success_passageID_l(dataset=dataset,
                                             username=username)
    # if not has_mrr_groundtruth and not has_success_groundtruth:
    #     raise Exception("dataset should have mrr groundtruth or success groundtruth")

    recall_gnd_id_m = read_method_tsv(base_dir=embedding_dir, dataset=dataset, method_name='groundtruth',
                                      topk=topk, build_index_suffix='',
                                      retrieval_suffix='')

    return (has_mrr_groundtruth, mrr_groundtruth_m, mrr_passageID_l), \
        (has_success_groundtruth, success_gnd_m, success_passageID_local2global_l, success_passage_l), \
        recall_gnd_id_m


def count_accuracy(username: str, dataset: str, topk: int,
                   method_name: str, build_index_suffix: str, retrieval_suffix: str,
                   mrr_gnd: tuple, success_gnd: tuple, recall_gnd_id_m: dict):
    answer_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer'

    has_mrr_groundtruth, mrr_groundtruth_m, mrr_passageID_l = mrr_gnd
    has_success_groundtruth, success_gnd_m, success_passageID_local2global_l, success_passage_l = success_gnd

    gnd_id_m = recall_gnd_id_m
    baseline_id_m = read_method_tsv(base_dir=answer_dir, dataset=dataset, method_name=method_name,
                                    topk=topk, build_index_suffix=build_index_suffix,
                                    retrieval_suffix=retrieval_suffix)

    recall_l = count_recall(gnd_id_m=gnd_id_m, est_id_m=baseline_id_m, topk=topk)
    recall_m = {
        'recall_p5': '{:.3f}'.format(np.percentile(recall_l, 5)),
        'recall_p50': '{:.3f}'.format(np.percentile(recall_l, 50)),
        'recall_p95': '{:.3f}'.format(np.percentile(recall_l, 95)),
        'recall_mean': '{:.3f}'.format(np.average(recall_l)),
    }

    mrr_l, success_l = None, None
    mrr_m, success_m = {}, {}
    if has_mrr_groundtruth:
        mrr_l = count_mrr(est_id_m=baseline_id_m, end2end_gnd_m=mrr_groundtruth_m,
                          end2end_passageID_l=mrr_passageID_l)
        e2e_recall_l = count_end2end_recall(est_id_m=baseline_id_m, end2end_gnd_m=mrr_groundtruth_m,
                                            end2end_passageID_l=mrr_passageID_l)
        ndcg_l = count_ndcg(est_id_m=baseline_id_m, end2end_gnd_m=mrr_groundtruth_m,
                            end2end_passageID_l=mrr_passageID_l, topk=topk)
        mrr_m = {
            'mrr_p5': '{:.3f}'.format(np.percentile(mrr_l, 5)),
            'mrr_p50': '{:.3f}'.format(np.percentile(mrr_l, 50)),
            'mrr_p95': '{:.3f}'.format(np.percentile(mrr_l, 95)),
            'mrr_max': '{:.3f}'.format(np.percentile(mrr_l, 100)),
            'mrr_mean': '{:.3f}'.format(np.average(mrr_l)),
            'e2e_recall_p5': '{:.3f}'.format(np.percentile(e2e_recall_l, 5)),
            'e2e_recall_p50': '{:.3f}'.format(np.percentile(e2e_recall_l, 50)),
            'e2e_recall_p95': '{:.3f}'.format(np.percentile(e2e_recall_l, 95)),
            'e2e_recall_max': '{:.3f}'.format(np.percentile(e2e_recall_l, 100)),
            'e2e_recall_mean': '{:.3f}'.format(np.average(e2e_recall_l)),
            'ndcg_p5': '{:.3f}'.format(np.percentile(ndcg_l, 5)),
            'ndcg_p50': '{:.3f}'.format(np.percentile(ndcg_l, 50)),
            'ndcg_p95': '{:.3f}'.format(np.percentile(ndcg_l, 95)),
            'ndcg_max': '{:.3f}'.format(np.percentile(ndcg_l, 100)),
            'ndcg_mean': '{:.3f}'.format(np.average(ndcg_l)),
        }
    if has_success_groundtruth:
        success_l = count_success(est_id_m=baseline_id_m, gnd_queryID2answer_m=success_gnd_m,
                                  passageID_local2global_l=success_passageID_local2global_l,
                                  passage_l=success_passage_l)
        success_m = {
            'success_p5': '{:.3f}'.format(np.percentile(success_l, 5)),
            'success_p50': '{:.3f}'.format(np.percentile(success_l, 50)),
            'success_p95': '{:.3f}'.format(np.percentile(success_l, 95)),
            'success_max': '{:.3f}'.format(np.percentile(success_l, 100)),
            'success_mean': '{:.3f}'.format(np.average(success_l)),
        }
    if not has_mrr_groundtruth and not has_success_groundtruth:
        mrr_m = {}
        success_m = {}
    search_accuracy_m = {**recall_m, **mrr_m, **success_m}

    return recall_l, mrr_l, success_l, search_accuracy_m


def count_accuracy_by_baseline_ip(username: str, dataset: str, topk_l: list,
                                  method_name: str,
                                  build_index_suffix: str,
                                  retrieval_suffix_l: list, retrieval_config_l: list):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    answer_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer'
    performance_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'

    for topk_i, topk in enumerate(topk_l, 0):
        recall_performance_l = []
        for retrieval_suffix, retrieval_config in zip(retrieval_suffix_l[topk_i], retrieval_config_l[topk_i]):
            baseline_id_m = read_method_tsv(base_dir=answer_dir, dataset=dataset, method_name=method_name,
                                            topk=topk, build_index_suffix=build_index_suffix,
                                            retrieval_suffix=retrieval_suffix)

            gnd_id_m = read_method_tsv(base_dir=embedding_dir, dataset=dataset, method_name='groundtruth-ip',
                                       topk=topk, build_index_suffix='',
                                       retrieval_suffix='')
            recall_l = count_recall(gnd_id_m=gnd_id_m, est_id_m=baseline_id_m, topk=topk)

            recall_performance = {
                'recall_p5': np.percentile(recall_l, 5),
                'recall_p50': np.percentile(recall_l, 50),
                'recall_p95': np.percentile(recall_l, 95),
                'recall_mean': np.average(recall_l),
            }
            recall_performance_l.append(recall_performance)
            # print(recall_performance)

        method_performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-time.json'
        time_performance_filename = os.path.join(performance_dir, method_performance_name)
        with open(time_performance_filename, "r", encoding="utf-8") as f:
            method_performance_m = json.load(f)
        time_search_l = method_performance_m['search']
        for i, time_search_performance, retrieval_config, recall_performance in zip(np.arange(len(time_search_l)),
                                                                                    time_search_l,
                                                                                    retrieval_config_l[topk_i],
                                                                                    recall_performance_l):
            time_search_config = time_search_performance['search_config']
            for time_search_key in time_search_config.keys():
                assert time_search_key in retrieval_config
                assert time_search_config[time_search_key] == retrieval_config[time_search_key]
            time_performance = time_search_performance['search_result']
            method_performance_m['search'][i]['search_result'] = {**time_performance, **recall_performance}
            print(method_performance_m['search'][i]['search_result'])
        os.remove(time_performance_filename)

        performance_name = f'{dataset}-retrieval-{method_name}-ip-top{topk}.json' if build_index_suffix == '' else f'{dataset}-retrieval-{method_name}-ip-top{topk}-{build_index_suffix}.json'
        performance_filename = os.path.join(performance_dir, performance_name)
        with open(performance_filename, "w") as f:
            json.dump(method_performance_m, f)


def count_accuracy_by_ID(username: str, dataset: str, topk: int,
                         method_name: str, baseline_id_l: list,
                         build_index_suffix: str, retrieval_suffix: str, retrieval_result_m: dict):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    rawdata_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}'
    answer_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer'
    performance_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'

    has_text_groundtruth = os.path.exists(rawdata_path)

    query_text_filename = os.path.join(rawdata_path, f'document/queries.dev.tsv')
    if has_text_groundtruth:
        queryID_l = []
        with open(query_text_filename, 'r', encoding="utf-8") as f:
            for line in f:
                query_text_l = line.split('\t')
                queryID_l.append(int(query_text_l[0]))
        assert len(queryID_l) == len(baseline_id_l)
    else:
        query_l = np.load(os.path.join(embedding_dir, 'query_embedding.npy'))
        queryID_l = np.arange(len(query_l))

    gnd_id_m = read_method_tsv(base_dir=embedding_dir, dataset=dataset, method_name='groundtruth',
                               topk=topk, build_index_suffix='', retrieval_suffix='')
    recall_l = count_recall_id_l(gnd_id_m=gnd_id_m, est_id_l=baseline_id_l, queryID_l=queryID_l, topk=topk)

    search_result_recall = {
        'recall_p5': '{:.3f}'.format(np.percentile(recall_l, 5)),
        'recall_p50': '{:.3f}'.format(np.percentile(recall_l, 50)),
        'recall_p95': '{:.3f}'.format(np.percentile(recall_l, 95)),
        'recall_mean': '{:.3f}'.format(np.average(recall_l)),
    }

    retrieval_result_json = copy.deepcopy(retrieval_result_m)

    retrieval_result_json['search_accuracy'] = search_result_recall
    print(retrieval_result_json)

    # recall_performance = {
    #     'search_result': {
    #         'recall_p5': np.percentile(recall_l, 5),
    #         'recall_p50': np.percentile(recall_l, 50),
    #         'recall_p95': np.percentile(recall_l, 95),
    #         'recall_mean': np.average(recall_l),
    #     },
    #     'search_config': retrieval_config
    # }

    performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
    performance_filename = os.path.join(performance_dir, performance_name)
    with open(performance_filename, "w") as f:
        json.dump(retrieval_result_json, f)


def count_accuracy_by_ID_ip(username: str, dataset: str, topk: int,
                            method_name: str, baseline_id_l: list,
                            build_index_suffix: str, retrieval_suffix: str, retrieval_result_m: dict):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    performance_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'

    query_l = np.load(os.path.join(embedding_dir, 'query_embedding.npy'))
    vec_dim = query_l.shape[2]
    query_l = query_l.reshape(-1, vec_dim)
    queryID_l = list(np.arange(len(query_l)))

    gnd_id_m = read_method_tsv(base_dir=embedding_dir, dataset=dataset, method_name='groundtruth-ip',
                               topk=topk, build_index_suffix='', retrieval_suffix='')
    recall_l = count_recall_id_l(gnd_id_m=gnd_id_m, est_id_l=baseline_id_l, queryID_l=queryID_l, topk=topk)

    search_result_recall = {
        'recall_p5': np.percentile(recall_l, 5),
        'recall_p50': np.percentile(recall_l, 50),
        'recall_p95': np.percentile(recall_l, 95),
        'recall_mean': np.average(recall_l),
    }

    retrieval_result_json = copy.deepcopy(retrieval_result_m)

    time_search_result = retrieval_result_json['search'][0]['search_result']
    retrieval_result_json['search'][0]['search_result'] = {
        **time_search_result,
        **search_result_recall
    }
    print(retrieval_result_json)

    performance_name = f'{dataset}-retrieval-{method_name}-ip-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
    performance_filename = os.path.join(performance_dir, performance_name)
    with open(performance_filename, "w") as f:
        json.dump(retrieval_result_json, f)


def retrieval_suffix_f(retrieval_config: dict):
    n_max_probe = retrieval_config['n_max_probe']
    refine_topk = retrieval_config['refine_topk']
    search_type = retrieval_config['search_type']
    print(f"retrieval: n_max_probe {n_max_probe}, refine_topk {refine_topk}, search_type {search_type}")

    if search_type == 'linear_scan':
        retrieval_suffix = f'{search_type}-n_max_probe_{n_max_probe}-refine_topk_{refine_topk}'
    elif search_type == 'ta':
        retrieval_suffix = f'{search_type}-n_max_probe_{n_max_probe}-refine_topk_{refine_topk}'
    elif search_type == 'nra':
        retrieval_suffix = f'{search_type}-n_max_probe_{n_max_probe}-refine_topk_{refine_topk}'
    elif search_type == 'ca':
        per_random_access = retrieval_config['per_random_access']
        retrieval_suffix = f'{search_type}-n_max_probe_{n_max_probe}-refine_topk_{refine_topk}-per_random_access_{per_random_access}'
    else:
        raise Exception("not support search type")

    return retrieval_suffix


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for search_type in grid_search_para['search_type']:
        for n_max_probe in grid_search_para['n_max_probe']:
            for refine_topk in grid_search_para['refine_topk']:
                if search_type == 'ca':
                    for per_random_access in grid_search_para['per_random_access']:
                        parameter_l.append(
                            {"search_type": search_type, "n_max_probe": n_max_probe,
                             "refine_topk": refine_topk, "per_random_access": per_random_access})
                else:
                    parameter_l.append(
                        {"search_type": search_type, "n_max_probe": n_max_probe,
                         "refine_topk": refine_topk})
    return parameter_l


if __name__ == '__main__':
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            'dataset_l': ['lotte'],
            'topk_l': [10],
            'build_index_parameter_l': [
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(8 * np.sqrt(x)))), 'n_bit': 2},
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x)))), 'n_bit': 2},
            ],
            'retrieval_parameter_l': [
                {'n_max_probe': 1, 'refine_topk': 100, 'search_type': 'ta'},
                {'n_max_probe': 1, 'refine_topk': 100, 'search_type': 'nra'},
            ],
            'grid_search': True,
            'grid_search_para': {
                'n_max_probe': [1, 4, 8, 16, 32, 64],
                'refine_topk': [10, 50, 100, 200, 300],
                'search_type': ['nra'],
            }
        },
        'local': {
            'username': 'bianzheng',
            'dataset_l': ['lotte-500-gnd'],
            'topk_l': [10],
            'build_index_parameter_l': [
                {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(8 * np.sqrt(x)))), 'n_bit': 2},
            ],
            'retrieval_parameter_l': [
                {'n_max_probe': 1, 'refine_topk': 20, 'search_type': 'nra'},
                {'n_max_probe': 2, 'refine_topk': 20, 'search_type': 'nra'},
                {'n_max_probe': 3, 'refine_topk': 20, 'search_type': 'nra'},
                {'n_max_probe': 4, 'refine_topk': 20, 'search_type': 'nra'},
            ],
            'grid_search': True,
            'grid_search_para': {
                'n_max_probe': [1, 2, 4],
                'refine_topk': [100, 200, 300],
                # 'search_type': ['ta', 'nra', 'ca'],
                'search_type': ['nra'],
                'per_random_access': [10, 20, 100]
            }
        }
    }
    config_name = 'dbg'
    config = config_l[config_name]
    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']
    build_index_parameter_l = config['build_index_parameter_l']
    retrieval_parameter_l = config['retrieval_parameter_l']

    grid_search = config['grid_search']
    if grid_search:
        retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'])
    else:
        retrieval_parameter_l = config['retrieval_parameter_l']

    module_name = 'AggTopk'
    for dataset in dataset_l:
        for build_index_config in build_index_parameter_l:
            embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
            vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
            n_item = np.load(os.path.join(embedding_dir, f'doclens.npy')).shape[0]
            item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
            n_vecs = np.sum(item_n_vec_l)

            n_centroid = build_index_config['n_centroid_f'](n_vecs)
            n_bit = build_index_config['n_bit']

            build_index_suffix = f'n_centroid_{n_centroid}-n_bit_{n_bit}'

            retrieval_suffix_m = {}
            for topk in topk_l:
                mrr_gnd, success_gnd, recall_gnd_id_m = load_groundtruth(username=username, dataset=dataset,
                                                                         topk=topk)
                answer_suffix_l = []
                for retrieval_parameter in retrieval_parameter_l:
                    retrieval_suffix = retrieval_suffix_f(
                        retrieval_config=retrieval_parameter
                    )

                    recall_l, mrr_l, success_l, search_accuracy_m = count_accuracy(
                        username=username, dataset=dataset, topk=topk,
                        method_name=module_name, build_index_suffix=build_index_suffix,
                        retrieval_suffix=retrieval_suffix,
                        mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)
                    print(recall_l, mrr_l, success_l)
