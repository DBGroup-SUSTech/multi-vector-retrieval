import numpy as np
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)

from baseline.ColBERT import run as colbert_run
from script.data import groundtruth
import util
import json


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -r %s' % dire
        print(command)
        os.system(command)


def extract_gnd_mrr(username: str, dataset: str, query_offsetID_l: np.ndarray):
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    gnd_filename = os.path.join(raw_data_path, f'{dataset}/document/queries.gnd.jsonl')
    old_gnd_l = []
    with open(gnd_filename, 'r') as f:
        for line in f:
            old_gnd_l.append(json.loads(line))
    new_gnd_l = []
    query_offset_2_gnd_m = {}
    for query_offsetID in query_offsetID_l:
        gnd_json = old_gnd_l[query_offsetID]
        new_gnd_l.append(gnd_json)
        query_offset_2_gnd_m[query_offsetID] = gnd_json['passage_id']
    return new_gnd_l, query_offset_2_gnd_m


def extract_gnd_openqa(username: str, dataset: str, query_offsetID_l: np.ndarray):
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    gnd_filename = os.path.join(raw_data_path, f'{dataset}/document/queries_short_answer.gnd.jsonl')
    old_gnd_l = []
    with open(gnd_filename, 'r') as f:
        for line in f:
            old_gnd_l.append(json.loads(line))
    new_gnd_l = []
    for query_offsetID in query_offsetID_l:
        gnd_json = old_gnd_l[query_offsetID]
        new_gnd_l.append(gnd_json)
    return new_gnd_l


def generate_data_by_sample(username: str, origin_dataset: str, new_dataset: str,
                            n_sample_item: int, n_sample_query: int):
    raw_data_base_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    delete_file_if_exist(os.path.join(raw_data_base_path, new_dataset))
    os.makedirs(os.path.join(raw_data_base_path, new_dataset, 'document'), exist_ok=False)

    # choose query for training the index
    query_train_old_filename = f"{raw_data_base_path}/{origin_dataset}/document/queries.train.tsv"
    query_train_new_filename = f"{raw_data_base_path}/{new_dataset}/document/queries.train.tsv"
    os.system(f'cp {query_train_old_filename} {query_train_new_filename}')

    # choose query for testing
    query_old_filename = f"{raw_data_base_path}/{origin_dataset}/document/queries.dev.tsv"
    query_new_filename = f"{raw_data_base_path}/{new_dataset}/document/queries.dev.tsv"
    with open(query_old_filename, 'r') as f:
        txt = f.read()
        txt_l = txt.split('\n')
        line_count_l = [1 if line != '' else 0 for line in txt_l]
        n_query = sum(line_count_l)
    print(f"dataset {origin_dataset}, n_total_query {n_query}, n_sample_query {n_sample_query}")
    if n_sample_query != -1:
        assert n_sample_query <= n_query
        line_l = []
        with open(query_old_filename, 'r') as f:
            for line in f:
                line_l.append(line)
        permutation_l = np.sort(np.random.permutation(n_query)[:n_sample_query])
        new_line_l = [line_l[i] for i in permutation_l]
        with open(query_new_filename, 'w') as f:
            for line in new_line_l:
                f.write(line)
        # os.system(f'head -n {n_sample_query} {query_old_filename} > {query_new_filename}')
        query_offsetID_l = permutation_l
    else:
        os.system(f'cp {query_old_filename} {query_new_filename}')
        query_offsetID_l = np.arange(0, n_query)

    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    queries_gnd_json_l_filename = os.path.join(raw_data_path, f'{origin_dataset}/document/queries.gnd.jsonl')
    queries_short_answer_gnd_json_l_filename = os.path.join(raw_data_path,
                                                            f'{origin_dataset}/document/queries_short_answer.gnd.jsonl')
    if os.path.exists(queries_gnd_json_l_filename):
        # choose groundtruth by the query
        groundtruth_l, query_offset_2_gnd_m = extract_gnd_mrr(username=username, dataset=origin_dataset,
                                                              query_offsetID_l=query_offsetID_l)
        groundtruth_filename = f"{raw_data_base_path}/{new_dataset}/document/queries.gnd.jsonl"
        with open(groundtruth_filename, 'w') as f:
            for gnd in groundtruth_l:
                f.write(json.dumps(gnd) + '\n')

        # choose itemID by the groundtruth
        collection_old_filename = f"{raw_data_base_path}/{origin_dataset}/document/collection.tsv"
        collection_new_filename = f"{raw_data_base_path}/{new_dataset}/document/collection.tsv"
        if n_sample_item != -1:
            line_l = []
            passageID_2_lineID_m = {}
            with open(collection_old_filename, 'r') as f:
                for i, line in enumerate(f):
                    passageID = int(line.split('\t')[0])
                    passageID_2_lineID_m[passageID] = i
                    line_l.append(line)
            n_item = len(line_l)
            print(f"new dataset {new_dataset}, n_total_item {n_item}, n_sample_item {n_sample_item}")

            gnd_passageID_cand_l = np.array([], dtype=np.int32)
            for key in query_offset_2_gnd_m.keys():
                gnd_passageID_cand_l = np.union1d(gnd_passageID_cand_l, query_offset_2_gnd_m[key])
            n_gnd_cand = len(gnd_passageID_cand_l)
            for passageID in gnd_passageID_cand_l:
                assert passageID in passageID_2_lineID_m.keys()
            if n_gnd_cand > n_sample_item:
                passageID_l = np.random.choice(gnd_passageID_cand_l, n_sample_item, replace=False)
            else:
                passageID_l = gnd_passageID_cand_l
                passageID_cand_l = list(passageID_2_lineID_m.keys())
                remain_passageID_l = np.setdiff1d(passageID_cand_l, gnd_passageID_cand_l)
                remain_passageID_l = remain_passageID_l[
                    np.random.permutation(len(remain_passageID_l))[:n_sample_item - len(passageID_l)]]
                passageID_l = np.union1d(passageID_l, remain_passageID_l)
            assert len(np.unique(passageID_l)) == n_sample_item and len(passageID_l) == n_sample_item
            lineID_l = np.sort([passageID_2_lineID_m[passageID] for passageID in passageID_l])
            new_line_l = [line_l[i] for i in lineID_l]
            with open(collection_new_filename, 'w') as f:
                for line in new_line_l:
                    f.write(line)
            # os.system(f'head -n {n_sample_item} {collection_old_filename} > {collection_new_filename}')
        else:
            os.system(f'cp {collection_old_filename} {collection_new_filename}')

    if os.path.exists(queries_short_answer_gnd_json_l_filename):
        # choose groundtruth by the query
        groundtruth_l = extract_gnd_openqa(username=username, dataset=origin_dataset,
                                           query_offsetID_l=query_offsetID_l)
        print("finish extract extract_gnd_openqa")
        groundtruth_filename = f"{raw_data_base_path}/{new_dataset}/document/queries_short_answer.gnd.jsonl"
        with open(groundtruth_filename, 'w') as f:
            for gnd in groundtruth_l:
                f.write(json.dumps(gnd) + '\n')

        # choose itemID by the groundtruth
        collection_old_filename = f"{raw_data_base_path}/{origin_dataset}/document/collection.tsv"
        collection_new_filename = f"{raw_data_base_path}/{new_dataset}/document/collection.tsv"
        if not os.path.exists(collection_new_filename):
            if n_sample_item != -1:
                line_l = []
                with open(collection_old_filename, 'r') as f:
                    for i, line in enumerate(f):
                        line_l.append(line)
                n_item = len(line_l)
                print(f"new dataset {new_dataset}, n_total_item {n_item}, n_sample_item {n_sample_item}")

                assert n_sample_item <= n_item
                new_line_l = [line_l[i] for i in range(n_sample_item)]
                with open(collection_new_filename, 'w') as f:
                    for line in new_line_l:
                        f.write(line)
                # os.system(f'head -n {n_sample_item} {collection_old_filename} > {collection_new_filename}')
            else:
                os.system(f'cp {collection_old_filename} {collection_new_filename}')

    if not os.path.exists(queries_gnd_json_l_filename) and not os.path.exists(queries_short_answer_gnd_json_l_filename):
        raise Exception("do not support such ground truth type")


def build_basic_index(username: str, origin_dataset: str, new_dataset: str, n_sample_item: int, n_sample_query: int,
                      topk_l: list):
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/Embedding'
    os.system(f'rm -r {embedding_path}/{new_dataset}')

    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/Index'
    os.system(f'rm -r {index_path}/{new_dataset}')

    rawdata_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/RawData'
    os.system(f'rm -r {rawdata_path}/{new_dataset}')

    result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/answer'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    os.system(f'rm {result_answer_path}/{new_dataset}-*')
    os.system(f'rm {result_performance_path}/{new_dataset}-*')

    print(bcolors.OKGREEN + f"sample raw data start {new_dataset}" + bcolors.ENDC)
    generate_data_by_sample(username=username,
                            origin_dataset=origin_dataset, new_dataset=new_dataset,
                            n_sample_item=n_sample_item, n_sample_query=n_sample_query)
    print(bcolors.OKGREEN + f"sample raw data finish {new_dataset}" + bcolors.ENDC)

    print(bcolors.OKGREEN + f"plaid build index start {new_dataset}" + bcolors.ENDC)
    colbert_run.build_index_official(username=username, dataset=new_dataset)
    # colbert_run.build_index_official(username=username, dataset=new_dataset,
    #                                  **method_config_m['plaid']['build_index'])
    print(bcolors.OKGREEN + f"plaid build index finish {new_dataset}" + bcolors.ENDC)

    module_name = 'BruteForceProgressive'
    print(bcolors.OKGREEN + f"groundtruth start {new_dataset}" + bcolors.ENDC)
    # util.compile_file(username=username, module_name=module_name, is_debug=False)
    est_dist_l_l, est_id_l_l = groundtruth.gnd_cpp(username=username, dataset=new_dataset, topk_l=topk_l,
                                                   compile_file=False, module_name=module_name)
    for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
        groundtruth.save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=new_dataset,
                                 topk=topk)
    print(bcolors.OKGREEN + f"groundtruth end {new_dataset}" + bcolors.ENDC)


def retrieval(username: str, dataset: str, retrieval_parameter_l: list, topk_l: list):
    print(bcolors.OKGREEN + f" plaid retrieval start {dataset}" + bcolors.ENDC)
    for topk in topk_l:
        colbert_run.retrieval_official(username=username, dataset=dataset,
                                       topk=topk, search_config_l=retrieval_parameter_l)
    print(bcolors.OKGREEN + f" plaid retrieval end {dataset}" + bcolors.ENDC)


if __name__ == '__main__':
    # 'ndocs': searcher.config.ndocs,
    # 'ncells': searcher.config.ncells,
    # 'centroid_score_threshold': searcher.config.centroid_score_threshold
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            'topk_l': [10],
            'retrieval_parameter_l': [
                {'ndocs': 4, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 8, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 16, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 32, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 64, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 128, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 256, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 512, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 1024, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 2048, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 4096, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},
                {'ndocs': 8192, 'ncells': 1, 'centroid_score_threshold': 0.5, "n_thread": 16},

                # {'ndocs': 8192, 'ncells': 4, 'centroid_score_threshold': 0.1, "n_thread": 16},
                # {'ndocs': 8192, 'ncells': 4, 'centroid_score_threshold': 0.3, "n_thread": 16},
                # {'ndocs': 8192, 'ncells': 4, 'centroid_score_threshold': 0.7, "n_thread": 16},
            ],
            'sample_dataset_info_l': [
                ['lotte', 'lotte-100K', 100_000, -1],
                ['msmacro', 'msmacro-100K', 100_000, -1],
            ]
        },
        'local': {
            'username': 'bianzheng',
            'topk_l': [10, 50],
            'retrieval_parameter_l': [
                # {'ndocs': 10, 'ncells': 3, 'centroid_score_threshold': 0.5, "n_thread": 1},
                {'ndocs': 32, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                # {'ndocs': 128, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                # {'ndocs': 512, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1}
                # {'ndocs': 2048, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1}
            ],
            'sample_dataset_info_l': [
                # ['lotte', 'lotte-100-gnd', 100, 100],
                ['lotte', 'lotte-500-gnd', 500, 100],
                # ['lotte', 'lotte-3K-gnd', 3000, 100],
                # ['wiki-nq', 'wiki-nq-500', 500, 10],
                # ['wikipedia', 'wikipedia-1K', 1000, 100],
                # ['msmacro', 'msmacro-500-gnd', 500, 100],
            ]
        }
    }
    host_name = 'local'
    load_index = False

    config = config_l[host_name]
    username = config['username']
    topk_l = config['topk_l']

    retrieval_parameter_l = config['retrieval_parameter_l']
    sample_dataset_info_l = config['sample_dataset_info_l']

    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    queries_gnd_json_l_filename = os.path.join(raw_data_path,
                                               f'{sample_dataset_info_l[0][0]}/document/queries.gnd.jsonl')
    queries_short_answer_gnd_json_l_filename = os.path.join(raw_data_path,
                                                            f'{sample_dataset_info_l[0][0]}/document/queries_short_answer.gnd.jsonl')
    print(queries_gnd_json_l_filename, os.path.exists(queries_gnd_json_l_filename))
    print(queries_short_answer_gnd_json_l_filename, os.path.exists(queries_short_answer_gnd_json_l_filename))

    module_name = 'BruteForceProgressive'
    util.compile_file(username=username, module_name=module_name, is_debug=True)
    for origin_dataset, new_dataset, n_sample_item, n_sample_query in sample_dataset_info_l:
        build_basic_index(username=username, origin_dataset=origin_dataset, new_dataset=new_dataset,
                          n_sample_item=n_sample_item, n_sample_query=n_sample_query,
                          topk_l=topk_l)
        retrieval(username=username, dataset=new_dataset, retrieval_parameter_l=retrieval_parameter_l, topk_l=topk_l)
