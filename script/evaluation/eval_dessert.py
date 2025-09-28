import numpy as np
import os
import sys
import argparse

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'Dessert')
sys.path.append(ROOT_PATH)

from baseline.Dessert import run as dessert_run


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


def build_basic_index(username: str, build_index_config: dict, dataset: str):
    print(
        bcolors.OKGREEN + f"dessert build index start {dataset} {build_index_config}" + bcolors.ENDC)
    dessert_run.build_index(username=username, dataset=dataset, **build_index_config)
    print(
        bcolors.OKGREEN + f"dessert build index finish {dataset} {build_index_config}" + bcolors.ENDC)


def retrieval(username: str, dataset: str, build_index_config: dict, retrieval_config_l: list, topk_l: list):
    print(bcolors.OKGREEN + f" dessert retrieval end {dataset} {build_index_config}" + bcolors.ENDC)
    for topk in topk_l:
        dessert_run.retrieval(username=username, dataset=dataset,
                              topk=topk,
                              retrieval_config_l=retrieval_config_l,
                              **build_index_config)
    print(bcolors.OKGREEN + f" dessert retrieval end {dataset} {build_index_config}" + bcolors.ENDC)


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for initial_filter_k in grid_search_para['initial_filter_k']:
        for nprobe_query in grid_search_para['nprobe_query']:
            for remove_centroid_dupes in grid_search_para['remove_centroid_dupes']:
                for n_thread in grid_search_para['n_thread']:
                    parameter_l.append(
                        {'initial_filter_k': initial_filter_k, "nprobe_query": nprobe_query,
                         'remove_centroid_dupes': remove_centroid_dupes, "n_thread": n_thread})
    return parameter_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--host_name', type=str, default='local')
    parser.add_argument('--n_table', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='lotte-500-gnd')

    config_l = {
        'dbg': {
            'username': 'username1',
            'topk_l': [10, 100, 1000],
            'retrieval_parameter_l': [
                {'initial_filter_k': 32, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 64, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 128, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 256, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 512, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 1024, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 2048, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 4096, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 8192, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 16384, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 16384, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 32768, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 65536, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 131072, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 262144, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'initial_filter_k': [32, 64, 128, 256, 512, 1024, 2048],
                    'nprobe_query': [1, 2, 4, 8],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
                100: {
                    'initial_filter_k': [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                    'nprobe_query': [1, 2, 4, 8],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
                1000: {
                    'initial_filter_k': [1024, 2048, 4096, 8192, 16384],
                    'nprobe_query': [1, 2, 4, 8],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
            }
        },
        'local': {
            'username': 'username2',
            'topk_l': [10],
            'retrieval_parameter_l': [
                {'initial_filter_k': 32, "nprobe_query": 4, 'remove_centroid_dupes': True, "n_thread": 8},
                {'initial_filter_k': 64, "nprobe_query": 4, 'remove_centroid_dupes': True, "n_thread": 8},
                {'initial_filter_k': 128, "nprobe_query": 4, 'remove_centroid_dupes': True, "n_thread": 8}
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'initial_filter_k': [10, 30],
                    'nprobe_query': [1, 2],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
                50: {
                    'initial_filter_k': [50, 70, 90],
                    'nprobe_query': [1, 2],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
            }
        }
    }
    args = parser.parse_args()
    host_name = args.host_name
    n_table = args.n_table
    dataset = args.dataset

    config = config_l[host_name]
    username = config['username']
    topk_l = config['topk_l']

    build_basic_index(username=username, build_index_config={'n_table': n_table}, dataset=dataset)

    for topk in topk_l:
        grid_search = config['grid_search']
        if grid_search:
            retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
        else:
            retrieval_parameter_l = config['retrieval_parameter_l']

        retrieval(username=username, dataset=dataset, build_index_config={'n_table': n_table},
                  retrieval_config_l=retrieval_parameter_l, topk_l=[topk])
