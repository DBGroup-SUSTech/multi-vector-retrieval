import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'Dessert')
sys.path.append(ROOT_PATH)

from baseline.ColBERT import run as colbert_run
from script.data import groundtruth
import util


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


if __name__ == '__main__':
    username = 'bianzheng'
    topk_l = [10, 100]
    dataset_l = [
        'lotte-500-gnd',

        # 'lotte',
        # 'msmacro',
        # 'wikipedia',
        # 'lotte-lifestyle',
        # 'quora',
        # 'hotpotqa',
        # 'wiki-nq',
    ]
    for dataset in dataset_l:
        print(bcolors.OKGREEN + f"plaid start {dataset}" + bcolors.ENDC)
        colbert_run.build_index_official(username=username, dataset=dataset)
        print(bcolors.OKGREEN + f"plaid finish {dataset}" + bcolors.ENDC)

        module_name = 'BruteForceProgressive'
        print(bcolors.OKGREEN + f"groundtruth start {dataset}" + bcolors.ENDC)
        util.compile_file(username=username, module_name=module_name, is_debug=True)
        est_dist_l_l, est_id_l_l = groundtruth.gnd_cpp(username=username, dataset=dataset, topk_l=topk_l,
                                                       compile_file=False, module_name=module_name)
        for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
            groundtruth.save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset,
                                     topk=topk)
        print(bcolors.OKGREEN + f"groundtruth end {dataset}" + bcolors.ENDC)
