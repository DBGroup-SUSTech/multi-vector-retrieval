import time

import importlib
import os
import numpy as np
import typing
import tqdm
import sys

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.data import util


def merge_result(final_distance_l: np.ndarray, final_id_l: np.ndarray,
                 distance_l: np.ndarray, id_l: np.ndarray):
    if np.array_equal(final_distance_l, []) and np.array_equal(final_id_l, []):
        return distance_l, id_l
    assert not np.array_equal(final_distance_l, []) and not np.array_equal(final_id_l, [])
    assert final_distance_l.shape[0] == final_id_l.shape[0] and final_id_l.shape[0] == distance_l.shape[0] and \
           distance_l.shape[0] == id_l.shape[0]
    assert final_distance_l.shape[1] == final_id_l.shape[1] and final_id_l.shape[1] == distance_l.shape[1] and \
           distance_l.shape[1] == id_l.shape[1]
    n_query = final_distance_l.shape[0]
    topk = final_distance_l.shape[1]

    result_dist_l = []
    result_id_l = []
    for queryID in range(n_query):
        tmp_dist_l = np.append(final_distance_l[queryID], distance_l[queryID])
        tmp_id_l = np.append(final_id_l[queryID], id_l[queryID])

        greater_id_l = np.argsort(tmp_dist_l)[-topk:]
        result_dist_l.append(tmp_dist_l[greater_id_l])
        result_id_l.append(tmp_id_l[greater_id_l])
    return np.array(result_dist_l), np.array(result_id_l)


def gnd_cpp(username: str, dataset: str, topk_l: list, n_batch_read: int = 1,
            compile_file: bool = True, module_name='BruteForceProgressive'):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_l = np.load(os.path.join(embedding_dir, 'query_embedding.npy'))

    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')
    vec_dim = np.load(os.path.join(base_embedding_dir, f'encoding0_float32.npy')).shape[1]

    final_distance_l = [[]] * len(topk_l)
    final_id_l = [[]] * len(topk_l)

    if compile_file:
        util.compile_file(username=username, module_name=module_name, is_debug=False)
    module = importlib.import_module(module_name)

    n_chunk = util.get_n_chunk(base_embedding_dir)
    print(f"n_chunk {n_chunk}")
    index = module.DocRetrieval(query_l=query_l, vec_dim=vec_dim)
    accu_itemID = 0
    for batchID, start_chunkID in enumerate(tqdm.trange(0, n_chunk, n_batch_read), 0):
        end_chunkID = min(start_chunkID + n_batch_read, n_chunk)

        start_load_time = time.time()
        n_item_batch_chunk = 0
        itemlen_batch_l = []
        item_vecs_batch_l = []
        for chunkID in range(start_chunkID, end_chunkID, 1):
            # print(f"batch_chunkID {batchID} chunkID {chunkID}")
            itemlen_l = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
            itemlen_batch_l = np.append(itemlen_batch_l, itemlen_l)

            item_vecs_l = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
            item_vecs_batch_l = np.append(item_vecs_batch_l, item_vecs_l)

            n_item_chunk = len(itemlen_l)
            n_item_batch_chunk += n_item_chunk

        item_vecs_batch_l = np.array(item_vecs_batch_l).reshape(-1, vec_dim)

        item_l = [
            util.item_vecs_in_chunk(vecs_l=item_vecs_batch_l, itemlen_l=itemlen_batch_l, itemID=itemID_batch_chunk) for
            itemID_batch_chunk in range(n_item_batch_chunk)]
        end_load_time = time.time()
        print("load embedding time {:.2f}s".format(end_load_time - start_load_time))

        assert vec_dim == item_l[0].shape[1]
        itemID_l = np.arange(accu_itemID, accu_itemID + len(item_l))
        index.computeScore(item_l=item_l, itemID_l=itemID_l)

        del item_l

        for topk_i, topk in enumerate(topk_l, 0):
            distance_l, id_l = index.searchKNN(topk=topk)

            # start_time = time.time()
            tmp_final_distance_l, tmp_final_id_l = module.DocRetrieval.merge_result(np.array(final_distance_l[topk_i]),
                                                                                    np.array(final_id_l[topk_i]),
                                                                                    distance_l, id_l)
            # end_time = time.time()
            # print(f"merge_result time {end_time - start_time:.2f}s, topk {topk}")

            # tmp_final_distance_l, tmp_final_id_l = merge_result(np.array(final_distance_l[topk_i]),
            #                                                     np.array(final_id_l[topk_i]),
            #                                                     distance_l, id_l)
            final_distance_l[topk_i] = tmp_final_distance_l
            final_id_l[topk_i] = tmp_final_id_l
        accu_itemID += n_item_batch_chunk
    index.finish_compute()
    del index, module
    return final_distance_l, final_id_l


def test_gnd(*, est_dist_l: typing.List[typing.List[float]], est_id_l: typing.List[typing.List[int]],
             gnd_dist_l: typing.List[typing.List[float]], gnd_id_l: typing.List[typing.List[int]], topk: int):
    assert len(est_dist_l) == len(gnd_dist_l) and len(est_id_l) == len(gnd_id_l) and len(est_dist_l) == len(est_id_l)
    n_query = len(est_dist_l)
    for est_dist, est_id, gnd_dist, gnd_id, qID in zip(est_dist_l, est_id_l, gnd_dist_l, gnd_id_l, np.arange(n_query)):
        # assert len(np.intersect1d(est_dist, gnd_dist)) == len(est_dist) and len(est_dist) == len(gnd_dist) and len(est_dist) == topk
        assert len(np.intersect1d(est_id, gnd_id)) == len(est_id) and len(est_id) == len(gnd_id) and \
               len(est_id) == topk, f'error qID {qID} \nest_dist {est_dist} \nest_id {est_id} \ngnd_dist {gnd_dist} \ngnd_id {gnd_id}'
        for rank_i in range(topk):
            est_d, est_i = est_dist[rank_i], est_id[rank_i]
            gnd_d, gnd_i = gnd_dist[rank_i], gnd_id[rank_i]
            assert np.abs(
                est_d - gnd_d) <= 0.01, f'error qID {qID} rank_i {rank_i}\nest_dist {est_dist} \nest_id {est_id} \ngnd_dist {gnd_dist} \ngnd_id {gnd_id}'
            assert gnd_i == est_i, f'error qID {qID} rank_i {rank_i}\nest_dist {est_dist} \nest_id {est_id} \ngnd_dist {gnd_dist} \ngnd_id {gnd_id}, {gnd_i}'
    return True


def save_gnd_npy(gnd_dist_l: np.ndarray, gnd_id_l: np.ndarray, username: str, dataset: str, topk: int):
    gnd_dist_l = gnd_dist_l.astype('float32')
    gnd_id_l = gnd_id_l.astype('int32')

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    np.save(os.path.join(embedding_dir, f'gnd_distance_top{topk}.npy'), gnd_dist_l)
    np.save(os.path.join(embedding_dir, f'gnd_id_top{topk}.npy'), gnd_id_l)


def save_gnd_tsv(gnd_dist_l: np.ndarray, gnd_id_l: np.ndarray, username: str, dataset: str, topk: int):
    gnd_dist_l = gnd_dist_l.astype('float32')
    gnd_id_l = gnd_id_l.astype('int32')

    gnd_rank_l = np.argsort(gnd_dist_l, axis=1)[:, ::-1]
    build_index_suffix = ''
    retrieval_suffix = ''

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    groundtruth_filename = os.path.join(embedding_dir, f'{dataset}-groundtruth-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv')

    rawdata_path = f'/home/{username}/Dataset/multi-vector-retrieval/RawData'
    query_text_filename = os.path.join(rawdata_path, f'{dataset}/document/queries.dev.tsv')
    if os.path.isfile(query_text_filename):
        qID_l = []
        with open(query_text_filename, 'r', encoding='utf-8') as f:
            for line in f:
                query_text_l = line.split('\t')
                qID_l.append(int(query_text_l[0]))
            assert len(qID_l) == len(gnd_dist_l)
    else:
        qID_l = np.arange(len(gnd_dist_l))

    with open(groundtruth_filename, 'w', encoding='utf-8') as f:
        assert len(gnd_dist_l) == len(gnd_id_l) and len(gnd_dist_l) == len(gnd_rank_l)
        for qID, gnd_dist, gnd_id, gnd_rank in zip(qID_l, gnd_dist_l, gnd_id_l, gnd_rank_l):
            assert len(gnd_dist) == len(gnd_id) and len(gnd_dist) == len(gnd_rank) and len(gnd_dist) == topk
            for rank in range(topk):
                rank_i = gnd_rank[rank]
                tmp_dist = gnd_dist[rank_i]
                tmp_itemID = gnd_id[rank_i]
                f.write(f'{qID}\t{tmp_itemID}\t{rank + 1}\t{tmp_dist}\n')


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
    module_name = 'BruteForceProgressive'
    n_batch_read = 2

    # dataset_l = ['fake-normal-tiny', 'fake-normal', 'fake-normal-huge']
    # dataset_l = ['fake-normal-tiny', 'fake-normal']
    dataset_l = ['fake-normal']
    # dataset_l = ['fake-normal-tiny']
    # dataset_l = ['fake-normal-huge']
    # topk_l = [10, 50, 100]
    topk_l = [50]
    # topk_l = [10]
    no_bug = True
    is_debug = True
    util.compile_file(username=username, module_name=module_name, is_debug=is_debug, move_path='evaluation')
    for dataset in dataset_l:

        print(f"dataset {dataset}, topk_l {topk_l}")
        est_dist_l_l, est_id_l_l = gnd_cpp(username=username, dataset=dataset, n_batch_read=n_batch_read, topk_l=topk_l,
                                           module_name=module_name, compile_file=False)
        # save_gnd(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset, topk=topk)
        # save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset, topk=topk)

        for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
            import _gnd_py

            gnd_dist_l, gnd_id_l = _gnd_py.gnd_py(username=username, dataset=dataset, topk=topk)
            # print(gnd_dist_l)
            # print(gnd_id_l)

            no_bug = no_bug and test_gnd(est_dist_l=est_dist_l, est_id_l=est_id_l,
                                         gnd_dist_l=gnd_dist_l, gnd_id_l=gnd_id_l, topk=topk)
    if no_bug:
        print('good, no bug')

    # dataset = 'lotte'
    # print(f"groundtruth start {dataset}")
    # util.compile_file(username=username, module_name=module_name, is_debug=False, move_path='evaluation')
    # est_dist_l_l, est_id_l_l = gnd_cpp(username=username, dataset=dataset, topk_l=topk_l,
    #                                    compile_file=False, module_name=module_name)
    # for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
    #     save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset,
    #                  topk=topk)
    # print(f"groundtruth end {dataset}")
