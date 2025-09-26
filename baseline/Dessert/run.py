import numpy as np
import torch
import os
import re
import sys

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)

from baseline.Dessert.experiments import run_dessert


# needed extract file:
# centroids.npy: {dessert_index_path}/centroids.npy
# all_centroid_ids.npy: {dessert_index_path}/all_centroid_ids.npy
# doclens.npy: {embedding_path}/doclens.npy

# not process
# encodings{i}_float32.py: {embedding_path}/base_embedding/encodings{i}_float32.py
# queries.dev.small.tsv: {document_path}/queries.dev.tsv
# small_queries_embeddings.npy: {embedding_path}/query_embedding.npy

def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -r %s' % dire
        print(command)
        os.system(command)


def get_n_chunk(base_dir: str):
    filename_l = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)


def build_index(username: str, dataset: str,
                n_table: int = 32, hashes_per_table: int = -1):
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    plaid_index_path = os.path.join(index_path, 'plaid')

    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'

    dessert_index_path = os.path.join(index_path, 'dessert')

    if not os.path.exists(dessert_index_path):
        delete_file_if_exist(dessert_index_path)
        os.makedirs(dessert_index_path, exist_ok=False)

        centroids = torch.load(os.path.join(plaid_index_path, 'centroids.pt'), map_location=torch.device('cpu'))
        np.save(os.path.join(dessert_index_path, 'centroids.npy'), centroids.numpy().astype('float32'))

        total_centroid_id = np.array([])
        n_chunk = get_n_chunk(os.path.join(embedding_path, 'base_embedding'))
        for chunkID in range(n_chunk):
            centroid_id_l = torch.load(os.path.join(plaid_index_path, f'{chunkID}.codes.pt'),
                                       map_location=torch.device('cpu'))
            # print(len(centroid_id_l))
            total_centroid_id = np.concatenate([total_centroid_id, centroid_id_l])

            # print(doclen, doclen.shape, np.sum(doclen))

            if chunkID % 20 == 0:
                print(f'load embedding{chunkID}')

        np.save(os.path.join(dessert_index_path, f'all_centroid_ids.npy'), total_centroid_id.astype('int32'))
        print("# of passage:", len(total_centroid_id))

    run_dessert.build_index(username=username, dataset=dataset,
                            num_tables=n_table, hashes_per_table=hashes_per_table)


def retrieval(username: str, dataset: str,
              topk: int, retrieval_config_l: list,
              n_table: int = 32):
    # from experiments import run_dessert_integrate
    # run_dessert_integrate.run_experiment(username=username, dataset=dataset,
    #                            top_k_return=10,
    #                            initial_filter_k=initial_filter_k,
    #                            nprobe_query=nprobe_query,
    #                            remove_centroid_dupes=remove_centroid_dupes,
    #                            hashes_per_table=hashes_per_table,
    #                            num_tables=num_tables)

    # initial_filter_k: int = 128
    # nprobe_query: int = 4
    # remove_centroid_dupes: bool = False,
    # hashes_per_table: int = 6
    # num_tables: int = 32

    return run_dessert.retrieval(username=username, dataset=dataset,
                                 topk=topk,
                                 num_tables=n_table,
                                 retrieval_config_l=retrieval_config_l)


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-small'
    # run(username=username, dataset=dataset)
