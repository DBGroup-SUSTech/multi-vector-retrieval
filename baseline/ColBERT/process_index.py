import os
from os import listdir
from os.path import isfile, join
import re
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from colbert.data import Queries
import numpy as np
import torch


def get_n_chunk(base_dir: str):
    filename_l = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = np.sort(match_chunkID_l[match_chunkID_l != np.array(None)])
    return match_chunkID_l[-1]


def process_index(username: str, dataset: str, rootpath: str):
    # move embeddings to the embedding file
    embedding_old_index_path = os.path.join(rootpath, 'ColBERT', 'embedding_vector/*')
    embedding_path = os.path.join(rootpath, f'Dataset/{dataset}/embedding')
    os.makedirs(embedding_path, exist_ok=True)
    os.system(f'mv {embedding_old_index_path} {embedding_path}')

    # move plaid index file to the dataset
    plaid_index_path = os.path.join(rootpath, f'Dataset/{dataset}/index/plaid')
    plaid_old_index_path = os.path.join(rootpath, 'ColBERT', f'experiments/{dataset}/indexes/{dataset}/*')
    os.makedirs(plaid_index_path, exist_ok=True)
    os.system(f'mv {plaid_old_index_path} {plaid_index_path}')

    dessert_index_path = os.path.join(rootpath, f'Dataset/{dataset}/index/dessert')
    os.makedirs(dessert_index_path, exist_ok=True)

    centroids_load_path = os.path.join(plaid_index_path, 'centroids.pt')
    centroids = torch.load(centroids_load_path).cpu()
    print(centroids.shape)
    centroids_save_path = os.path.join(dessert_index_path, 'centroids.npy')
    np.save(centroids_save_path, centroids.cpu().numpy().astype('float32'))

    embedding_base_path = os.path.join(embedding_path, 'base_embedding')
    n_chunk = get_n_chunk(base_dir=embedding_base_path)
    total_centroid_id = np.array([])
    total_doclen = np.array([])
    for i in range(0, n_chunk + 1, 1):
        centroid_id_l = torch.load(os.path.join(plaid_index_path, f'{i}.codes.pt'))
        # print(len(centroid_id_l))
        total_centroid_id = np.concatenate([total_centroid_id, centroid_id_l])

        doclens_path = os.path.join(embedding_base_path, f'doclens{i}.npy')
        doclen = np.load(doclens_path)
        total_doclen = np.concatenate([total_doclen, doclen])
        # print(doclen, doclen.shape, np.sum(doclen))

        if i % 20 == 0:
            print(f'load embedding{i}')

    np.save(os.path.join(dessert_index_path, f'all_centroid_ids.npy'), total_centroid_id.astype('int32'))
    np.save(os.path.join(embedding_path, f'doclens.npy'), total_doclen.astype('int32'))
    assert len(total_centroid_id) == np.sum(total_doclen)
    print(len(total_centroid_id))


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-small'
    process_index(username=username, dataset=dataset, rootpath=rootpath)
