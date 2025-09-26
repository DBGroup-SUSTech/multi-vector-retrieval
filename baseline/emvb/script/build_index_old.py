import faiss
import numpy as np
from faiss.loader import swig_ptr
from tqdm.notebook import tqdm
import os
import time
from faiss.contrib.inspect_tools import get_invlist
from faiss.contrib import inspect_tools
import random
import sys

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from baseline.emvb.script import util


def sample_itemID4kmeans(n_item):
    # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
    # Keep in mind that, say, 15% still means at least 100k.
    # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
    # Then we subsample the vectors to 100 * num_partitions

    typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
    n_sample_pid = 16 * np.sqrt(typical_doclen * n_item)
    # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
    n_sample_pid = min(1 + int(n_sample_pid), n_item)

    random.seed(12345)
    sample_pid_l = random.sample(range(n_item), n_sample_pid)

    return sample_pid_l


def get_sample_vecs_l(sample_itemID_l: list, DEFAULT_CHUNKSIZE: int, username: str, dataset: str, vec_dim: int):
    sample_itemID_l = np.sort(sample_itemID_l)
    item_chunkID_l = [_ // DEFAULT_CHUNKSIZE for _ in sample_itemID_l]
    item_chunk_offset_l = [_ % DEFAULT_CHUNKSIZE for _ in sample_itemID_l]
    chunkID2offset_m = {}
    for chunkID, chunk_offset in zip(item_chunkID_l, item_chunk_offset_l):
        if chunkID not in chunkID2offset_m:
            chunkID2offset_m[chunkID] = [chunk_offset]
        else:
            chunkID2offset_m[chunkID].append(chunk_offset)

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')

    item_n_vecs_l = np.load(os.path.join(embedding_dir, 'doclens.npy')).astype(np.uint64)
    item_n_vecs_offset_l = np.cumsum(item_n_vecs_l)
    item_n_vecs_offset_l = np.concatenate(([0], item_n_vecs_offset_l))
    n_item = len(item_n_vecs_l)

    print("load chunk data")
    sample_vecs_l = np.array([])
    sample_item_n_vec_l = np.array([], dtype=np.uint32)
    vecsID_l = np.array([])
    for chunkID, offset_itemID_l in chunkID2offset_m.items():
        print(f"chunkID {chunkID}, n_item in chunk {len(offset_itemID_l)}")
        item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        item_n_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        item_n_vecs_offset_chunk_l = np.cumsum(item_n_vecs_l_chunk)
        item_n_vecs_offset_chunk_l = np.concatenate(([0], item_n_vecs_offset_chunk_l))
        item_n_vecs_offset_chunk_l = np.array(item_n_vecs_offset_chunk_l, dtype=np.uint64)

        base_itemID = chunkID * DEFAULT_CHUNKSIZE
        vecsID_l_chunk = np.array([])
        item_n_vec_l_chunk = np.array([])

        for offset_itemID in offset_itemID_l:
            itemID = base_itemID + offset_itemID
            item_n_vecs = item_n_vecs_l[itemID]
            base_vecID_chunk = item_n_vecs_offset_chunk_l[offset_itemID]
            vecsID_l_chunk = np.concatenate(
                (vecsID_l_chunk, np.arange(base_vecID_chunk, base_vecID_chunk + item_n_vecs, 1))).astype(np.uint64)
            item_n_vec_l_chunk = np.concatenate((item_n_vec_l_chunk, [item_n_vecs]))

        sample_vecs_l_chunk = item_vecs_l_chunk[vecsID_l_chunk, :]
        sample_vecs_l = sample_vecs_l_chunk if len(sample_vecs_l) == 0 else np.concatenate(
            (sample_vecs_l, sample_vecs_l_chunk)).reshape(-1, vec_dim)

        vecsID_l_chunk = vecsID_l_chunk + item_n_vecs_offset_l[base_itemID]
        vecsID_l = np.concatenate(
            (vecsID_l, vecsID_l_chunk))

        sample_item_n_vec_l = np.concatenate((sample_item_n_vec_l, item_n_vec_l_chunk))

        print("finish load chunkID")

    sample_vecs_l = sample_vecs_l.reshape(-1, vec_dim)

    assert len(vecsID_l) == len(
        sample_vecs_l), f"len(vecsID_l) {len(vecsID_l)}, len(sample_vecs_l) {len(sample_vecs_l)}"
    vecsID_l = np.array(vecsID_l, dtype=np.uint64)

    sample_item_n_vec_l = sample_item_n_vec_l.astype(np.uint32)

    return sample_vecs_l, sample_item_n_vec_l, vecsID_l


def sample_vector(username: str, dataset: str):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
    item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
    n_item = item_n_vec_l.shape[0]

    print("sample itemID for kmeans")
    sample_itemID_l = sample_itemID4kmeans(n_item=n_item)
    DEFAULT_CHUNKSIZE = len(np.load(os.path.join(embedding_dir, 'base_embedding', f'doclens{0}.npy')))

    print("read sample vector from disk")
    sample_vecs_l, sample_item_n_vec_l, _ = get_sample_vecs_l(sample_itemID_l=sample_itemID_l,
                                                              DEFAULT_CHUNKSIZE=DEFAULT_CHUNKSIZE,
                                                              username=username, dataset=dataset, vec_dim=vec_dim)
    return sample_vecs_l, sample_item_n_vec_l


def index_add_vector(username: str, dataset: str, index: faiss.IndexIVFPQ):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')

    n_chunk = util.get_n_chunk(base_embedding_dir)
    for chunkID in tqdm(range(n_chunk)):
        itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        n_vec_chunk = int(np.sum(itemlen_l_chunk))
        item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))

        index.add(item_vecs_l_chunk)


def build_index(username: str, dataset: str, n_centroid: int, pq_n_partition: int, pq_n_bit_per_partition: int):
    '''build the IVFPQ index'''
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/emvb'

    os.makedirs(index_path, exist_ok=True)

    np.save(os.path.join(index_path, "doclens.npy"),
            np.load(f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/doclens.npy').astype(
                np.int32))

    # n_query = np.load(f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/query_embedding.npy').shape[0]
    # qID_l = np.arange(n_query)
    # np.savetxt(os.path.join(index_path, "qID_l.txt"), qID_l, fmt='%d')

    query_text_fname = f'/home/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}/document/queries.dev.tsv'
    qID_l = []
    with open(query_text_fname, 'r') as f:
        for line in f:
            if line == '':
                continue
            qID, txt = line.split('\t')
            qID = int(qID)
            qID_l.append(qID)
    np.savetxt(os.path.join(index_path, "qID_l.txt"), qID_l, fmt='%d')

    # if dataset == 'lotte':
    #     assert os.path.exists(index_path)
    #     n_centroid = np.load(os.path.join(index_path, "centroids.npy")).shape[0]
    #     pq_n_partition = np.load(os.path.join(index_path, "residuals.npy")).shape[1]
    #     assert pq_n_bit_per_partition == 8
    #     return {'n_centroid': n_centroid, 'pq_n_partition': pq_n_partition,
    #             "pq_n_bit_per_partition": pq_n_bit_per_partition}

    query_embeddings_path = os.path.join(index_path, "query_embeddings.npy")
    item_l_path = os.path.join(embedding_path, 'base_embedding', 'encoding0_float32.npy')
    doclens_path = os.path.join(embedding_path, 'doclens.npy')
    centroids_to_pids_path = os.path.join(index_path, "centroids_to_pids.txt")
    residuals_path = os.path.join(index_path, "residuals.npy")
    centroids_path = os.path.join(index_path, "centroids.npy")
    index_assignment_path = os.path.join(index_path, "index_assignment.npy")
    pq_centroids_path = os.path.join(index_path, "pq_centroids.npy")
    if os.path.isfile(query_embeddings_path) and os.path.isfile(doclens_path) and os.path.isfile(item_l_path) \
            and os.path.isfile(centroids_to_pids_path) and os.path.isfile(residuals_path) and os.path.isfile(centroids_path) \
            and os.path.isfile(index_assignment_path) and os.path.isfile(pq_centroids_path):
        print("exist index, skip build index")
        return {'n_centroid': n_centroid, 'pq_n_partition': pq_n_partition,
                "pq_n_bit_per_partition": pq_n_bit_per_partition}

    np.save(os.path.join(index_path, "query_embeddings.npy"),
            np.load(f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/query_embedding.npy'))

    # sample the training set
    item_l_path = os.path.join(embedding_path, 'base_embedding', 'encoding0_float32.npy')
    training_set = np.load(item_l_path)
    vec_dim = training_set.shape[1]

    sample_vecs_l, sample_item_n_vec_l = sample_vector(username=username, dataset=dataset)
    print("start faiss build PQ centroid")
    res = faiss.StandardGpuResources()  # use a single GPU
    quantizer = faiss.IndexFlatL2(vec_dim)
    index = faiss.IndexIVFPQ(quantizer, vec_dim, n_centroid, pq_n_partition, pq_n_bit_per_partition)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.verbose = True
    index.train(sample_vecs_l)
    print("end faiss build PQ centroid")

    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    assert centroids.shape[0] == n_centroid

    index_add_vector(username=username, dataset=dataset, index=index)
    index = faiss.index_gpu_to_cpu(index)

    residuals = np.zeros([index.ntotal, index.pq.M], dtype=np.uint8)
    all_indices = np.zeros([index.ntotal], dtype=np.uint64)

    centroids_to_pids = [None] * n_centroid

    item_n_vec_l = np.load(os.path.join(embedding_path, 'doclens.npy')).astype(np.uint32)
    ## total number of embeddings in your collection. Usually it can be obtained from index.ntotal
    tot_embedding = int(np.sum(np.load(os.path.join(embedding_path, 'doclens.npy'))))

    n_item = len(item_n_vec_l)
    emb2pid = np.zeros(tot_embedding, dtype=np.int64)
    offset = 0
    for i in range(n_item):
        l = item_n_vec_l[i]
        emb2pid[offset: offset + l] = i
        offset = offset + l
    doc_offsets = np.zeros(n_item, dtype=np.int64)
    for i in range(1, n_item):
        doc_offsets[i] = doc_offsets[i - 1] + item_n_vec_l[i - 1]

    # print(index.__dir__())
    # print(faiss.vector_to_array(index.getListIndices(0)))
    print(inspect_tools.get_invlist_sizes(index.invlists))
    for i in tqdm(range(index.nlist)):
        ids, codes = get_invlist(index.invlists, i)
        residuals[ids] = codes
        all_indices[ids] = i
        centroids_to_pids[i] = emb2pid[ids]

    # Write centroids to pids
    # print("centroids_to_pids", centroids_to_pids)
    with open(os.path.join(index_path, "centroids_to_pids.txt"), "w") as file:
        for centroids_list in tqdm(centroids_to_pids):
            for x in centroids_list:
                file.write(f"{x} ")
            file.write("\n")

    # Write residuals
    np.save(os.path.join(index_path, "residuals.npy"), residuals)

    # Write centroids
    np.save(os.path.join(index_path, "centroids.npy"), centroids)

    # Write index_assignments
    np.save(os.path.join(index_path, "index_assignment.npy"), all_indices)

    # Write pq_centroids
    pq_centroids = faiss.vector_to_array(index.pq.centroids)
    np.save(os.path.join(index_path, "pq_centroids.npy"), pq_centroids)

    return {'n_centroid': n_centroid, 'pq_n_partition': pq_n_partition,
            "pq_n_bit_per_partition": pq_n_bit_per_partition}


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    n_centroid = 2 ** 10
    pq_n_partition = 16  # specify the number of partitions of vec_dim
    pq_n_bit_per_partition = 8

    build_index(username=username, dataset=dataset, n_centroid=n_centroid,
                pq_n_partition=pq_n_partition, pq_n_bit_per_partition=pq_n_bit_per_partition)
