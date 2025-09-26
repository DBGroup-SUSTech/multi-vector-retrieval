import numpy as np
import torch
import os
import tqdm
import sys
import random
import faiss

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.data import util
from script.evaluation.technique import vq


def compress_into_codes(embs: np.ndarray, centroid_l_cuda: torch.Tensor):
    codes = []

    # print(centroid_l)
    bsize = (1 << 29) // centroid_l_cuda.shape[0]
    embs = torch.from_numpy(embs)
    for batch in embs.split(bsize):
        indices = (centroid_l_cuda @ batch.T.to('cuda').float()).max(dim=0).indices
        indices = indices.to('cpu')
        codes.append(indices)

    return torch.cat(codes)


def compute_assignment(username: str, dataset: str,
                       centroid_l: np.ndarray,
                       module: object,
                       cutoff_l: np.ndarray, weight_l: np.ndarray, n_bit: int,
                       vec_dim: int):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')

    doclens = np.load(os.path.join(embedding_dir, 'doclens.npy'))
    n_vec = int(np.sum(doclens))
    vq_code_l = torch.empty((n_vec), dtype=torch.int32)

    sq_ins = module.CompressResidualCode(centroid_l=centroid_l, cutoff_l=cutoff_l, weight_l=weight_l, n_bit=n_bit)
    n_val_per_vec = sq_ins.n_val_per_vec

    residual_code_l = np.empty(shape=(n_vec * n_val_per_vec), dtype=np.uint8)

    print("compute assignment of centroid vector")
    n_chunk = util.get_n_chunk(base_embedding_dir)
    vq_code_offset = 0
    residual_code_offset = 0
    centroid_l_cuda = torch.Tensor(centroid_l).to('cuda')
    for chunkID in tqdm.trange(n_chunk):
        itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        n_vec_chunk = int(np.sum(itemlen_l_chunk))
        item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))

        vq_code_l_chunk = compress_into_codes(embs=item_vecs_l_chunk, centroid_l_cuda=centroid_l_cuda)

        residual_code_l_chunk, = sq_ins.compute_residual_code(vec_l=item_vecs_l_chunk,
                                                            code_l=vq_code_l_chunk.numpy())
        residual_code_l_chunk = np.array(residual_code_l_chunk, dtype=np.uint8)

        vq_code_l[vq_code_offset:vq_code_offset + n_vec_chunk] = vq_code_l_chunk
        vq_code_offset += n_vec_chunk

        residual_code_l[residual_code_offset:residual_code_offset + n_vec_chunk * n_val_per_vec] = residual_code_l_chunk
        residual_code_offset += n_vec_chunk * n_val_per_vec

    vq_code_l = np.array(vq_code_l, dtype=np.uint32)
    # residual_code_l = np.array(residual_code_l, dtype=np.uint8)
    assert vq_code_l.shape[0] * n_val_per_vec == residual_code_l.shape[0]
    assert residual_code_l.ndim == 1
    return vq_code_l, residual_code_l


def faiss_kmeans(sample_vecs_l: np.ndarray, n_centroid: int):
    print("build kmeans")
    vec_dim = sample_vecs_l.shape[1]
    kmeans = faiss.Kmeans(vec_dim, n_centroid, niter=20, gpu=True, verbose=True, seed=123)
    kmeans.train(sample_vecs_l)

    centroids = torch.from_numpy(kmeans.centroids)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    centroids = centroids.float()
    centroids = centroids.numpy()

    code_l = vq.compress_into_codes(embs=sample_vecs_l, centroid_l=centroids)
    return centroids, code_l


def vq_sq_ivf(username: str, dataset: str, module: object, n_centroid: int, n_bit: int):
    sample_vecs_l, sample_item_n_vec_l = vq.sample_vector(username=username, dataset=dataset)

    centroid_l, sample_code_l = faiss_kmeans(sample_vecs_l=sample_vecs_l, n_centroid=n_centroid)
    vec_dim = centroid_l.shape[1]

    cutoff_l, weight_l = module.compute_quantized_scalar(item_vec_l=sample_vecs_l, centroid_l=centroid_l,
                                                         code_l=sample_code_l,
                                                         n_bit=n_bit)
    vq_code_l, residual_code_l = compute_assignment(username=username, dataset=dataset,
                                                    centroid_l=centroid_l, module=module,
                                                    cutoff_l=cutoff_l, weight_l=weight_l, n_bit=n_bit,
                                                    vec_dim=vec_dim)
    centroid_l = np.array(centroid_l, dtype=np.float32)
    vq_code_l = np.array(vq_code_l, dtype=np.uint32)
    weight_l = np.array(weight_l, dtype=np.float32)
    residual_code_l = np.array(residual_code_l, dtype=np.uint8)

    return centroid_l, vq_code_l, weight_l, residual_code_l
