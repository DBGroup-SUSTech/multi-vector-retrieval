import numpy as np
import os
from typing import Dict
import sys
import time
import json
import struct
import faiss

import torch
import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import evaluation_pipeline
from script.data import util


def approximate_solution_retrieval(index: object, retrieval_config: dict, query_l: np.ndarray, topk: int):
    n_candidate = retrieval_config['n_candidate']
    print(f"retrieval: n_candidate {n_candidate}")

    (est_dist_l, est_id_l, retrieval_time_l,
     transform_time_l, ip_time_l, decode_time_l, refine_time_l,
     n_search_candidate_l) = index.search(
        query_l=query_l, topk=topk,
        n_candidate=n_candidate)

    search_time_m = {
        "retrieval_time_single_query_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5) * 1e3),
        "retrieval_time_single_query_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50) * 1e3),
        "retrieval_time_single_query_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95) * 1e3),
        "retrieval_time_single_query_average(ms)": '{:.3f}'.format(1.0 * np.average(retrieval_time_l) * 1e3),

        "transform_time_average(ms)": '{:.3f}'.format(1.0 * np.average(transform_time_l) * 1e3),
        "ip_time_average(ms)": '{:.3f}'.format(1.0 * np.average(ip_time_l) * 1e3),
        "decode_time_average(ms)": '{:.3f}'.format(1.0 * np.average(decode_time_l) * 1e3),
        "refine_time_average(ms)": '{:.3f}'.format(1.0 * np.average(refine_time_l) * 1e3),

        "n_search_candidate_average": '{:.3f}'.format(1.0 * np.average(n_search_candidate_l)),
    }
    retrieval_suffix = f'n_candidate_{n_candidate}'

    retrieval_time_ms_l = np.around(retrieval_time_l * 1e3, 3)
    return est_dist_l, est_id_l, retrieval_suffix, search_time_m, retrieval_time_ms_l


def compress_into_codes(embs: np.ndarray, centroid_l: np.ndarray):
    codes = []

    centroid_l = torch.Tensor(centroid_l).to('cuda')
    # print(centroid_l)
    bsize = (1 << 29) // centroid_l.shape[0]
    embs = torch.from_numpy(embs)
    for batch in embs.split(bsize):
        indices = (centroid_l.to('cuda') @ batch.T.to('cuda').float()).max(dim=0).indices
        indices = indices.to('cpu')
        codes.append(indices)

    return torch.cat(codes).numpy().astype(np.uint32)


def build_index(username: str, dataset: str, index: object, module: object,
                vec_dim: int, k_sim: int, d_proj: int, r_reps: int,
                n_centroid_per_subspace: int, dim_per_subspace: int):
    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}/'
    base_embedding_dir = os.path.join(embedding_dir, 'base_embedding')

    # generate the random gaussian vector
    partition_vec_l_np = np.random.normal(loc=0, scale=1.0, size=(r_reps, k_sim, vec_dim))
    partition_vec_l = torch.Tensor(partition_vec_l_np)
    # Generate a {-1, 1} random matrix
    random_matrix_l = torch.randint(0, 2, (r_reps, d_proj, vec_dim), dtype=torch.int32)
    random_matrix_l = random_matrix_l * 2 - 1
    random_matrix_l = random_matrix_l.float()
    random_matrix_l = random_matrix_l
    random_matrix_l_np = random_matrix_l.numpy().astype(np.float32)

    index.add_projection(partition_vec_l=partition_vec_l_np, random_matrix_l=random_matrix_l_np)

    item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
    n_vecs = np.sum(item_n_vec_l)
    n_item = item_n_vec_l.shape[0]
    ip_vector_l = np.zeros(shape=(n_item, r_reps * (2 ** k_sim) * d_proj), dtype=np.float32)
    vec_l = np.zeros(shape=(n_vecs, vec_dim), dtype=np.float32)

    partition_vec_l_cuda = partition_vec_l.cuda()
    random_matrix_l_cuda = random_matrix_l.cuda()

    print("compute assignment of centroid vector")
    n_chunk = util.get_n_chunk(base_embedding_dir)
    itemID_offset = 0
    n_vec_offset = 0
    for chunkID in tqdm.trange(n_chunk):
        itemlen_l_chunk = np.load(os.path.join(base_embedding_dir, f'doclens{chunkID}.npy'))
        item_vecs_l_chunk = np.load(os.path.join(base_embedding_dir, f'encoding{chunkID}_float32.npy'))
        n_item_chunk = itemlen_l_chunk.shape[0]
        n_vec_chunk = item_vecs_l_chunk.shape[0]

        item_vecs_l_chunk_cuda = torch.from_numpy(item_vecs_l_chunk).cuda()
        # (batch_n_vec, vec_dim) * (r_reps, k_sim, vec_dim) -> (batch_n_vec, r_reps, k_sim)
        partition_bit_l_cuda = torch.einsum('ij,klj->ikl', item_vecs_l_chunk_cuda, partition_vec_l_cuda)
        partition_bit_l_cuda = torch.sign(partition_bit_l_cuda).int()
        print(partition_bit_l_cuda.dtype)
        partition_bit_l_cuda[partition_bit_l_cuda == -1] = 0
        print(partition_bit_l_cuda)
        vec_cluster_bit_l = partition_bit_l_cuda.cpu().numpy()
        vec_cluster_bit_l = vec_cluster_bit_l.astype(np.uint8)
        assert vec_cluster_bit_l.shape == (n_vec_chunk, r_reps, k_sim)

        # output should be (n_item_chunk, r_reps, 2 ** k_sim, vec_dim)
        print("before assign_cluster_vector")
        cluster_vector_l = module.assign_cluster_vector(vec_cluster_bit_l=vec_cluster_bit_l,
                                                        item_vecs_l_chunk=item_vecs_l_chunk,
                                                        item_n_vec_l_chunk=itemlen_l_chunk,
                                                        batch_n_vec=n_vec_chunk, batch_n_item=n_item_chunk,
                                                        r_reps=r_reps, k_sim=k_sim, vec_dim=vec_dim)
        print("after assign_cluster_vector")

        cluster_vector_l_cuda = torch.from_numpy(cluster_vector_l).cuda()
        print(cluster_vector_l_cuda.dtype, random_matrix_l_cuda.dtype)
        # (n_item_chunk, r_reps, 2 ** k_sim, vec_dim) * (r_reps, d_proj, vec_dim) -> (n_item_chunk, r_reps, 2 ** k_sim, d_proj)
        ip_vector_l_cuda = torch.einsum('iljk,lmk->iljm', cluster_vector_l_cuda, random_matrix_l_cuda)

        ip_vector_l_chunk = ip_vector_l_cuda.cpu().numpy().reshape(n_item_chunk, -1).astype(np.float32)
        assert ip_vector_l_chunk.shape == (n_item_chunk, r_reps * (2 ** k_sim) * d_proj)
        ip_vector_l[itemID_offset:itemID_offset + n_item_chunk] = ip_vector_l_chunk
        itemID_offset += n_item_chunk
        vec_l[n_vec_offset:n_vec_offset + n_vec_chunk] = item_vecs_l_chunk
        n_vec_offset += n_vec_chunk

    index.build_graph_index(ip_vector_l=ip_vector_l)
    index.add_item_vector_l(vec_l=vec_l)

    # perform product quantization
    n_dim = (2 ** k_sim) * r_reps * d_proj
    n_subspace = n_dim // dim_per_subspace + (0 if n_dim % dim_per_subspace == 0 else 1)

    sub_centroid_l_l = np.zeros(shape=(n_subspace, n_centroid_per_subspace, dim_per_subspace), dtype=np.float32)
    sub_code_l_l = np.zeros(shape=(n_subspace, n_item), dtype=np.uint32)

    for subspaceID in range(n_subspace):
        start_dim = subspaceID * dim_per_subspace
        end_dim = min(start_dim + dim_per_subspace, n_dim)
        n_dim_subspace = end_dim - start_dim
        sub_ip_vector_l = ip_vector_l[:, start_dim:end_dim]
        if n_dim_subspace < dim_per_subspace:
            sub_ip_vector_l = np.pad(sub_ip_vector_l, pad_width=((0, 0), (0, dim_per_subspace - n_dim_subspace)),
                                     mode='constant', constant_values=0)
        assert sub_ip_vector_l.shape == (n_item, dim_per_subspace)

        kmeans = faiss.Kmeans(dim_per_subspace, n_centroid_per_subspace, niter=20, gpu=True, verbose=True, seed=123)
        if dataset == 'msmacro':
            sample_sub_ip_vector_l = sub_ip_vector_l[:300_000, :]
        else:
            sample_sub_ip_vector_l = sub_ip_vector_l[:1_000_000, :]
        kmeans.train(sample_sub_ip_vector_l)

        sub_centroid_l = torch.from_numpy(kmeans.centroids)
        # sub_centroid_l = torch.nn.functional.normalize(sub_centroid_l, dim=-1)
        sub_centroid_l = sub_centroid_l.float().numpy()  # n_centroid_per_subspace * n_dim_subspace
        assert sub_centroid_l.shape == (n_centroid_per_subspace, dim_per_subspace)

        _, sub_code_l = kmeans.assign(sub_ip_vector_l)
        # sub_code_l = compress_into_codes(embs=sub_ip_vector_l, centroid_l=sub_centroid_l)  # n_vec
        assert sub_code_l.shape == (n_item,)

        sub_centroid_l_l[subspaceID, :] = sub_centroid_l
        sub_code_l_l[subspaceID, :] = sub_code_l

    index.add_pq_code_l(sub_centroid_l_l=sub_centroid_l_l, sub_code_l_l=sub_code_l_l)


def approximate_solution_build_index(username: str, dataset: str,
                                     constructor_insert_item: dict, module: object,
                                     module_name: str,
                                     build_index_config: dict, build_index_suffix: str):
    index = module.DocRetrieval(**constructor_insert_item)
    print(f"start insert item")
    start_time = time.time()

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
    k_sim = build_index_config['k_sim']
    d_proj = build_index_config['d_proj']
    r_reps = build_index_config['r_reps']
    n_centroid_per_subspace = build_index_config['n_centroid_per_subspace']
    dim_per_subspace = build_index_config['dim_per_subspace']

    build_index(username=username, dataset=dataset, index=index, module=module,
                vec_dim=vec_dim, k_sim=k_sim, d_proj=d_proj, r_reps=r_reps,
                n_centroid_per_subspace=n_centroid_per_subspace, dim_per_subspace=dim_per_subspace)

    end_time = time.time()
    build_index_time_sec = end_time - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")

    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval/Result/performance'
    build_index_performance_filename = os.path.join(result_performance_path,
                                                    f'{dataset}-build_index-{module_name}-{build_index_suffix}.json')
    with open(build_index_performance_filename, 'w') as f:
        json.dump({'build_index_time (s)': build_index_time_sec, }, f)

    return index


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for n_candidate in grid_search_para['n_candidate']:
        parameter_l.append(
            {"n_candidate": n_candidate})
    return parameter_l


if __name__ == '__main__':
    # default value {'n_centroid_f': lambda x: int(2 ** np.floor(np.log2(16 * np.sqrt(x))))},
    config_l = {
        'dbg': {
            'username': 'username1',
            'dataset_l': ['lotte'],
            'topk_l': [10, 100],
            'is_debug': False,
            'build_index_parameter_l': [
                {'k_sim': 3, 'd_proj': 8, 'r_reps': 20,
                 'R': 200, 'L': 600, 'alpha': 1.2,
                 'n_centroid_per_subspace': 256, 'dim_per_subspace': 16},
            ],
            'retrieval_parameter_l': [
                {'n_candidate': 8},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'n_candidate': [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
                },
                100: {
                    'n_candidate': [100, 500, 1000, 2000, 4000, 8000, 16000]
                },
                1000: {
                    'n_candidate': [8]
                }
            }
        },
        'local': {
            'username': 'username2',
            # 'dataset_l': ['fake-normal', 'lotte-500-gnd'],
            'dataset_l': ['lotte-500-gnd'],
            'topk_l': [10],
            'is_debug': True,
            'build_index_parameter_l': [
                {'k_sim': 5, 'd_proj': 16, 'r_reps': 20,
                 'R': 20, 'L': 50, 'alpha': 1.2,
                 'n_centroid_per_subspace': 256, 'dim_per_subspace': 16},
            ],
            'retrieval_parameter_l': [
                {'n_candidate': 20},
                {'n_candidate': 30},
                {'n_candidate': 50},
            ],
            'grid_search': False,
            'grid_search_para': {
                10: {
                    'n_candidate': [8]
                },
                50: {
                    'n_candidate': [8]
                },
            }
        }
    }
    host_name = 'local'
    config = config_l[host_name]

    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']
    is_debug = config['is_debug']
    build_index_parameter_l = config['build_index_parameter_l']

    module_name = 'MUVERA'
    move_path = 'evaluation'

    util.compile_file(username=username, module_name=module_name, is_debug=is_debug, move_path=move_path)
    for dataset in dataset_l:
        for build_index_config in build_index_parameter_l:
            embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
            vec_dim = np.load(os.path.join(embedding_dir, 'base_embedding', f'encoding0_float32.npy')).shape[1]
            n_item = np.load(os.path.join(embedding_dir, f'doclens.npy')).shape[0]
            item_n_vec_l = np.load(os.path.join(embedding_dir, f'doclens.npy')).astype(np.uint32)
            n_vecs = np.sum(item_n_vec_l)

            k_sim = build_index_config['k_sim']
            d_proj = build_index_config['d_proj']
            r_reps = build_index_config['r_reps']
            R = build_index_config['R']
            L = build_index_config['L']
            alpha = build_index_config['alpha']
            n_centroid_per_subspace = build_index_config['n_centroid_per_subspace']
            dim_per_subspace = build_index_config['dim_per_subspace']

            constructor_insert_item = {'item_n_vec_l': item_n_vec_l.tolist(),
                                       'n_item': n_item, 'vec_dim': vec_dim,
                                       'k_sim': k_sim, 'd_proj': d_proj, 'r_reps': r_reps,
                                       'R': R, 'L': L, 'alpha': alpha,
                                       'n_centroid_per_subspace': n_centroid_per_subspace,
                                       'dim_per_subspace': dim_per_subspace}
            build_index_suffix = f'k_sim_{k_sim}-d_proj_{d_proj}-r_reps_{r_reps}'

            module = evaluation_pipeline.approximate_solution_compile_load(
                username=username, dataset=dataset,
                module_name=module_name, compile_file=False,
                is_debug=is_debug, move_path=move_path)

            index = approximate_solution_build_index(
                username=username, dataset=dataset,
                constructor_insert_item=constructor_insert_item,
                module=module, module_name=module_name,
                build_index_config=build_index_config, build_index_suffix=build_index_suffix)

            for topk in topk_l:
                grid_search = config['grid_search']
                if grid_search:
                    retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
                else:
                    retrieval_parameter_l = config['retrieval_parameter_l']

                evaluation_pipeline.approximate_solution_retrieval_outter(
                    username=username, dataset=dataset,
                    index=index, module_name=module_name,
                    build_index_suffix=build_index_suffix,
                    constructor_insert_item=constructor_insert_item,
                    topk=topk,
                    retrieval_parameter_l=retrieval_parameter_l,
                    retrieval_f=approximate_solution_retrieval
                )
