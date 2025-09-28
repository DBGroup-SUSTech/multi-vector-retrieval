import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats
import numpy as np
import time

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        self.checkpoint = self.checkpoint.cuda()
        self.ranker = IndexScorer(self.index, False)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries):
        queries = text if type(text) is list else [text]
        # bsize = 128 if len(queries) > 128 else None
        bsize = 1 if len(queries) > 1 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True)

        return Q

    def search(self, text: str, k=10, filter_fn=None):
        Q = self.encode(text)
        self.config.total_visible_gpus = 0
        return self.dense_search(Q, k, filter_fn=filter_fn)

    def search_all_batch(self, queries: TextQueries, k=10, filter_fn=None):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        start = time.time()
        Q = self.encode(queries_)
        encode_time = time.time() - start
        # # @username2
        # import numpy as np
        # numpy_32 = Q.numpy().astype("float32")
        # np.save('./embedding_vector/query_embedding.npy', numpy_32)
        # print('save embeddings')
        # # print(Q)
        ranking, search_time_l = self._search_all_Q_batch(queries, Q, k, filter_fn=filter_fn)
        return ranking, encode_time, search_time_l

    def _search_all_Q_batch(self, queries, Q, k, filter_fn=None):
        # all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx + 1], k, filter_fn=filter_fn)))
        #                    for query_idx in tqdm(range(Q.size(0)))]

        self.config.total_visible_gpus = 0

        all_scored_pids = []
        time_l = []  # unit: ms
        for query_idx in tqdm(range(Q.size(0))):
            time_start = time.time_ns()
            pids, ranks, scores, ivf_time_ms, filter_time_ms, refine_time_ms, n_refine_ivf, n_refine_filter, n_vec_score_refine = self.dense_search(
                Q[query_idx:query_idx + 1], k,
                filter_fn=filter_fn)
            all_scored_pids.append(list(zip(pids, ranks, scores)))
            time_end = time.time_ns()
            time_l.append((time_end - time_start) * 1e-6)

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data,
                       provenance=provenance), time_l,

    def search_all_single(self, queries: TextQueries, k=10, filter_fn=None):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())
        qid_l = queries.keys()

        ranking_l = []
        encode_time_l = []
        search_time_l = []
        # print("device in encoding phase: ", Q.device, ", search phase: ",
        #       'gpu' if self.config.total_visible_gpus > 0 else 'cpu')

        for qid, query in tqdm(zip(qid_l, queries_)):
            start = time.time_ns()
            Q = self.encode([query])
            end = time.time_ns()
            encode_time = (end - start) * 1e-6
            # # @username2
            # import numpy as np
            # numpy_32 = Q.numpy().astype("float32")
            # np.save('./embedding_vector/query_embedding.npy', numpy_32)
            # print('save embeddings')
            # # print(Q)

            rank_res, time_l = self._search_all_Q_single(
                queries, qid, Q, k, filter_fn=filter_fn)

            encode_time_l.append(encode_time)
            search_time_l.append(time_l[0])
            ranking_l.append(rank_res)
        return ranking_l, encode_time_l, search_time_l

    def _search_all_Q_single(self, queries, qid, Q, k, filter_fn=None):
        # all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx + 1], k, filter_fn=filter_fn)))
        #                    for query_idx in tqdm(range(Q.size(0)))]

        self.config.total_visible_gpus = 0

        all_scored_pids = []
        time_l = []  # unit: ms
        for query_idx in range(Q.size(0)):
            time_start = time.time_ns()
            pids, ranks, scores, ivf_time_ms, filter_time_ms, refine_time_ms, n_refine_ivf, n_refine_filter, n_vec_score_refine = self.dense_search(
                Q[query_idx:query_idx + 1], k,
                filter_fn=filter_fn)
            all_scored_pids.append(list(zip(pids, ranks, scores)))
            time_end = time.time_ns()
            time_l.append((time_end - time_start) * 1e-6)

        data = {qid: val for qid, val in zip([qid], all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data,
                       provenance=provenance), time_l

    def search_all_embedding(self, query_embd_filename, k=10, filter_fn=None):
        # queries = Queries.cast(queries)
        # queries_ = list(queries.values())
        #
        # Q = self.encode(queries_)

        # @username2
        import numpy as np
        query_emb = np.load(query_embd_filename)
        # query_len = self.config.query_maxlen
        qid_l = np.arange(int(len(query_emb)))
        # assert query_emb.shape[1] == query_len
        query_emb = torch.Tensor(query_emb)
        # print(Q)

        return self._search_all_Q_embedding(query_embd_filename, qid_l, query_emb, k, filter_fn=filter_fn)

    def search_all_embedding_by_vector(self, query_emb: np.ndarray, query_embd_filename: str,
                                       qid_l: list, k=10, filter_fn=None):
        # queries = Queries.cast(queries)
        # queries_ = list(queries.values())
        #
        # Q = self.encode(queries_)

        # @username2
        # query_len = self.config.query_maxlen
        # assert query_emb.shape[1] == query_len
        query_emb = torch.Tensor(query_emb)
        # print(Q)

        return self._search_all_Q_embedding(query_embd_filename, qid_l, query_emb, k, filter_fn=filter_fn)

    def _search_all_Q_embedding(self, query_embd_filename, qid_l, Q, k, filter_fn=None):
        # all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx + 1], k, filter_fn=filter_fn)))
        #                    for query_idx in tqdm(range(Q.size(0)))]

        self.config.total_visible_gpus = 0

        all_scored_pids = []
        time_l = []  # unit: ms
        time_ivf_l = []  # unit: ms
        time_filter_l = []  # unit: ms
        time_refine_l = []  # unit: ms
        n_refine_ivf_l = []
        n_refine_filter_l = []
        n_vec_score_refine_l = []
        for query_idx in tqdm(range(Q.size(0))):
            time_start = time.time_ns()
            pids, ranks, scores, ivf_time_ms, filter_time_ms, refine_time_ms, n_refine_ivf, n_refine_filter, n_vec_score_refine = self.dense_search(
                Q[query_idx:query_idx + 1], k,
                filter_fn=filter_fn)
            all_scored_pids.append(list(zip(pids, ranks, scores)))
            time_end = time.time_ns()
            time_l.append((time_end - time_start) * 1e-6)
            time_ivf_l.append(ivf_time_ms)
            time_filter_l.append(filter_time_ms)
            time_refine_l.append(refine_time_ms)
            n_refine_ivf_l.append(n_refine_ivf)
            n_refine_filter_l.append(n_refine_filter)
            n_vec_score_refine_l.append(n_vec_score_refine)

        # # print(all_scored_pids)
        # print([len(_) for _ in all_scored_pids], k)
        # print(zip(*self.dense_search(Q[0:0 + 1], k, filter_fn=filter_fn)))

        data = {qid: val for qid, val in zip(qid_l, all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = query_embd_filename
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data,
                       provenance=provenance), time_l, time_ivf_l, time_filter_l, time_refine_l, n_refine_ivf_l, n_refine_filter_l, n_vec_score_refine_l

    def save_query_embedding(self, queries: TextQueries, save_path: str):
        start_time = time.time_ns()
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_)
        # @username2

        numpy_32 = Q.cpu().numpy().astype("float32")
        end_time = time.time_ns()
        total_encode_time_ms = (end_time - start_time) * 1e-6
        n_encode_query = len(queries)
        np.save(save_path, numpy_32)
        print('save embeddings')
        return total_encode_time_ms, n_encode_query

    def save_query_embedding_cpu(self, queries: TextQueries, save_path: str):
        torch.set_num_threads(os.cpu_count())
        # print("--------------------------")
        # print(f"max_n_thread {os.cpu_count()}")
        # print(f"n_thread {torch.get_num_threads()}")
        # print("--------------------------")

        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        queries = queries_ if type(queries_) is list else [queries_]
        # bsize = 128 if len(queries) > 128 else None
        bsize = 1 if len(queries) > 1 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        checkpoint_cpu = self.checkpoint.to('cpu')
        start_time = time.time()
        Q = checkpoint_cpu.queryFromText(queries, bsize=bsize, to_cpu=True)
        # @username2
        end_time = time.time()

        numpy_32 = Q.cpu().numpy().astype("float32")

        total_encode_time_ms = (end_time - start_time) * 1e3
        n_encode_query = len(queries)
        np.save(save_path, numpy_32)
        print('save embeddings')
        return total_encode_time_ms, n_encode_query

    def get_query_embedding(self, queries: TextQueries):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())
        Q = self.encode(queries_)

        numpy_32 = Q.cpu().numpy().astype("float32")
        return numpy_32

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores, ivf_time_ms, filter_time_ms, refine_time_ms, n_refine_ivf, n_refine_filter, n_vec_score_refine = self.ranker.rank(
            self.config, Q,
            filter_fn=filter_fn)

        return pids[:k], list(range(1, k + 1)), scores[
                                                :k], ivf_time_ms, filter_time_ms, refine_time_ms, n_refine_ivf, n_refine_filter, n_vec_score_refine
