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


class Searcher4Encode:
    def __init__(self, checkpoint=None, config=None):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_

        self.checkpoint = checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, initial_config)

        self.configure(checkpoint=self.checkpoint)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            print("searcher4encode, use_gpu", use_gpu)
            self.checkpoint = self.checkpoint.cuda()
        self.checkpoint = self.checkpoint.cuda()

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

    def save_query_embedding(self, queries: TextQueries, save_path: str):
        start_time = time.time_ns()
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_)
        # @bianzheng

        numpy_32 = Q.cpu().numpy().astype("float32")
        end_time = time.time_ns()
        total_encode_time_ms = (end_time - start_time) * 1e-6
        n_encode_query = len(queries)
        np.save(save_path, numpy_32)
        print('save embeddings')
        return total_encode_time_ms, n_encode_query
