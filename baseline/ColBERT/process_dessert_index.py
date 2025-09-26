import numpy as np
import torch
import os


# needed extract file:
# centroids.npy: {dessert_index_path}/centroids.npy
# all_centroid_ids.npy: {dessert_index_path}/all_centroid_ids.npy
# doclens.npy: {embedding_path}/doclens.npy

# not process
# encodings{i}_float32.py: {embedding_path}/base_embedding/encodings{i}_float32.py
# queries.dev.small.tsv: {document_path}/queries.dev.tsv
# small_queries_embeddings.npy: {embedding_path}/query_embedding.npy

def process_dataset(username: str, dataset: str):
    plaid_index_path = f'/home/{username}/multi-vector-retrieval/Dataset/{dataset}/index/plaid/'
    dessert_index_path = f'/home/{username}/multi-vector-retrieval/Dataset/{dataset}/index/dessert'
    embedding_path = f'/home/{username}/multi-vector-retrieval/Dataset/{dataset}/embedding'

    os.mkdir(dessert_index_path)
    centroids_load_path = os.path.join(plaid_index_path, 'centroids.pt')
    centroids = torch.load(centroids_load_path).cpu()
    print(centroids.shape)
    centroids_save_path = os.path.join(dessert_index_path, 'centroids.npy')
    np.save(centroids_save_path, centroids.cpu().numpy().astype('float32'))

    total_centroid_id = np.array([])
    total_doclen = np.array([])
    for i in range(0, 354, 1):
        centroid_id_l = torch.load(os.path.join(plaid_index_path, f'{i}.codes.pt'))
        # print(len(centroid_id_l))
        total_centroid_id = np.concatenate([total_centroid_id, centroid_id_l])

        doclens_path = os.path.join(embedding_path, 'base_embedding', f'doclens{i}.npy')
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
    username = 'zhengbian'
    dataset = 'msmacro'
