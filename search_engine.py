import faiss
import numpy as np
import pickle
from scipy.spatial import distance
from functools import partial

import time
import os
import glob
import glob2
import json

class SearchEngine:
    def search_preprocess(self, data, is_train=False):
        if self.faiss_params['preprocess_opt'] == 'norm':
            return np.float32((data + 1e-6) / (np.linalg.norm(data + 1e-6, keepdims=True, axis=-1) + 1e-30))
        elif self.faiss_params['preprocess_opt'] == 'false':
            return np.float32(data)
        elif self.faiss_params['preprocess_opt'] == 'covar':
            if is_train:
                cov = np.np.cov(data)
                L = np.linalg.cholesky(cov)
                self.faiss_params['mahalanobis_transform'] = np.linalg.inv(L)
            return np.float32(np.dot(data, self.faiss_params['mahalanobis_transform'].T))

    def translate_distance_to_pdist(self, dist_name):
        if dist_name == 'inner':
            return lambda x, y: x.dot(y)
        return dist_name

    def create_faiss(self):
        quantiser = faiss.IndexFlatL2(self.num_dimensions)
        self.faiss_params = {}
        if self.similarity_metric == 'cosine':
            self.faiss_params['preprocess_opt'] = 'norm'
            self.faiss_params['metric'] = faiss.METRIC_L2
        
        elif self.similarity_metric == 'inner':
            self.faiss_params['preprocess_opt'] = 'false'
            self.faiss_params['metric'] = faiss.METRIC_INNER_PRODUCT
        
        elif self.similarity_metric == 'euclidean':
            self.faiss_params['preprocess_opt'] = 'false'
            self.faiss_params['metric'] = faiss.METRIC_L2

        elif self.similarity_metric == 'mahalanobis':
            self.faiss_params['preprocess_opt'] = 'covar'
            self.faiss_params['metric'] = faiss.METRIC_L2

        self.faiss_index = faiss.IndexIVFFlat(quantiser, self.num_dimensions, self.num_centroids, self.faiss_params['metric'])
        vectors = self.search_preprocess(np.stack(self.doc_embeddings_vectors, axis=0), is_train=True)
        self.faiss_index.train(vectors)
        self.faiss_index.add(vectors)

        # In the case of faiss we can remove this documents lists
        del self.doc_embeddings_vectors

    def __init__(self, method, doc_embeddings, use_faiss=False, similarity_metric='cosine'):
        self.method = method
        self.doc_embeddings_vectors = []
        self.doc_embeddings_hash_id = []
        for doc in doc_embeddings:
            if doc['vector'] is None or np.prod(doc['vector'].shape) == 0:
                continue

            self.doc_embeddings_vectors.append(doc['vector'])
            self.doc_embeddings_hash_id.append(doc['hash_id'])

        self.use_faiss = use_faiss
        self.similarity_metric = similarity_metric

        if self.use_faiss:
            print('FAISS INDEXING...', end=' ')
            self.create_faiss()
            print('DONE')

    def get_similar_docs_than(self, text, k=10):
        vector = self.method.compute_mean_vector_from_text(text)
        
        if self.use_faiss:
            vector = self.search_preprocess(vector)
            vector = np.expand_dims(vector, axis=0)

            # Find similars
            _, indices = self.faiss_index.search(vector, k)
            docs_hash_id = [self.doc_embeddings_hash_id[idx] for idx in indices[0]]
        else:
            distances = distance.cdist(np.expand_dims(vector, axis=0), np.stack(self.doc_embeddings_vectors, axis=0), self.translate_distance_to_pdist(self.similarity_metric))
            indices = distances.argsort(axis=-1)[:, :k]
            docs_hash_id = [self.doc_embeddings_hash_id[idx] for idx in indices[0]]
        return docs_hash_id