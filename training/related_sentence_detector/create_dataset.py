import sys
import os
# isn't there a better way to do this?
sys.path.append(os.path.abspath('../../'))

# from methods import *
from database_core import Database

import json
import torch
import flair
import pickle

import scispacy
import spacy
spacy.require_gpu()
from tqdm import tqdm

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings

import concurrent.futures as cf
import torch.multiprocessing as mp
mp.set_start_method("spawn")
from torch.multiprocessing import Pool


curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, "data")
tmp_path = os.path.join(data_path, "tmp_data")
os.makedirs(os.path.join(tmp_path, "doc"), exist_ok=True)
os.makedirs(os.path.join(tmp_path, "sentences"), exist_ok=True)

nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

flair.device = torch.device('cuda:1')
flair.embedding_storage_mode = None
flair_emb = DocumentPoolEmbeddings([
        FlairEmbeddings('en-forward-fast'), 
        FlairEmbeddings('en-backward-fast')
    ],
    pooling='mean',
)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

poss_sections = {
    '#gender': ['gender', 'sex', 'gentleman'],
    '#male': ['male', 'man', 'men'],
    '#female': ['female', 'woman', 'women'],
}

# substitute term by its embedding
for list_candidates in tqdm(poss_sections.values(), desc='Embedding search terms'):
    for i in range(len(list_candidates)):
        sentence = Sentence(list_candidates[i].lower())
        flair_emb.embed(sentence)
        list_candidates[i] = sentence.embedding
        sentence.clear_embeddings()

dataset = []
print('Retrieving documents from database...')
documents = Database.list_raw_documents()

def create_emb(tmp_sentences_path, doc):
    batch_size = 60

    dataset_section = []
    for section_title, section_text in doc['raw']['sections'].items():
        text = section_text.strip()
        # tokenize each sentence
        # skip language detection, embedding wont match anyway
        nlp_doc = nlp(text)

        texts = []
        flair_sentences = []
        for sentence in nlp_doc.sents:
            flair_sent = Sentence(sentence.text)
            flair_sentences.append(flair_sent)
            texts.append(sentence.text)

        with open(tmp_sentences_path, 'w') as f:
            json.dump(texts, f)

        for i in range(0, len(flair_sentences), batch_size):
            flair_emb.embed(flair_sentences[i:i+batch_size])

        for flair_sent, text in zip(flair_sentences, texts):
            for token in flair_sent.tokens:
                mean_vector = token.embedding
                max_value = -2
                max_part = None
                for possible_part, candidates in poss_sections.items():
                    for candidate in candidates:
                        score = cos(mean_vector, candidate)
                        if score > 0.9: # consideramos valido
                            if max_value < score:
                                max_value = score
                                max_part = possible_part

            flair_sent.clear_embeddings()

            if max_part is not None:
                dataset_section.append({
                    'hash_id': doc['hash_id'],
                    'title': section_title,
                    'text': text,
                    'match': max_part
                })

    return dataset_section

# xx = Pool(2)
# with xx as executor:
#     dataset_raw = tqdm(executor.map(create_emb, documents), desc='Embedding documents')
#     dataset = [data for dataset_section in dataset_raw for data in dataset_section]

dataset = []
for doc in tqdm(documents, desc='Embedding documents'):
    tmp_doc_path = os.path.join(tmp_path, "doc", doc['hash_id'] + ".json")
    tmp_sentences_path = os.path.join(tmp_path, "sentences", doc['hash_id'] + ".json")
    if not os.path.exists(tmp_doc_path):
        doc_matchs = create_emb(tmp_sentences_path, doc)
        with open(tmp_doc_path, 'w') as f:
            json.dump(doc_matchs, f)
    else:  
        try:
            with open(tmp_doc_path, 'r') as f:
                doc_matchs = json.load(f)
        except:
            doc_matchs = create_emb(tmp_sentences_path, doc)
            with open(tmp_doc_path, 'w') as f:
                json.dump(doc_matchs, f)
    
    dataset += doc_matchs

# for i, doc in tqdm(enumerate(documents), desc='Embedding documents'):
#     for section_title, section_text in doc['raw']['sections'].items():
#         text = section_text.strip()
#         # tokenize each sentence
#         # skip language detection, embedding wont match anyway
#         nlp_doc = nlp(text)
#         for sentence in nlp_doc.sents:
#             flair_sent = Sentence(sentence.text)
#             flair_emb.embed(flair_sent)

#             for token in flair_sent.tokens:
#                 mean_vector = token.embedding
#                 max_value = -2
#                 max_part = None
#                 for possible_part, candidates in poss_sections.items():
#                     for candidate in candidates:
#                         score = cos(mean_vector, candidate)
#                         if score > 0.9: # consideramos valido
#                             if max_value < score:
#                                 max_value = score
#                                 max_part = possible_part

#             flair_sent.clear_embeddings()

#             if max_part is not None:
#                 dataset.append({
#                     'hash_id': doc['hash_id'],
#                     'title': section_title,
#                     'text': sentence.text,
#                     'match': max_part
#                 })
    
#     print(i, len(documents))

os.makedirs(os.path.join(data_path), exist_ok=True)
with open(os.path.join(data_path, 'classification_dataset.pickle'), 'wb') as f:
    pickle.dump(dataset, f)
