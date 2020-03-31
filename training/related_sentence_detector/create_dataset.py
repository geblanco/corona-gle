import sys
import os
# isn't there a better way to do this?
sys.path.append(os.path.abspath('../../'))

# from methods import *
from database import Database

import torch
import flair
import pickle

import scispacy
import spacy
from tqdm import tqdm

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings


curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, "data")

nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

flair.device = torch.device('cuda:0')
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
for i, doc in tqdm(enumerate(documents), desc='Embedding documents'):
    for section_title, section_text in doc['raw']['sections'].items():
        text = section_text.strip()
        # tokenize each sentence
        # skip language detection, embedding wont match anyway
        nlp_doc = nlp(text)
        for sentence in nlp_doc.sents:
            flair_sent = Sentence(sentence.text)
            flair_emb.embed(flair_sent)

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
                dataset.append({
                    'hash_id': doc['hash_id'],
                    'title': section_title,
                    'text': sentence.text,
                    'match': max_part
                })
    
    print(i, len(documents))

os.makedirs(os.path.join(data_path), exist_ok=True)
with open(os.path.join(data_path, 'classification_dataset.pickle'), 'wb') as f:
    pickle.dump(dataset, f)
