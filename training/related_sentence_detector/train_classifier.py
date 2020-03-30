import random
import pickle
import pandas as pd
import os
import sys
import numpy as np
import csv
csv.field_size_limit(sys.maxsize)

curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, "data")
logs_path = os.path.join(curr_path, "logs")

if not os.path.exists(os.path.join(data_path, "train.csv")):
    with open(os.path.join(data_path, 'classification_dataset.pickle'), 'rb') as handle:
        data = pickle.load(handle)

    labels = []
    texts = []
    for d in data:
        texts.append(d['section_text'].replace("\t", " "))
        labels.append(d['section_match'])

    # Others
    #for d in

    indices = list(range(len(labels)))
    random.shuffle(indices)

    split_v = 0.85
    total_length = len(texts)
    train_length = int(total_length * split_v)

    test_dev_length = total_length - train_length

    dev_length = int(test_dev_length / 2)
    test_length = test_dev_length - dev_length

    train_indices = indices[:train_length]
    dev_indices = indices[train_length:train_length+dev_length]
    test_indices = indices[train_length+dev_length:]

    labels_train = []
    texts_train = []
    labels_test = []
    texts_test = []
    labels_dev = []
    texts_dev = []
    for i, (label, text) in enumerate(zip(labels, texts)):
        if text == "":
            continue

        if i in train_indices:
            labels_train.append(label)
            texts_train.append(text)

        elif i in test_indices:
            labels_test.append(label)
            texts_test.append(text)

        elif i in dev_indices:
            labels_dev.append(label)
            texts_dev.append(text)
    
    labels_splits = {
        'train': {
            'label': labels_train, #[label for i, (text, label) in enumerate(zip(texts, labels)) if i in train_indices and text != ""],
            'text': texts_train, #[text for i, text in enumerate(texts) if i in train_indices and text != ""]
        },
        'test': {
            'label': labels_test, #[label for i, (text, label) in enumerate(zip(texts, labels)) if i in test_indices and text != ""],
            'text': texts_test, #[text for i, text in enumerate(texts) if i in test_indices and text != ""]
        },
        'dev': {
            'label': labels_dev, #[label for i, (text, label) in enumerate(zip(texts, labels)) if i in dev_indices and text != ""],
            'text': texts_dev, #[text for i, text in enumerate(texts) if i in dev_indices and text != ""]
        },
    }

    for split_name, split_value in labels_splits.items():
        df = pd.DataFrame(split_value)
        df.to_csv(
            os.path.join(data_path, f'{split_name}.csv'), 
            index=False,
            header=False,
            sep="\t"
        )

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
corpus = CSVClassificationCorpus(
    data_path,
    column_name_map={0: 'label', 1: 'text'},
    skip_header=False,
    delimiter='\t',
    in_memory=False,
    max_tokens_per_doc=1000*10
)

import torch
import flair
flair.devide = torch.device('cuda:0')
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [# comment in flair embeddings for state-of-the-art results
   #FlairEmbeddings('news-forward-fast', chars_per_chunk=512),
   #FlairEmbeddings('news-backward-fast', chars_per_chunk=128),
   WordEmbeddings('glove')
]

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings = DocumentRNNEmbeddings(word_embeddings,
     hidden_size=128,
     reproject_words=True,
     reproject_words_dimension=128,
 )

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train(logs_path,
              learning_rate=0.1,
              mini_batch_size=8,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150,
              checkpoint=True,
              monitor_train=False,
              monitor_test=False,
              embeddings_storage_mode="cpu",
              #eval_mini_batch_size=4,
              train_with_dev=True,
              use_amp=True,
              amp_opt_level='O1')

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_weights(os.path.join(logs_path, 'weights.txt'))