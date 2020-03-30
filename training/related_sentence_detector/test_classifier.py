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
os.makedirs(logs_path, exist_ok=True)

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
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import DataLoader
mt = ModelTrainer.load_checkpoint(os.path.join(logs_path, "checkpoint.pt"), corpus)


test_results, test_loss = mt.model.evaluate(
    DataLoader(
        corpus.test,
        batch_size=4,
        num_workers=4,
    ),
    out_path=os.path.join(logs_path, "test.tsv"),
    embedding_storage_mode="none",
)

with open(os.path.join(logs_path, "test.txt"), "w") as f:
    f.write(str(test_results.main_score) + "\n\n")
    f.write(str(test_results.log_header) + "\n")
    f.write(str(test_results.log_line) + "\n\n")
    f.write(str(test_results.detailed_results))