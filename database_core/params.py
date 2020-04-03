from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()

import os

mongo_data = dict(
  user = quote(os.getenv('MONGO_USER', 'root')),
  passwd = quote(os.getenv('MONGO_PASS', 'claveRoot')),
  ip = os.getenv('MONGO_IP', '127.0.0.1'),
  port = os.getenv('MONGO_PORT', '3310'),
)

import os, sys
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve1d

def v(p, num):
    if p == 1:
        return np.ones(shape=(num, ))
    else:
        return np.zeros(shape=(num, ))

def create_prob_distribution(bin_signal, amplitude_2, laplace_smooth):
    if isinstance(bin_signal, (list, tuple)):
        bin_signal = np.concatenate(bin_signal, axis=0)

    limit_signal = signal.hann(2 * amplitude_2)
    aux = convolve1d(bin_signal, limit_signal, mode='nearest')
    aux = aux / np.sum(aux)
    
    aux = aux + laplace_smooth
    aux = aux / np.sum(aux)
    return aux

class Params:
    DB_URL = 'mongodb://{user}:{passwd}@{ip}:{port}'.format(**mongo_data)
    DB_NAME = "coronagle_db"

    DATASET_KAGGLE_NAME = 'allen-institute-for-ai/CORD-19-research-challenge'
    DATASET_KAGGLE_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

    SCAN_WORKERS = 8
    COMPUTE_VECTORS_WORKERS = 8
    READ_EMBEDDINGS_WORKERS = 12
    
    SECTIONS_CLASSIFIER_KEYWORDS = {
        '#introduction': ['intro', 'introduction', 'starting'],
        '#abstract': ['abstract', 'abstracts'],
        '#sota': ['background', 'backgrounds', 'state of the art', 'previous', 'related work'],
        '#method': ['method', 'methods', 'methodology', 'material', 'materials', 'development', 'description', 'model', 'procedures'],
        '#experiments_or_results': ['experiments', 'experiment', 'analysis', 'analytics', 'analisy', 'statistics', 'regression', 
            'analises', 'results', 'result', 'evaluation', 'measures', 'correlation', 'comparison', 'tests', 'test', 'lab', 'laboratory'],
        '#conclusions': ['conclusion', 'conclusions', 'discussion', 'discussions'],
    }
    SECTIONS_CLASSIFIER_POSITIONS_COND_CLASS = { # P_pos_cond_c
        '#introduction': create_prob_distribution([v(0, 5), v(1, 25), v(0, 70)], 15, 0.005),
        '#abstract': create_prob_distribution([v(1, 35), v(0, 65)], 15, 0.005),
        '#sota': create_prob_distribution([v(0, 5), v(1, 25), v(0, 70)], 15, 0.005),
        '#method': create_prob_distribution([v(0, 15), v(1, 70), v(0, 15)], 15, 0.005),
        '#experiments_or_results': create_prob_distribution([v(0, 25), v(1, 70), v(0, 5)], 15, 0.005),
        '#conclusions': create_prob_distribution([v(0, 70), v(1, 30)], 15, 0.005),
    }
    SECTIONS_CLASSIFIER_PRIORS = {
        '#introduction': 1/6,
        '#abstract': 1/6,
        '#sota': 1/6,
        '#method': 1/6,
        '#experiments_or_results': 1/6,
        '#conclusions': 1/6,
    }