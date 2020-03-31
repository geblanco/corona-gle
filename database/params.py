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

class Params:
	DB_URL = 'mongodb://{user}:{passwd}@{ip}:{port}'.format(**mongo_data)
	DB_NAME = "coronagle_db"

	DATASET_KAGGLE_NAME = 'allen-institute-for-ai/CORD-19-research-challenge'
	DATASET_KAGGLE_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

	SCAN_WORKERS = 8
	COMPUTE_VECTORS_WORKERS = 8
	READ_EMBEDDINGS_WORKERS = 12
