import os

class Params:
	DB_URL = 'mongodb://root:claveRoot@127.0.0.1:3310'
	DB_NAME = "coronagle_db"

	DATASET_KAGGLE_NAME = 'allen-institute-for-ai/CORD-19-research-challenge'
	DATASET_KAGGLE_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

	SCAN_WORKERS = 8
	COMPUTE_VECTORS_WORKERS = 8
	READ_EMBEDDINGS_WORKERS = 12