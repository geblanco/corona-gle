import sys, os
sys.path.append(os.path.abspath('./../'))
from database_core import *
import time
#Database.sync()
Database.update_mean_vectors('FlairEmbeddings', use='raw')
# a0 = time.time()
# Database.list_doc_embeddings('SpacyEmbeddings')
# print(time.time() - a0)

# a0 = time.time()
# Database.list_doc_embeddings('SpacyEmbeddings')
# print(time.time() - a0)