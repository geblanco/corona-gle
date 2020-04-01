import sys, os
sys.path.append(os.path.abspath('..'))
from database_core import Connection

conn = Connection()
print(list(conn.DB.documents.find().limit(1)))
