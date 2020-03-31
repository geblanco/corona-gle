from pymongo import *
from collections import OrderedDict
from . import Params

class Connection:
	CLIENT = None
	DB = None
try:
    Connection.CLIENT = MongoClient(Params.DB_URL, document_class=OrderedDict)
    Connection.DB = Connection.CLIENT[Params.DB_NAME]
    Connection.CLIENT.server_info()
except Exception as e:
    raise e