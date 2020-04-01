# Dependences
from flask import Flask, Response, request, current_app, abort
from flask_cors import CORS
from datetime import datetime, timedelta
from flask_jwt import JWT, JWTError, jwt_required, current_identity, _jwt
from werkzeug.exceptions import HTTPException
from functools import wraps, partial
import os
import json

import traceback
#import uuid
#import pymongo
import glob2
from database_core import Database
from search_engine import SearchEngine

from threading import Thread

# Start APP
app = Flask(__name__)
CORS(app)
SERVER_PORT = 7575

# Load dataset
EMBEDDINGS = {}
ENGINES = {}
for method_name in Database.list_methods():
    method = Database.get_method(method_name)

    EMBEDDINGS[method_name] = Database.list_doc_embeddings(method_name)
    ENGINES[method_name] = SearchEngine(method, EMBEDDINGS[method_name], use_faiss=False) 
"""
==========================================0
    USERS
==========================================0
"""
from werkzeug.security import safe_str_cmp
from functools import wraps
def authenticate(username, password):
    data = json.loads(request.data)
    user = User.get(username=username)
    if not (user and safe_str_cmp(user.password.encode('utf-8'), User.cypher(password).encode('utf-8'))):
        return

    if 'group' in data and data['group'] != user.group:
        raise JWTError('Bad Request', 'Invalid group')

    if not user.enable:
        raise JWTError('Bad Request', 'Removed user')

    user.ref_id = str(user.ref_id)
    return user

def identity(payload):
    user_id = payload['identity']
    user = User.get(ref_id=user_id)
    user.ref_id = str(user.ref_id)
    return user

def group_decorator(group=None, groups=[]):
    if group is not None:
        groups = [group]
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            if current_identity.group not in groups:
                raise JWTError('Bad Request', 'Invalid group')
            return fn(*args, **kwargs)
        return decorator
    return wrapper

"""
==========================================0
    APP
==========================================0
"""
@app.route('/init', methods=["GET"])
def init():
    return {
        'algorithms': Database.list_methods()
    }

@app.route('/search', methods=["POST"])
def search():
    data = json.loads(request.data)
    search_query = data['query']
    algo_query = data['algorithm']
    hash_ids = ENGINES[algo_query].get_similar_docs_than(search_query, k=300)
    documents = Database.list_raw_documents(hash_ids=hash_ids)

    documents_return = []
    for i, doc in enumerate(documents):
        documents_return.append({
            'rank': i,
            'reference_id': doc['hash_id'],
            'title': doc['title'],
            'abstract': doc['raw']['sections']['abstract'] if 'abstract' in doc['raw']['sections'] else "-",
            'body': "\n".join([v for k, v in doc['raw']['sections'].items() if k != "abstract"])
        })

    return {
        'documents': documents_return
    }

if __name__ == '__main__':
    app.run(port=SERVER_PORT)