from flask import Flask,request
import json
import numpy as np
import requests

import spacy
import scispacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_lg", disable=["tagger"])

app = Flask(__name__)


#model = sentence_transformers.SentenceTransformer("../common/inputData/sentence_transformer_nli/model_results") #from CoronaWhy dataset
#model.cuda();

sess = requests.Session()


@app.route('/encode',methods=['POST'])
def encode():
	query = request.form['query']
	print('Model processing {}'.format(query))
	result = nlp(query)

	res_vec = result.vector.tolist()
	#print('Model sending to FAISS: {}'.format(res_vec))
	res = sess.post('http://indexmap:5001/semsearch',data={'query_vec':res_vec})
	return {
		'result': res.json()['result']
	}


@app.route('/config',methods=['GET'])
def get_config():
	pass
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)