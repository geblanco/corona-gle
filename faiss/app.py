import faiss
import os
import numpy as np

import requests
import logging as log

from flask import Flask,request

sess = requests.Session()

index_dir = '/media/data/faiss'


if os.environ['FLASK_ENV'] == 'development':
	log.info('reading dev faiss')
	index = faiss.read_index(os.path.join(index_dir, "dev_faiss.index"))
else:
	log.info('reading from {}'.format(os.path.join(index_dir)
        index = faiss.read_index(os.path.join(index_dir, "faiss.index"))


in_dir = '/media/data/'

'''
def str_num_sort(inp_str):
    return int(inp_str.split('_')[-1].split('.')[0])


pickle_list = [in_dir + p for p in os.listdir(in_dir)]
pickle_list.sort(key=str_num_sort)


mini_df = pd.DataFrame([],columns=['paper_id','sentence_id','sentence'])
for pickle in pickle_list[:2]:
	pickle_df = pd.read_csv(pickle,sep='\t',usecols=['paper_id','sentence_id','sentence'])
	mini_df = pd.concat([mini_df,pickle_df])

logger.info('loaded mini_df: {}'.format(len(mini_df)))
'''

app = Flask(__name__)
app.debug = True
def search_vec(query_vector,k=5):
	query_vector = np.array(query_vector,dtype=np.float32).reshape(1,len(query_vector))
	D,I = index.search(query_vector, k)

	_id = I.tolist()[0]
	_id = list(map(lambda x: hex(x).split('x')[1],_id))

	return _id



@app.route('/semsearch',methods=['POST'])
def search():
	query_vec = request.form.getlist('query_vec')
	#print("FAISS has received : {}".format(query_vec))
	return {
		'result': search_vec(query_vec)
	}



def get_config():
	pass
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)
