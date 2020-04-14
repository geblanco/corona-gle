import faiss
import pandas as pd
import os
import numpy as np

def str_num_sort(inp_str):
    return int(inp_str.split('_')[-1].split('.')[0])

PATH = '/media/data/NLPDatasets_v6_preprocessed_v6_vectors/'
pickle_list = [PATH + p for p in os.listdir(PATH)]
pickle_list.sort(key=str_num_sort)



columns = [str(x) for x in range(200)]


index = faiss.IndexIDMap(faiss.IndexFlatIP(200))

for pickle in pickle_list:
    name = pickle.split('/')[-1]
    print('----------file_{}------------'.format(name))
    p_df = pd.read_pickle(pickle,compression='gzip')
    vec_arr = np.ascontiguousarray(p_df[columns].to_numpy(),dtype=np.float32)

    
    p_df['sentence_id'] = p_df['sentence_id'].map(lambda s_id: int(s_id,16))

    index.add_with_ids(vec_arr, p_df['sentence_id'].values)
    print('added {} vectors to index'.format(len(p_df)))




state_dir = '/media/data/'

faiss.write_index(
    index,
    os.path.join(state_dir, "faiss.index")
)

#index = faiss.read_index(os.path.join(state_dir, "faiss.index"))