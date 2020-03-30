from methods import *
from database import Database
import torch
import flair
import pickle
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings

curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, "data")

flair.device = torch.device('cuda:0')
flair.embedding_storage_mode = None
flair_emb = DocumentPoolEmbeddings([
        FlairEmbeddings('en-forward-fast'), 
        FlairEmbeddings('en-backward-fast')
    ],
    pooling='mean',
)
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

poss_sections = {
    '#introduction': ['intro', 'introduction', 'starting'],
    '#abstract': ['abstract', 'abstracts'],
    '#sota': ['background', 'backgrounds', 'state of the art', 'previous', 'related work'],
    '#method': ['method', 'methods', 'methodology', 'material', 'materials', 'development', 'description', 'model', 'procedures'],
    '#experiments_or_results': ['experiments', 'experiment', 'analysis', 'analytics', 'analisy', 'statistics', 'regression', 
        'analises', 'results', 'result', 'evaluation', 'measures', 'correlation', 'comparison', 'tests', 'test', 'lab', 'laboratory'],
    '#conclusions': ['conclusion', 'conclusions', 'discussion', 'discussions'],
}

for list_candidates in poss_sections.values():
    for i in range(len(list_candidates)):
        sentence = Sentence(list_candidates[i].lower())
        flair_emb.embed(sentence)
        list_candidates[i] = sentence.embedding
        sentence.clear_embeddings()

dataset = []
documents = Database.list_raw_documents()
for i, doc in enumerate(documents):
    for section_title, section_text in doc['raw']['sections'].items():
        if section_title == "" or section_title.isnumeric():
            continue

        sentence = Sentence(section_title.lower())
        flair_emb.embed(sentence)
        mean_vector = sentence.embedding

        max_value = -2
        max_section = None
        for possible_section, candidates in poss_sections.items():
            for candidate in candidates:
                score = cos(mean_vector, candidate)
                if score > 0.9: # consideramos valido
                    if max_value < score:
                        max_value = score
                        max_section = possible_section

        sentence.clear_embeddings()

        if max_section is not None: # Hay seccion seleccionada
            dataset.append({
                'hash_id': doc['hash_id'],
                'title': section_title,
                'text': section_text,
                'match': max_section
            })
    
    print(i, len(documents))

os.makedirs(os.path.join(data_path), exist_ok=True)
with open(os.path.join(data_path, 'classification_dataset.pickle'), 'wb') as f:
    pickle.dump(dataset, f)
