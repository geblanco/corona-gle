import requests_random_user_agent
import concurrent.futures as cf
import requests
import config
import json

from tqdm import tqdm
from pymongo import errors, MongoClient

NUM_WORKERS = 8
threads_amount = 5
threads = list()

"""
sample
{'#name': 'entry',
 '$': {'id': 'sec3', 'type': 'sections', 'depth': '1'},
 '$$': [{'#name': 'label', '_': '3'},
  {'#name': 'title', '$': {'id': 'sectitle0065'}, '_': 'Results'},
  {'#name': 'entry',
   '$': {'id': 'sec3.1', 'depth': '2'},
   '$$': [{'#name': 'label', '_': '3.1'},
    {'#name': 'title',
     '$': {'id': 'sectitle0070'},
     '$$': [{'#name': '__text__',
       '_': 'Alteration of miRNAs after IBV infection '},
      {'#name': 'italic', '$': {'xmlns': True}, '_': 'in vivo'}]}]},
  {'#name': 'entry',
   '$': {'id': 'sec3.2', 'depth': '2'},
   '$$': [{'#name': 'label', '_': '3.2'},
    {'#name': 'title',
     '$': {'id': 'sectitle0075'},
     '_': 'Up-regulation of miR-146a-5p in cells infected with Beaudette'}]},
  {'#name': 'entry',
   '$': {'id': 'sec3.3', 'depth': '2'},
   '$$': [{'#name': 'label', '_': '3.3'},
    {'#name': 'title',
     '$': {'id': 'sectitle0080'},
     '_': 'miR-146a-5p related genes'}]}]}
"""
def get_title(text):
    aux = ''
    if '_' in text.keys():
        aux = text['_']
    elif text['$$']:
        aux = []
        for part in text['$$']:
            if '_' in part.keys():
                aux.append(part['_'].strip())
            elif part['#name'] == 'title':
                # only one level down
                aux.extend([p['_'].strip() for p in part['$$']])
        aux = ' '.join(aux).strip()
    return aux

def get_type(text):
    ret = 'UNK'
    if '$' in text.keys() and 'type' in text['$'].keys():
        ret = text['$']['type']
    return ret

def process_entry(text):
    section = {}
    if text['#name'] == 'entry':
        section['name'] = get_title(text)
        section['type'] = get_type(text)
        section['subsections'] = []
        if '$$' in text.keys():
            for entry in text['$$']:
                subsection = process_entry(entry)
                if subsection is not None:
                    section['subsections'].append(subsection)
    return section if len(section.keys()) > 0 else None

def process_outline(json_outline):
    sections = []
    for entry in json_outline:
        sections.append(process_entry(entry))
    return sections

def process_raw_column(raw_column):
    json_outline = json.loads(raw_column['raw'])['outline']
    sections = process_outline(json_outline)
    return { 'sha': raw_column['sha'], 'sections': sections }

def main(client, database):
    filter_checked = {'checked': True}
    documents = database.find(filter_checked)
    with cf.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for sha, sections in tqdm(executor.map(process_raw_column, documents), total=len(documents)):
            with Connection.CLIENT.start_session() as session:
                with session.start_transaction():
                    database.update_one({'sha': sha}, {'$set': {'checked':True, 'toc': sections}})

if __name__ == "__main__":
    col_elsevier = config.collection_elsevier
    db_client = MongoClient(config.mongoURL)
    db_papers = db_client[config.db_name]
    main(db_client, db_papers[col_elsevier])
    db_client.close()
