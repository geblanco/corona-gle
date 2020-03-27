import requests
import requests_random_user_agent
import signal
import threading
from pymongo import errors, MongoClient
import config
from progress.bar import Bar

base_url = 'https://www.sciencedirect.com/sdfe/arp/pii/{}/toc'
headers = {"Connection":"close","Accept-Language":"en-US,en;q=0.5","Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Upgrade-Insecure-Requests":"1"}

threads_amount = 5
threads = list()

file_ = 'elsevier_papers.csv'

def get_title(text):
    if '_' in text.keys():
        return(text['_'])
    else:
        aux = ''
        for part in text['$$']:
            aux += part['_']
        return(aux)

def process_entry(text, sections):
    if text['#name'] == 'title':
        if '_' in text.keys():
            sections.append(get_title(text))                        
    elif text['#name'] == 'entry':
        for elem in text['$$']:
            sections = process_entry(elem, sections)
    return(sections)

def process_outline(json_outline):
    sections = []
    for elem in json_outline:
        if elem['$']['type'] == 'sections':
            for sec in elem['$$']:
                if sec['#name'] == 'title':
                    sections.append(get_title(sec))
                elif sec['#name'] == 'entry':
                    for entry in sec['$$']:
                        sections = process_entry(entry, sections)
    return sections

def get_info(bar, sem, db_client, col, sha, link):
    with sem:
        bar.next()
        db_papers = db_client[config.db_name]
        col_to_work = db_papers[col]
        try:
            r1 = requests.get(link, headers = headers)
            id_paper = r1.url.split('/')[-1]
            res = requests.get(base_url.format(id_paper), headers = headers)
            res_json = res.json()
            json_outline = res_json['outline']
            sections = process_outline(json_outline)
            col_to_work.update_one({'sha':sha}, {'$set': {'checked':True, 'toc': sections, 'raw': json.dumps(obj=res_json)}})
        except Exception as e:
            col_to_work.update_one({'sha':sha}, {'$set': {'checked':True}})
            # raise(e) 
        
def thread_caller(sem, db_client, col):
    db_papers = db_client[config.db_name]
    col_to_work = db_papers[col]
    documents = col_to_work.find({'checked': False})
    bar = Bar('Processing', max=col_to_work.count_documents({'checked': False}))
    for document in documents:
        try:
            t = threading.Thread(target=get_info, args=(bar, sem, db_client, col, document['sha'], document['link']))
            threads.append(t)
            t.start()
        except Exception as e:
            pass
            # raise(e)
            # print('[0] error obtaining info from "{}"'.format(document['sha']))
    for t in threads:
        t.join()
    bar.finish()
        
def insert_elems_db_elsevier(db_client, file_):
    db_papers = db_client[config.db_name]
    col_elsevier = db_papers[config.collection_elsevier]
    with open(file_, 'r') as file_in:
        for line in file_in:
            try:
                (id_, sha, origin, link) = line[:-1].split(',')
                if not col_elsevier.find_one({'sha': sha}):
                    col_elsevier.insert_one({
                        'sha': sha,
                        'link': link,
                        'checked': False,
                        'toc': None
                    })
            except:
                print('[2] error inserting in db info from "{}"'.format(id_))
    
if __name__ == "__main__":
    col_elsevier = config.collection_elsevier
    db_client = MongoClient(config.mongoURL)
    db_papers = db_client[config.db_name]
    col_to_work = db_papers[col_elsevier]
        
    with open(file_, 'r') as file_in:
        n_lines = len(file_in.readlines())
    if n_lines != col_to_work.count_documents({}):
        insert_elems_db_elsevier(db_client, file_)
    sem = threading.Semaphore(threads_amount)
    thread_caller(sem, db_client, col_elsevier)
    db_client.close()

