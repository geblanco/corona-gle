import requests
import requests_random_user_agent
import threading
from pymongo import errors, MongoClient
import config
import json
from progress.bar import Bar

threads_amount = 5
threads = list()

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
        sections.append(get_title(text))
    elif text['#name'] == 'entry':
        for elem in text['$$']:
            sections = process_entry(elem, sections)
            # res = process_entry(elem, [], 0)
            # print('res:' + str(res))
            # sections['subsections'] = res
            # sections.append(aux)
    # print('->' + str(sections) + '<-')
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
                        pass
    print(sections)
    return(sections)

def process_outlines(bar, sem, db_client, col, sha):
    '''
    After the processing of the outline, we can have three states:
    1. Checked = True; toc = None
    2. Checked = True; toc = []
    3. Checked = True; toc = [content]

    The first state means that it has been process, but an error has been encountered; 
    the second means that it has been correctly processed, but there are no content; 
    and the last one means correct execution and available content
    '''
    with sem:
        # bar.next()
        db_papers = db_client[config.db_name]
        col_to_work = db_papers[col]
        sections = {}
        try:
            json_outline = json.loads(col_to_work.find_one({'sha':sha}, {'_id':0, 'raw':1})['raw'])['outline']
            sections = process_outline(json_outline)
            col_to_work.update_one({'sha':sha}, {'$set': {'checked':True, 'toc': sections}})
        except Exception as e:
            col_to_work.update_one({'sha':sha}, {'$set': {'checked':True, 'toc': None}})
            raise(e) 
        
def thread_caller(sem, db_client, col):
    db_papers = db_client[config.db_name]
    col_to_work = db_papers[col]
    filter_checked = {'checked': True}
    documents = col_to_work.find(filter_checked, {'_id': 0, 'sha': 1})
    bar = Bar('Processing', max=col_to_work.count_documents)(filter_checked)
    for document in documents:
        try:
            t = threading.Thread(target=process_outlines, args=(bar, sem, db_client, col, document['sha']))
            threads.append(t)
            t.start()
        except RuntimeError:
            pass
        except Exception as e:
            raise(e)
            # print('[0] error obtaining info from "{}"'.format(document['sha']))
    for t in threads:
        t.join()
    bar.finish()

if __name__ == "__main__":
    col_elsevier = config.collection_elsevier
    db_client = MongoClient(config.mongoURL)
    sem = threading.Semaphore(threads_amount)
    thread_caller(sem, db_client, col_elsevier)
    # process_outlines(None, sem, db_client, col_elsevier, '28ef2f6aa0e0c14f92055d24d72b151fd4da910a')
    db_client.close()

