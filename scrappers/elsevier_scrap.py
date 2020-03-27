import requests
import requests_random_user_agent
import signal
import threading
# from termcolor import cprint

base_url = 'https://www.sciencedirect.com/sdfe/arp/pii/{}/toc'
headers = {"Connection":"close","Accept-Language":"en-US,en;q=0.5","Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Upgrade-Insecure-Requests":"1"}

threads_amount = 5
threads = list()

# file_ = 'elsevier_papers.csv'
file_ = 'elsevier_papers_1001-2000.csv'
result = file_.split('.')[0]+'_res.csv'

def get_title(text):
    if '_' in text.keys():
        return(text['_'])
    else:
        aux = ''
        for part in text['$$']:
            aux += part['_']
        return(aux)

def process_entry(text, sections):
    # print('---------------------')
    # print(sections)
    # print('---------------------')
    if text['#name'] == 'title':
        if '_' in text.keys():
            sections.append(get_title(text))
    elif text['#name'] == 'entry':
        for elem in text['$$']:
            sections = process_entry(elem, sections)
    return(sections)

def get_info(lock, sha, link):
    with lock:
        try:
            sections = []
            r1 = requests.get(link, headers = headers)
            id_paper = r1.url.split('/')[-1]
            # print(r1.url)
            res = requests.get(base_url.format(id_paper), headers = headers)
            for elem in res.json()['outline']:
                if elem['$']['type'] == 'sections':
                    for sec in elem['$$']:
                        if sec['#name'] == 'title':
                            sections.append(get_title(sec))
                        elif sec['#name'] == 'entry':
                            for entry in sec['$$']:
                                sections = process_entry(entry, sections)
                                # print('---{}---'.format(str(sections)))
            with open(result, 'a') as fich_out:
                fich_out.write(sha)
                fich_out.write(',')
                fich_out.write(str(sections)[1:-1].replace("', '", ';').replace("'", ''))
                fich_out.write('\n')
            print('{} escrito correctamente'.format(link))
            # print('{},{}'.format(sha, sections))
        except Exception as e:
            print('[1] error obtaining info from "{}"'.format(link))
            print(sections)
            print(res.json()['outline'])
            raise(e) 
        
def thread_caller(lock, file_):
    elsev_file = open(file_, 'r')
    for line in elsev_file:
        try:
            (id_, sha, origin, link) = line[:-1].split(',')
            t = threading.Thread(target=get_info, args=(lock, sha, link))
            threads.append(t)
            t.start()
        except:
            print('[0] error obtaining info from "{}"'.format(id_))
    for t in threads:
        t.join()

if __name__ == "__main__":
    # test = firebase_url_checker.FRB_Permissions_Checker()
    sem = threading.Semaphore(threads_amount)
    lock = threading.Lock()
    # get_info('asdf', 'https://www.sciencedirect.com/science/article/pii/S0022283697911347')
    # get_info(lock,'asdf', 'https://www.sciencedirect.com/science/article/pii/S1044579X09000157')
    thread_caller(lock, file_)
