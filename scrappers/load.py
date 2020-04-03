import sys
import json
import argparse
import sys, os
sys.path.append(os.path.abspath('..'))
from database_core.connection import Connection
from collections import OrderedDict
from tqdm import tqdm

flags = None

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Scrapped data to process.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def load_one(conn, row):
    with conn.CLIENT.start_session() as session:
        with session.start_transaction():
            #index_attr = 'cood_uid'
            #if index_attr not in row.keys():
            index_attr = 'hash_id'
            if index_attr not in row.keys():
                print('Skipping document without hash...\n%r' % row)
                return

            doc = conn.DB.documents.find_one({index_attr: row[index_attr]}, {'raw.sections': 1})
            if doc is None:
                print('Skipping document not exists...\n%r' % row)
                return
            #if doc is None:
            aux = doc['raw']['sections']
            doc['raw']['sections'] = OrderedDict()
            for i, key in enumerate(aux.keys()):
                if 'toc' in row and row['toc'] is not None and i < len(row['toc']):
                    new_key = row['toc'][i]
                else:
                    new_key = key
                doc['raw']['sections'][new_key] = aux[key]
            
            conn.DB.documents.update_one({index_attr: row[index_attr]}, 
                {'$set': {'url': row['link'], 'raw.sections': doc['raw']['sections']}}, upsert=True)

def main():
  data = json.load(open(flags.data, 'r'), object_pairs_hook=OrderedDict)
  for row in tqdm(data):
    load_one(Connection, row)

if __name__ == '__main__':
  flags = parse_flags()
  main()
