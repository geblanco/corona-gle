import sys
import json
import argparse

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
      index_attr = 'cood_uid'
      if row[index_attr] is not None:
        index_attr = 'hash_id'
      if row[index_attr] is None:
        print('Skipping document without hash...\n%r' % row)
        return
      conn.DB.documents.update_one({index_attr: row[index_attr]}, 
        {'$set': {'sections_scrapped': row['sections']}}, upsert=True)

def main():
  data = json.load(open(flags.data, 'r'))
  for row in data:
    load(row)

if __name__ == '__main__':
  flags = parse_flags()
  main()
