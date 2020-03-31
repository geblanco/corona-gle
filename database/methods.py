from . import clean_text
from . import Database
import numpy as np

class SpacyEmbeddings:
    NAME = 'SpacyEmbeddings'
    NUM_DIMENSIONS = 200

    @classmethod
    def init(cls):
        import scispacy
        import spacy

        cls.NLP = spacy.load("en_core_sci_lg")

    @classmethod
    def get_mean_vector(cls, sections_vector):
        accum_vector = np.zeros(shape=(cls.NUM_DIMENSIONS, ))
        num_total_elements = 0

        for k in sections_vector.keys():
            if sections_vector[k] is None:
                continue
            vector = sections_vector[k]['vector']
            accum_vector += vector
            num_elements = sections_vector[k]['num_elements']
            num_total_elements += num_elements

        if num_total_elements > 0:
            accum_vector /= num_total_elements
            return accum_vector
        
        return None

    @classmethod
    def get_mean_vector_from_section(cls, sections_vector, section, translate_lut=None):
        accum_vector = np.zeros(shape=(cls.NUM_DIMENSIONS, ))
        num_total_elements = 0

        for k in sections_vector.keys():
            if translate_lut is None:
                fix_section = k
            else:
                fix_section = translate_lut[k]

            if fix_section == section:
                if sections_vector[k] is None:
                    continue
                vector = sections_vector[k]['vector']
                accum_vector += vector
                num_elements = sections_vector[k]['num_elements']
                num_total_elements += num_elements

        if num_total_elements > 0:
            accum_vector /= num_total_elements
            return accum_vector
        
        return None

    @classmethod
    def compute_mean_vector(cls, raw_or_clean_doc):
        sections_vector = {}
        for k, v_text in raw_or_clean_doc['sections'].items():
            c_text = clean_text(v_text)
            if c_text is None:
                sections_vector[k] = None
                continue

            doc_spacy = cls.NLP(c_text)
            sections_vector[k] = {
                'vector': doc_spacy.vector,
                'num_elements': sum([token.has_vector for token in doc_spacy])
            }
        return sections_vector
Database.register_method(SpacyEmbeddings)

class FlairEmbeddings:
    NAME = 'FlairEmbeddings'
    NUM_DIMENSIONS = None
    SENTENCE = None
    FLAIR_EMB = None
    NUM_WORKERS = 1
    TYPE_THREADING = None

    @classmethod
    def init(cls):
        import torch
        import flair
        flair.device = torch.device('cuda:1')
        from flair.data import Sentence
        from flair.embeddings import FlairEmbeddings as FlairEmbeddings__
        cls.SENTENCE = Sentence
        cls.FLAIR_EMB = FlairEmbeddings__('en-forward-fast')
        cls.NUM_DIMENSIONS = cls.FLAIR_EMB.embedding_length
        cls.BATCH_SIZE = 3
        cls.FLAIR = flair
        flair.embedding_storage_mode = None

    @classmethod
    def get_mean_vector(cls, sections_vector):
        accum_vector = np.zeros(shape=(cls.NUM_DIMENSIONS, ))
        num_total_elements = 0

        for k in sections_vector.keys():
            if sections_vector[k] is None:
                continue
            vector = sections_vector[k]['vector']
            accum_vector += vector
            num_elements = sections_vector[k]['num_elements']
            num_total_elements += num_elements

        if num_total_elements > 0:
            accum_vector /= num_total_elements
            return accum_vector
        
        return None

    @classmethod
    def get_mean_vector_from_section(cls, sections_vector, section, translate_lut=None):
        accum_vector = np.zeros(shape=(cls.NUM_DIMENSIONS, ))
        num_total_elements = 0

        for k in sections_vector.keys():
            if translate_lut is None:
                fix_section = k
            else:
                fix_section = translate_lut[k]

            if fix_section == section:
                if sections_vector[k] is None:
                    continue
                vector = sections_vector[k]['vector']
                accum_vector += vector
                num_elements = sections_vector[k]['num_elements']
                num_total_elements += num_elements

        if num_total_elements > 0:
            accum_vector /= num_total_elements
            return accum_vector
        
        return None

    @classmethod
    def compute_mean_vector(cls, raw_or_clean_doc):        
        sections_vector = {}

        clean_text_sentences_keys = []
        clean_text_sentences_values = []
        for k, v_text in raw_or_clean_doc['sections'].items():
            c_text = clean_text(v_text)
            if c_text is None or c_text == "":
                sections_vector[k] = None
                continue

            clean_text_sentences_keys.append(k)
            clean_text_sentences_values.append(cls.SENTENCE(c_text))

        for i in range(0, len(clean_text_sentences_values), cls.BATCH_SIZE):
            c_batch = clean_text_sentences_values[i:i+cls.BATCH_SIZE]
            cls.FLAIR_EMB.embed(c_batch)

        for k, doc_flair in zip(clean_text_sentences_keys, clean_text_sentences_values):                
            mean_vector = [token.embedding.cpu().numpy() for token in doc_flair]
            num_elements = len(mean_vector)
            mean_vector = np.mean(mean_vector, axis=0)
            doc_flair.clear_embeddings()

            
            sections_vector[k] = {
                'vector': mean_vector,
                'num_elements': num_elements
            }

        #cls.THREAD(target=cls.GC.collect).start()
        return sections_vector
Database.register_method(FlairEmbeddings)

# class Word2Vec:
#   NAME = 'Word2Vec'
#   NUM_DIMENSIONS = 200

#   @classmethod
#   def get_mean_vector(doc):
#       accum_vector = np.zeros(shape=(Word2Vec.NUM_DIMENSIONS, ))
#       num_total_elements = 0

#       for k in doc['sections'].keys():
#           accum_vector += doc['sections'][k]['vector']
#           num_elements = doc['sections'][k]['num_elements']
#           num_total_elements += num_elements

#       if num_total_elements > 0:
#           accum_vector /= num_total_elements
#       return accum_vector

#   @classmethod
#   def get_mean_vector_from_section(doc, section, translate_lut=None):
#       accum_vector = np.zeros(shape=(Word2Vec.NUM_DIMENSIONS, ))
#       num_total_elements = 0

#       for k in doc['sections'].keys():
#           if translate_lut is None:
#               fix_section = k
#           else:
#               fix_section = translate_lut[k]

#           if fix_section == section:
#               accum_vector += doc['sections'][k]['vector']
#               num_elements = doc['sections'][k]['num_elements']
#               num_total_elements += num_elements

#       if num_total_elements > 0:
#           accum_vector /= num_total_elements
#       return accum_vector

#   @classmethod
#   def compute_mean_vector(use, doc):
#       return np.ones(shape=(200, ))

# Database.register_method(Word2Vec)