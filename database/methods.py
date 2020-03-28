from utils import clean_text
from database import Database
import numpy as np

class SpacyEmbeddings:
    NAME = 'SpacyEmbeddings'
    NUM_DIMENSIONS = 200

    @staticmethod
    def init():
        import scispacy
        import spacy

        SpacyEmbeddings.NLP = spacy.load("en_core_sci_lg")

    @staticmethod
    def get_mean_vector(sections_vector):
        accum_vector = np.zeros(shape=(SpacyEmbeddings.NUM_DIMENSIONS, ))
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

    @staticmethod
    def get_mean_vector_from_section(sections_vector, section, translate_lut=None):
        accum_vector = np.zeros(shape=(SpacyEmbeddings.NUM_DIMENSIONS, ))
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

    @staticmethod
    def compute_mean_vector(raw_or_clean_doc):
        sections_vector = {}
        for k, v_text in raw_or_clean_doc['sections'].items():
            c_text = clean_text(v_text)
            if c_text is None:
                sections_vector[k] = None
                continue

            doc_spacy = SpacyEmbeddings.NLP(c_text)
            sections_vector[k] = {
                'vector': doc_spacy.vector,
                'num_elements': sum([token.has_vector for token in doc_spacy])
            }
        return sections_vector

Database.register_method(SpacyEmbeddings)

# class Word2Vec:
#   NAME = 'Word2Vec'
#   NUM_DIMENSIONS = 200

#   @staticmethod
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

#   @staticmethod
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

#   @staticmethod
#   def compute_mean_vector(use, doc):
#       return np.ones(shape=(200, ))

# Database.register_method(Word2Vec)