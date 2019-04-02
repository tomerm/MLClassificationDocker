import re
from nltk.stem.util import suffix_replace

class ArabicNormalizer(object):
    __vocalization = re.compile(r'[\u064b-\u064c-\u064d-\u064e-\u064f-\u0650-\u0651-\u0652]')
    __kasheeda = re.compile(r'[\u0640]') # tatweel/kasheeda
    __arabic_punctuation_marks = re.compile(r'[\u060C-\u061B-\u061F]')
    __last_hamzat = ('\u0623', '\u0625', '\u0622', '\u0624', '\u0626')
    __initial_hamzat = re.compile(r'^[\u0622\u0623\u0625]')
    __waw_hamza = re.compile(r'[\u0624]')
    __yeh_hamza = re.compile(r'[\u0626]')
    __alefat = re.compile(r'[\u0623\u0622\u0625]')

    def normalize(self, token):
        # strip diacritics
        token = self.__vocalization.sub('', token)
        #strip kasheeda
        token = self.__kasheeda.sub('', token)
        # strip punctuation marks
        token = self.__arabic_punctuation_marks.sub('', token)
        # normalize last hamza
        for hamza in self.__last_hamzat:
            if token.endswith(hamza):
                token = suffix_replace(token, hamza, '\u0621')
                break
        # normalize other hamzat
        token = self.__initial_hamzat.sub('\u0627', token)
        token = self.__waw_hamza.sub('\u0648', token)
        token = self.__yeh_hamza.sub('\u064a', token)
        token = self.__alefat.sub('\u0627', token)
        return token
