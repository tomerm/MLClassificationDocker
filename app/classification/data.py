import numpy
from keras.preprocessing.sequence import pad_sequences

def get_word_vectors_sum(Config, text, w2v_model):
    ndim = Config["w2v"]["ndim"]
    vec = numpy.zeros(ndim).reshape((1, ndim))
    count = 0.
    for word in text.split():
        try:
            vec += w2v_model[word].reshape((1, ndim))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_word_vectors_matrix(Config, text, indexer, w2v_model):
    ndim = Config["w2v"]["ndim"]
    maxWords = 300000
    sequence = pad_sequences(indexer.texts_to_sequences([text]), maxlen=int(Config["tokenization"]["maxseqlen"]))
    matrix = numpy.zeros((maxWords, ndim))
    word_index = indexer.word_index
    nf = 0
    for word, i in word_index.items():
        if i < maxWords:
            try:
                embedding_vector = w2v_model[word]
            except KeyError:
                continue
            if embedding_vector is not None:
                matrix[i] = embedding_vector
    return sequence


def get_char_vectors(Config, text):
    chDict = get_dictionary()
    maxLen = int(Config["tokenization"]["maxcharsseqlen"])
    str2ind = numpy.zeros(maxLen, dtype='int64')
    strLen = min(len(text), maxLen)
    for i in range(1, strLen + 1):
        c = text[-i]
        if c in chDict:
            str2ind[i - 1] = chDict[c]
    return str2ind.reshape(1, maxLen)

def get_dictionary():
    start = ord('\u0600')
    end = ord('\u06ff')
    alphabet = ''
    for i in range(start, end + 1):
        ch = chr(i)
        alphabet = alphabet + ch
    start = ord('\u0750')
    end = ord('\u077f')
    for i in range(start, end + 1):
        ch = chr(i)
        alphabet = alphabet + ch
    char_dict = {}
    for idx, char in enumerate(alphabet):
        char_dict[char] = idx + 1
    return char_dict

def vectorize(Config, text, vectorizer):
    return vectorizer.transform([text])
