import numpy
from keras.preprocessing.sequence import pad_sequences

def getWordVectorsSum(Config, text, w2vModel):
    ndim = Config["w2v"]["ndim"]
    vec = numpy.zeros(ndim).reshape((1, ndim))
    count = 0.
    for word in text.split():
        try:
            vec += w2vModel[word].reshape((1, ndim))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def getWordVectorsMatrix(Config, text, indexer, w2vModel):
    ndim = Config["w2v"]["ndim"]
    maxWords = 300000
    sequence = pad_sequences(indexer.texts_to_sequences([text]), maxlen=int(Config["tokenization"]["maxseqlen"]))
    matrix = numpy.zeros((maxWords, ndim))
    word_index = indexer.word_index
    nf = 0
    for word, i in word_index.items():
        if i < maxWords:
            try:
                embedding_vector = w2vModel[word]
            except KeyError:
                continue
            if embedding_vector is not None:
                matrix[i] = embedding_vector
    return sequence


def getCharVectors(Config, text):
    chDict = getDictionary()
    maxLen = int(Config["tokenization"]["maxcharsseqlen"])
    str2ind = numpy.zeros(maxLen, dtype='int64')
    strLen = min(len(text), maxLen)
    for i in range(1, strLen + 1):
        c = text[-i]
        if c in chDict:
            str2ind[i - 1] = chDict[c]
    return str2ind.reshape(1, maxLen)

def getDictionary():
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
    charDict = {}
    for idx, char in enumerate(alphabet):
        charDict[char] = idx + 1
    return charDict

def vectorize(Config, text, vectorizer):
    return vectorizer.transform([text])
