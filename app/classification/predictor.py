import os
import json
import gensim
import pickle
import torch
import nltk
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import subprocess
from subprocess import run, PIPE
from keras.models import load_model
from keras import backend as K
from sklearn.externals import joblib
from classification.bertClassifier import BertForMultiLabelSequenceClassification, getBertModel, bertPredicts

from classification.normalization import ArabicNormalizer
from classification.data import *


class Predictor(object):
    def __init__(self):
        self.Config = {}
        self.error = "OK"
        self.ready = False
        curDir = os.path.dirname(__file__)
        resPath = curDir + "/../resources/"
        if not os.path.isfile(resPath + "config.json"):
            self.error = "Missing configuration file"
            return
        with open(resPath + "config.json") as json_file:
            self.Config = json.load(json_file)
        json_file.close()
        if "models" not in self.Config:
            self.error = "Wrong configuration."
            return
        self.device = "cpu"
        self.models = {}
        self.modelPredictions = {}
        self.ranks = {}
        self.predictions = []
        self.rankThreshold = 0.5
        self.diffThreshold = 10
        if "labels" not in self.Config or not os.path.isfile(resPath + self.Config["labels"]):
            self.error = "Missing file of labels."
            return
        else:
            with open(resPath + self.Config["labels"], "r", encoding="utf-8") as file:
                labels = file.read()
            file.close()
        self.labels = labels.split(",")
        if "ptBertModel" in self.Config:
            if not os.path.isfile(resPath + self.Config["ptBertModel"]):
                self.error = "Missing resource: " + self.Config["ptBertModel"]
                return
            self.ptBertModel = resPath + self.Config["ptBertModel"]
        if "vocabPath" in self.Config:
            if not os.path.isfile(resPath + self.Config["vocabPath"]):
                self.error = "Missing resource: " + self.Config["vocabPath"]
                return
            self.vocabPath = resPath + self.Config["vocabPath"]
        if "consolidatedRank" in self.Config:
            try:
                self.rankThreshold = float(self.Config["consolidatedRank"])
                if self.rankThreshold == 0:
                    self.rankThreshold = 0.5
            except ValueError:
                self.rankThreshold = 0.5
        isFirstKeras = True
        for key, val in self.Config["models"].items():
            if not os.path.isfile(resPath + val["modelPath"]):
                self.error = "Missing resource: " + val["modelPath"]
                return
            modelPath = resPath + val["modelPath"]
            if val["modelType"] == "keras":
                if isFirstKeras:
                    config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                        inter_op_parallelism_threads=4, allow_soft_placement=True,
                                        device_count={'CPU': 4, 'GPU': 0})
                    session = tf.Session(config=config)
                    self.graph = session.graph
                    set_session(session)
                    isFirstKeras = False
                self.models[key] = load_model(modelPath)
                self.ranks[key] = 0.5
            elif val["modelType"] == "skl":
                self.models[key] = joblib.load(modelPath)
                self.ranks[key] = 1.0
            elif val["modelType"] == "torch":
                self.models[key] = getBertModel(modelPath, self.ptBertModel, len(self.labels), self.device)
                self.ranks[key] = 0.5
            else:
                self.error = "Unsupported model's type."
                return
            if "handleType" not in val or val["handleType"] not in [
                        "wordVectorsSum", "wordVectorsMatrix", "charVectors", "vectorize"]:
                self.error = "Unknown or unsupported type of input data handling."
                return
            if "rankThreshold" in val:
                try:
                    self.ranks[key] = float(val["rankThreshold"])
                    if self.ranks[key] == 0:
                        if val["modelType"] != "skl":
                            self.ranks[key] = 0.5
                        else:
                            self.ranks[key] = 1.0
                except ValueError:
                    if val["modelType"] != "skl":
                        self.ranks[key] = 0.5
                    else:
                        self.ranks[key] = 1.0
        if  "w2v" in self.Config:
            if not os.path.isfile(resPath + self.Config["w2v"]["modelPath"]):
                self.error = "Missing resource: " + self.Config["w2v"]["modelPath"]
                return
            #self.models["w2v"] = gensim.models.KeyedVectors.load_word2vec_format(
            #                            resPath + self.Config["w2v"]["modelPath"])
            with open(resPath + self.Config["w2v"]["modelPath"], 'rb') as f:
                    self.models["w2v"] = pickle.load(f)
            f.close()
        if "indexer" in self.Config:
            if not os.path.isfile(resPath + self.Config["indexer"]):
                self.error = "Missing resource: " + self.Config["indexer"]
                return
            with open(resPath + self.Config["indexer"], 'rb') as handle:
                self.models["indexer"] = pickle.load(handle)
        if "vectorizer" in self.Config:
            if not os.path.isfile(resPath + self.Config["vectorizer"]):
                self.error = "Missing resource: " + self.Config["vectorizer"]
                return
            with open(resPath + self.Config["vectorizer"], 'rb') as handle:
                self.models["vectorizer"] = pickle.load(handle)
        if "ptBertModel" in self.Config:
            if not os.path.isfile(resPath + self.Config["ptBertModel"]):
                self.error = "Missing resource: " + self.Config["ptBertModel"]
                return
        if "rttaggerpath" in self.Config["tokenization"] and self.Config["tokenization"]["actualtoks"] == "yes":
            if not os.path.isfile(resPath +  self.Config["tokenization"]["rttaggerpath"]):
                self.error = "Missing resource: " + self.Config["tokenization"]["rttaggerpath"]
                return
            taggerPath = resPath +  self.Config["tokenization"]["rttaggerpath"]
            self.jar = subprocess.Popen(
                        'java -Xmx2g -jar ' + taggerPath + ' "' + self.Config["tokenization"]["expos"] + '"',
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                            encoding="utf-8")
        if self.Config["tokenization"]["stopwords"] == "yes":
            self.stopWords = set(nltk.corpus.stopwords.words('arabic'))
        else:
            self.stopWords = set()
        if self.Config["tokenization"]["normalization"] == "yes":
            self.normalizer = ArabicNormalizer()
        self.ready = True

    def isReady(self):
        return self.ready

    def predict(self, text):
        if self.error != "OK":
            return self.error, "Not classified"
        self.predictions = [0] * len(self.labels)
        text = self.preprocess(text)
        for key, val in self.Config["models"].items():
            self.askModel(key, text)
        self.collector()
        reply = []
        for i in range(len(self.labels)):
            if self.predictions[i] == 1:
                reply.append(self.labels[i])
        if len(reply) == 0:
            reply.append("Not classified")
        return self.error, ",".join(reply)

    def preprocess(self, text):
        if self.Config["tokenization"]["actualtoks"] == "yes":
            text = text.replace("\n", " ").strip()
            self.jar.stdin.write(text + '\n')
            self.jar.stdin.flush()
            text = self.jar.stdout.readline()
        words = [w for w in text.strip().split() if w not in self.stopWords]
        words = [w for w in words if w not in self.Config["tokenization"]["extrawords"]]
        if self.Config["tokenization"]["normalization"] == "yes":
            words = [self.normalizer.normalize(w) for w in words]
        text = " ".join(words)
        return text

    def askModel(self, key, text):
        input = self.prepareData(key, text)
        if self.Config["models"][key]["modelType"] == "keras":
            with self.graph.as_default():
                self.modelPredictions[key] = self.models[key].predict(input)
        elif self.Config["models"][key]["modelType"] != "torch":
            self.modelPredictions[key] = self.models[key].predict(input)
        else:
            self.modelPredictions[key] = bertPredicts(self.models[key], self.vocabPath, self.labels,
                             min(self.Config["tokenization"]["maxseqlen"],512), text, self.device)

    def prepareData(self, key, text):
        hType = self.Config["models"][key]["handleType"]
        if hType == "wordVectorsSum":
            return getWordVectorsSum(self.Config, text, self.models["w2v"])
        elif hType == "wordVectorsMatrix":
            return getWordVectorsMatrix(self.Config, text, self.models["indexer"], self.models["w2v"])
        elif hType == "charVectors":
            return getCharVectors(self.Config, text)
        elif hType == "vectorize":
            return vectorize(self.Config, text, self.models["vectorizer"])
        else:
            return None

    def collector(self):
        qModels = len(self.modelPredictions)
        for key, res in self.modelPredictions.items():
            isFound = False
            for i in range(len(self.labels)):
                if res[0][i] >= self.ranks[key]:
                    self.predictions[i] += 1
                    isFound = True
        for i in range(len(self.predictions)):
            if self.predictions[i] >= qModels * self.rankThreshold:
                self.predictions[i] = 1
            else:
                self.predictions[i] = 0
