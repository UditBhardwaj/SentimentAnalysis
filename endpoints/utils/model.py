import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import model_from_json
import pickle
from tqdm import tqdm

nltk.download('stopwords')
import os

PATH = os.path.abspath(os.getcwd())


class Preprocessor:
    def __init__(self):

        self.regex_tokenizer = nltk.RegexpTokenizer(r"[^_\W]+")
        # self.english_stopwords = set(stopwords.words('english'))
        self.vocab_tokenizer_path = PATH + '/model_resources/tokenizer.pickle'
        self.vocabulary_tokenizer = None
        with open(self.vocab_tokenizer_path, 'rb') as tokenizer:
            self.vocabulary_tokenizer = pickle.load(tokenizer)

    def __clean(self, sentances):
        cleaned_sentances = []
        for sentance in sentances:
            sentance = sentance.lower()

            words = self.regex_tokenizer.tokenize(sentance)
            # words = list(filter(lambda word: word not in self.english_stopwords, words))
            cleaned_sentances.append(' '.join(words))
        return cleaned_sentances

    def process(self, sentances):
        cleaned_sentances = self.__clean(sentances)
        sequences = self.vocabulary_tokenizer.texts_to_sequences(cleaned_sentances)
        preprocessed_data = pad_sequences(sequences, maxlen=300)
        return preprocessed_data


class SentimentAnalysisModel:
    def __init__(self):
        self.model_path = PATH + '/model_resources/modelA91.json'
        self.model_weights_path = PATH + '/model_resources/modelA91.h5'
        self.preprocessor = Preprocessor()

        with open(self.model_path, 'r') as model_json:
            model_json = model_json.read()

        self.model = model_from_json(model_json)
        self.model.load_weights(self.model_weights_path)

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy', Precision(), Recall()])

    def classify(self, sentances):
        preprocessed_data = self.preprocessor.process(sentances)
        predictions = self.model.predict(preprocessed_data)
        pred = [str(p[0]) for p in predictions]
        return pred

