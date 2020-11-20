import re
import jsonlines
import json
from num2words import num2words
import pymorphy2
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora
from time import time
from fractions import Fraction
from decimal import *

wnl = WordNetLemmatizer()
#porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
t = time()
lemmas = []

def file_processing(file):
    tokenized_texts = []
    tokens = []
    with jsonlines.open(file, mode='r') as reader:
        for line in (list(reader)):
            title=line['title']
            text=line['text']
            text = re.sub(r'[^\w\s]', ' ', text).lower().replace('\n', ' ')
            for tok in nltk.word_tokenize(text):
                tokenized_texts.append(tok)

    for token in tokenized_texts:
       if token not in stop_words:

           if token.isnumeric() and not token.isdecimal() and not token.isdigit():
               #print("Vulgar fractions: ", token)
               tokens.append('')
               continue
           if token.isdecimal():
               tokens.append(convert_num2words(str(token)))
               continue
           tokens.append(token)

    #print(tokens)
    lemmatization(tokens)

def lemmatization(words):

    file = open('lemmas.txt', 'r+', encoding='utf-8')
    for word in words:
        lemma = wnl.lemmatize(word)
        lemmas.append(lemma)
    file.write(' '.join(lemmas) + '\n')
    print("Length of lemmas:", len(lemmas))
    vector_space_model(file)

def convert_num2words(number):
    return num2words(number)

def vector_space_model(data):
    model = gensim.models.Word2Vec(size=128,
                                   window=3,
                                   min_count=2,
                                   workers=8)

    model.build_vocab(LineSentence(data))
    print("Word2Vec vocabulary length:", len(model.wv.vocab))
    model.train(LineSentence(data), total_examples=model.corpus_count, epochs=5)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    model.save('wiki100k_model.model')

def load():
    model = Word2Vec.load('wiki100k_model.model')
    # model = gensim.models.KeyedVectors.load_word2vec_format('E:/ИЗ ИНЕТА/GoogleNews-vectors-negative300.bin', binary=True)
    model.train(LineSentence('lemmas.txt'), total_examples=model.corpus_count, epochs=15)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    model.save('wiki100k_model.model')
    print("Chanda VS Zambian", model.wv.similarity('chanda', 'zambian'))
    print("Chanda VS Football", model.wv.similarity('chanda', 'football'))
    print("Chanda VS Linguistics", model.wv.similarity('chanda', 'linguistics'))
    print("Sherwood VS Linguistics", model.wv.similarity('sherwood', 'linguistics'))
    print("Sherwood VS businessman", model.wv.most_similar(positive=['first'], negative=['second'], topn=3))
    return model

#print(file_processing('D:\Projects\Semantic Search\wiki100k.jsonl'))
#vector_space_model('lemmas.txt')
load()

# w1 = "dirty"
# print(model.wv.most_similar (positive=w1))
