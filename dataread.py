# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:15:18 2019

@author: hui.wang
"""

import gensim
import numpy as np
from sklearn.cluster import KMeans
import operator
from nltk.tokenize import word_tokenize     
import numpy as np  
import random
from nltk.corpus import stopwords
import nltk
import pickle
import copy
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from lemma import lemmatizer

class Read_data:
    def __init__(self, review, aspect,vocab,stopword=None):
        self.reviews = review[aspect]  #already clean symbol, lower the letter and sentence tokenize
        self.aspect = aspect
        self.vocab = vocab
        if stopword == True:
            stop_word = stopwords.words('english')
            self.stopword = [i for i in stop_word if i not in ['no','not','nor','t']]
        else:
            self.stopword = []
            
    def sentence_convert(self):
        review_in_indice = []
        for sentence in self.reviews:
            indices = []
            all_word = [i for i in lemmatizer().lemmatize_sentence(sentence) if i not in self.stopword]
            all_word = [i for i in all_word if not i.isdigit()]
#            all_word=[lmtzr.lemmatize(i) for i in all_word]
            all_word = [i for i in all_word if i in self.vocab]
            if not all_word:
                all_word.append('<pad>')
            for word in all_word:
                indices.append(self.vocab[word])
            review_in_indice.append(indices)
        return review_in_indice, max(len(k) for k in review_in_indice)
    
    def link_grade(self,index,grade):
        review = self.reviews
        review_index = {}
        for k in range(len(review)):
            review_index.update({review[k]:index[k]})
        review_grade = {}
        for key, value in grade.items():
            if key in review:
                review_grade.update({key:value})
        index_grade = {tuple(review_index[k]):v for k,v in review_grade.items()}
        opposite_pool = {'positive':[],'negative':[]}
        for key, value in index_grade.items():
            if value > 3:
                opposite_pool['positive'].append(key)
            elif value < 3:
                opposite_pool['negative'].append(key)
        return opposite_pool 
                
    
    
    
    
    
    
    
    