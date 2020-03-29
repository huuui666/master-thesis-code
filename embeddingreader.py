# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:55:06 2019

@author: hui.wang
"""

import numpy as np
import gensim
import random

class Embeddingreader:# build a dict for all word in review and matrix for all word 
    def __init__(self, source,vocab):
        self.embeddings = {}
#        self.emb_matrix=[]
        model = gensim.models.Word2Vec.load(source)
    
        
        for word in model.wv.vocab:
            self.embeddings[word] = list(model.wv[word])
#            self.emb_matrix.append(list(model.wv[word]))
            
#        self.emb_matrix=np.asarray(self.emb_matrix)    
        self.emb_dim = len(random.choice(list(self.embeddings.values())))# word embedding vector dimension
        self.embedding_matrix = np.zeros((len(vocab), self.emb_dim))
        for word, i in vocab.items():
            embedding_vector = self.embeddings.get(word)
            if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector