# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:38:28 2019

@author: hui.wang
"""

import nltk
from nltk.tokenize import word_tokenize 
from nltk import pos_tag   
#nltk.download("wordnet")
#nltk.download("punkt")
#nltk.download("maxent_treebank_pos_tagger")
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from nltk.corpus import wordnet



class lemmatizer:
    
    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
   
    def lemmatize_sentence(self,sentence):
        res = []
        
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lmtzr.lemmatize(word, pos=wordnet_pos))

        return res