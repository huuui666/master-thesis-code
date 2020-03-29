# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:22:05 2019

@author: huuui
"""
from nltk import pos_tag  
import numpy as np
from lemma import lemmatizer



class seed_word_extraction:
    def __init__(self,review,vocab,threshold=1):
        self.review = review
        self.vocab = vocab
        self.threshold = threshold
        
        
    def convert_vector_back(self,batch,vocab):
        converted_sentence = []
        for i in range(len(batch)):
            if batch[i] > 0:
                converted_sentence.extend([word for word, index in vocab.items() if index==batch[i]])
        return converted_sentence    
      
    def most_common(self,data):
        word_count = {i:data.count(i) for i in data}
        count = max(list(word_count.values()))
        output = []
        for key, value in word_count.items():
            if value == count:
                output.append(key)
        return output    
            
    def cosin(self,a,b):
        dot = np.dot(a,b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        if norma and norma !=0:
            cos=dot/(norma*normb)
        else:
            cos=0.
        return (cos)   

        
        
    def build_speech(self):
        part_of_speech_vocab = {}
        for i in range(len(self.review)):
            for word, pos in pos_tag(lemmatizer().lemmatize_sentence(self.review[i])):
                if word not in part_of_speech_vocab:
                    part_of_speech_vocab.update({word:[pos]})
                else:
                    part_of_speech_vocab[word].append(pos)
        return part_of_speech_vocab

    def  new_sentiment_word(self,weight,probability,sentence_vector,part_of_speech_vocab,emb, RB=False):
        vocab = self.vocab
        positive_large_seed = []
        negative_large_seed = []
        for i in range(len(weight[0])):
            if probability[0][i][0] > 0.5:
                max_weight=max(weight[0][i])
                index=list(weight[0][i]).index(max_weight)
                positive_large_seed.append(sentence_vector[i][index])
            elif probability[0][i][1] > 0.5:
                max_weight = max(weight[0][i])
                index = list(weight[0][i]).index(max_weight)
                negative_large_seed.append(sentence_vector[i][index])

        positive_output = self.convert_vector_back(positive_large_seed,self.vocab)  
        positive_count = {i:positive_output.count(i) for i in list(set(positive_output))}
        
        negative_output = self.convert_vector_back(negative_large_seed,self.vocab)  
        negative_count = {i:negative_output.count(i) for i in list(set(negative_output))}
        
        sent_pos = {}
        sent_neg = {}
        if RB == False:
            for word in positive_count:
                if 'JJ'  in part_of_speech_vocab[word]:
                    if positive_count[word] > self.threshold:
                        sent_pos.update({word:positive_count[word]})
                
            for word in negative_count:
                if 'JJ' in part_of_speech_vocab[word] :
                    if negative_count[word] > self.threshold:
                        sent_neg.update({word:negative_count[word]})
         
        else:
            for word in positive_count:
                if 'JJ' or 'RB' in part_of_speech_vocab[word]:
                    if positive_count[word] > self.threshold:
                        sent_pos.update({word:positive_count[word]})
                
            for word in negative_count:
                if 'JJ' or 'RB' in part_of_speech_vocab[word] :
                    if negative_count[word] > self.threshold:
                        sent_neg.update({word:negative_count[word]})         
        
        sent_pos = {i:self.cosin(emb.embedding_matrix[vocab[i]],emb.embedding_matrix[vocab['great']])  
          for i in sent_pos if self.cosin(emb.embedding_matrix[vocab[i]],emb.embedding_matrix[vocab['great']])>
          self.cosin(emb.embedding_matrix[vocab[i]],emb.embedding_matrix[vocab['terrible']])}
        sent_neg = {i:self.cosin(emb.embedding_matrix[vocab[i]],emb.embedding_matrix[vocab['terrible']]) 
         for i in sent_neg if self.cosin(emb.embedding_matrix[vocab[i]],emb.embedding_matrix[vocab['great']])<
          self.cosin(emb.embedding_matrix[vocab[i]],emb.embedding_matrix[vocab['terrible']])}
        
        
        return sent_pos,sent_neg