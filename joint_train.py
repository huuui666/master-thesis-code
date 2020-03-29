# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:13:39 2019

@author: hui.wang
"""

from Detection import detection
import embeddingreader
import pickle
import statistics as st
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from dataread import Read_data
from keras.preprocessing import sequence
import copy
from sentiment_train import Train
from lemma import lemmatizer

class all_for_one:
    def __init__(self,review,vocab,emb,threshold,sentiment_seedword,centroid=None):
        self.review = review
        self.vocab = vocab
        self.emb = emb
        self.threshold = threshold
        self.sentiment_seedword =sentiment_seedword
        self.centroid = centroid    


    def sentence_average(self,data):
        emb = self.emb
        sentence_average = np.empty((len(data),emb.emb_dim))
        i = -1
        for sentence in data:
            i+= 1
            sentence_vector = np.zeros((1,emb.emb_dim))
            for word in sentence:
                sentence_vector+= emb.embedding_matrix[word]
            sentence_average[i]= sentence_vector/len(sentence)      
        return sentence_average
    
    def cosin(self,a,b):
        dot = np.dot(a,b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        if norma and norma !=0:
            cos = dot/(norma*normb)
        else:
            cos = 0.
        return (cos)    
    
    def find_highest(self,data ,percentage):
        order = math.ceil(len(data)*percentage)
        ordered_data = sorted(data,reverse=True)
        
        return ordered_data[order]
        
    def init_detect(self,similarity_matrix,overall_dict):
        review = self.review
        t1,t2,t3,t4 = detection(self.emb.embedding_matrix,similarity_matrix,overall_dict,self.vocab,review,emb_dim=self.emb.emb_dim,k=14).score(input_centeriod=self.centroid) #200 8
        value_dict = {aspect:[t1[j][aspect] for j in t1] for aspect in self.threshold }
        detected_review = {aspect:[review[i] for i in range(len(review)) if value_dict[aspect][i]>self.threshold[aspect]] for aspect in self.threshold}
        median = {aspect:st.median([i for i in value_dict[aspect] if i >self.threshold[aspect]]) for aspect in self.threshold}
        high_detected_review = {aspect:[review[i] for i in range(len(review)) if value_dict[aspect][i]>median[aspect]] for aspect in self.threshold}
        
#        return detected_review,t1,t2,t3
        return high_detected_review,t1,t2,t3

    def sentiment_split(self,review1): # high_detected_review as review1, all in sentence since SemEval already all in sentence
        stopword = stopwords.words('english')
        stopword = [i for i in stopword if i not in ['no','not','nor','t']]
        review_in_index= {}
        for aspect in review1:
            review11 = [lemmatizer().lemmatize_sentence(i) for i in review1[aspect]]
            review11 = [[i for i in j if i not in stopword] for j in review11] #clean stopword
            review11 = [[i for i in j if not i.isdigit()] for j in review11] # clean number
#            review11=[[lmtzr.lemmatize(i) for i in j] for j in review11]     #lemma     
            review11 = [[i for i in j if i in self.vocab] for j in review11]   
            index = [[self.vocab[i] for i in j] for j in review11]
            for i in index:
                if not i:
                    i.append(0)
            review_in_index.update({aspect:index})
        
        sentence_average = {}
        for aspect in review_in_index:
            sentence_average.update({aspect:self.sentence_average(review_in_index[aspect])}) #average sentence embeddings
        
        positive_seed = {aspect:[[self.vocab[i]] for i in self.sentiment_seedword[aspect]['positive']] for aspect in self.sentiment_seedword}
        negative_seed = {aspect:[[self.vocab[i]] for i in self.sentiment_seedword[aspect]['negative']] for aspect in self.sentiment_seedword}
        ps = {aspect:self.sentence_average(positive_seed[aspect]) for aspect in self.sentiment_seedword}
        ns = {aspect:self.sentence_average(negative_seed[aspect]) for aspect in self.sentiment_seedword}
        ps1 = {aspect:ps[aspect]+self.emb.embedding_matrix[self.vocab[aspect]]  for aspect in ps}
        ns1 = {aspect:ns[aspect]+self.emb.embedding_matrix[self.vocab[aspect]]  for aspect in ns}
        
        
        
        pos_sim = {aspect:[] for aspect in sentence_average}
        for aspect in sentence_average:
            for i in range(len(sentence_average[aspect])):
                aux_sim = []
                for j in range(ps1[aspect].shape[0]):
                    aux_sim.append(self.cosin(sentence_average[aspect][i],ps1[aspect][j]))
                pos_sim[aspect].append(max(aux_sim))
        neg_sim = {aspect:[] for aspect in sentence_average}
        for aspect in sentence_average:
            for i in range(len(sentence_average[aspect])):
                aux_sim = []
                for j in range(ns1[aspect].shape[0]):
                    aux_sim.append(self.cosin(sentence_average[aspect][i],ns1[aspect][j]))
                neg_sim[aspect].append(max(aux_sim))        
        
        aspect_pos = {aspect:[] for aspect in review1}
        aspect_neg = {aspect:[] for aspect in review1}
        for aspect in review1:
            for i in range(len(review1[aspect])):
                if pos_sim[aspect][i]>neg_sim[aspect][i]:
                    aspect_pos[aspect].append(review1[aspect][i])
                elif neg_sim[aspect][i]>pos_sim[aspect][i]:
                    aspect_neg[aspect].append(review1[aspect][i])
        
        output_review = {aspect:aspect_pos[aspect]+aspect_neg[aspect] for aspect in aspect_pos}
        grade = {aspect:{} for aspect in aspect_pos}
        for aspect in grade:
            for i in range(len(aspect_pos[aspect])):
                grade[aspect].update({aspect_pos[aspect][i]:5})
            for i in range(len(aspect_neg[aspect])):
                grade[aspect].update({aspect_neg[aspect][i]:1})
        
        
#        grade={aspect:len(aspect_pos[aspect])*[5]+len(aspect_neg[aspect])*[1] for aspect in aspect_pos}
        
        
        return output_review, grade   
    
    
    def iterative_train(self,output_review,grade,args,t2,similarity_matrix,overall_dict):
        new_detected_review = {aspect:[] for aspect in self.sentiment_seedword}
        new_grade = {aspect:{} for aspect in self.sentiment_seedword}
        all_cos_pos = {}
        all_cos_neg = {}
        for aspect in self.sentiment_seedword:
            data = Read_data(output_review, aspect,self.vocab,stopword=True)
            review_in_index,review_index_length = data.sentence_convert()
            train_x = sequence.pad_sequences(review_in_index, maxlen= review_index_length)
            train_x_copy = copy.deepcopy(train_x)
            opposite_pool = data.link_grade(train_x,grade[aspect])
            probability,sentence_vector,weight = Train(self.sentiment_seedword[aspect],train_x,train_x_copy, review_index_length,
                        self.vocab,self.emb,args,opposite_pool).train_model()
            
            positive_sentence_vector = 0 
            negative_sentence_vector = 0
            p_number = 0
            n_number = 0
          
            for i in range(probability[0].shape[0]):
                if probability[0][i][0] > max(0.9,self.find_highest([probability[0][i][0] for i in range(probability[0].shape[0])],0.1)):
                    p_number+= 1
                    positive_sentence_vector+= sentence_vector[0][i]
                    
                elif probability[0][i][1] > max(0.9,self.find_highest([probability[0][i][1] for i in range(probability[0].shape[0])],0.1)):# 0.5 for price, ambience
                    n_number+= 1
                    negative_sentence_vector+= sentence_vector[0][i]
            average_positive = positive_sentence_vector/p_number
            average_negative = negative_sentence_vector/n_number
            
#            centriod, sentence_average=detection(self.emb.embedding_matrix,similarity_matrix,overall_dict,
#                                                 self.vocab,self.review,emb_dim=200,k=8).cluster_init()# just get all reviews'average
            all_review_index, length = Read_data({'all':self.review}, 'all',self.vocab,stopword=True).sentence_convert()
            sentence_average = self.sentence_average(all_review_index)
            
            cosin_pos = [self.cosin(sentence_average[i],average_positive) for i in range(len(sentence_average))]
            cosin_neg = [self.cosin(sentence_average[i],average_negative) for i in range(len(sentence_average))]
            for i in cosin_pos:
                if i < 0:
                    i = 0
            for i in cosin_neg:
                if i < 0:
                    i = 0
            aspect_soft = [i[aspect] for i in t2.values()]
            for i in range(len(sentence_average)):
                if float(aspect_soft[i]) == 0:
                    cosin_pos[i] = 0
                    cosin_neg[i] = 0
            
            aspect_soft1 = []
            for i in t2:
                aspect_soft1.append(list(t2[i].values()))
            l2_norm = [np.linalg.norm(i) for i in aspect_soft1]
            for i in range(len(sentence_average)):
                if l2_norm[i] != 0:
                    aspect_soft[i] = float(aspect_soft[i])/l2_norm[i]
                else:
                    aspect_soft[i] = 0
                    
            finial_cosin_pos = [0.3*aspect_soft[i]+0.7*cosin_pos[i] for i in range(len(aspect_soft))]
            finial_cosin_neg = [0.3*aspect_soft[i]+0.7*cosin_neg[i] for i in range(len(aspect_soft))]
            
            new_aspect_pos = [self.review[i] for i in range(len(self.review)) if finial_cosin_pos[i] > self.find_highest(finial_cosin_pos,0.05) and
                            finial_cosin_pos[i] > finial_cosin_neg[i]]
            new_aspect_neg = [self.review[i] for i in range(len(self.review)) if finial_cosin_neg[i] > self.find_highest(finial_cosin_neg,0.05) and
                            finial_cosin_pos[i] < finial_cosin_neg[i]]
            
            new_detected_review[aspect] = new_aspect_pos+new_aspect_neg
            
            for i in new_aspect_pos:   
                new_grade[aspect].update({i:5})
            for i in new_aspect_neg:
                new_grade[aspect].update({i:1})
            
            all_cos_pos.update({aspect:finial_cosin_pos})
            all_cos_neg.update({aspect:finial_cosin_neg})
            
            print ('finish: ',aspect)   
        return new_detected_review, new_grade, all_cos_pos, all_cos_neg
                
            
            
            
    
     
    




