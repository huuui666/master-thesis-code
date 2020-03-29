# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:19:50 2020

@author: huuui
"""
import numpy as np

class  centroid_generate:
    def __init__(self, probability, sentence_average, threshold=0.9):
        self.probability = probability[0]
        self.sentence_average = sentence_average
        self.threshold = threshold
        
    def generate(self):
        sentence_pos=0
        sentence_neg=0
        pos_n=0
        neg_n=0
        for i in range(len(self.probability)):
            if self.probability[i][0]>self.threshold:
                sentence_pos+=self.sentence_average[i]
                pos_n+=1
            elif self.probability[i][1]>self.threshold:
                sentence_neg+=self.sentence_average[i]
                neg_n+=1
    
        sp=sentence_pos/pos_n
        sn=sentence_neg/neg_n
        sp1=sp.reshape((1,200))
        sn1=sn.reshape((1,200))
        center=np.concatenate((sp1,sn1),axis=0)
        return center