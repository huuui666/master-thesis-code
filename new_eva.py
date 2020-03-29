# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:49:19 2020

@author: huuui
"""
from dataread import Read_data
from keras.preprocessing import sequence
import keras.backend as K
import numpy as np
import copy 

class evaluate:
    def __init__(self, aspect, golden, test_review, vocab,model):
        self.aspect = aspect
        self.golden = golden
        self.test_review = test_review
        self.vocab = vocab
        self.model = model
        
    
    def sentiment_accuracy(self, boundary = 0.5):
        aspect_review = []
        aspect_polarity = []
        for i in range(len(self.test_review)):
            if self.aspect in self.golden[i]:
                if self.golden[i][self.aspect] == 'positive' or self.golden[i][self.aspect] == 'negative':
                    aspect_review.append(self.test_review[i])
                    aspect_polarity.append(self.golden[i][self.aspect])
        
        test_r1, test_r2 = Read_data({self.aspect:aspect_review}, self.aspect,self.vocab,stopword=True).sentence_convert()   
        test_train_x = sequence.pad_sequences(test_r1, maxlen= self.model.layers[0].output.get_shape()[1])
        class_output_model=K.function([self.model.get_layer('sentence_input').input],[self.model.get_layer('class').output])
        probability_test=np.asarray(class_output_model([test_train_x]))    
        predict_label=[]
        for i in range(probability_test[0].shape[0]):
            if probability_test[0][i][0]> boundary:
                predict_label.append('positive')
            else:
                predict_label.append('negative')
                
        accuracy = 0
        for i in range(len(predict_label)):
            if predict_label[i] == aspect_polarity[i]:
                accuracy += 1
        TP_pos = 0 
        FP_pos = 0
        FN_pos = 0
        TP_neg = 0 
        FP_neg = 0
        FN_neg = 0
        for i in range(len(predict_label)):
            if  predict_label[i] == aspect_polarity[i] and predict_label[i] == 'positive':   
                TP_pos += 1            
            elif   predict_label[i] != aspect_polarity[i] and predict_label[i] == 'positive':
                FP_pos += 1
            elif   predict_label[i] != aspect_polarity[i] and aspect_polarity[i] == 'positive':
                FN_pos += 1
        for i in range(len(predict_label)):        
            if  predict_label[i] == aspect_polarity[i] and predict_label[i] == 'negative':   
                TP_neg += 1            
            elif   predict_label[i] != aspect_polarity[i] and predict_label[i] == 'negative':
                FP_neg += 1
            elif   predict_label[i] != aspect_polarity[i] and aspect_polarity[i] == 'negative':
                FN_neg += 1
                
        precision_pos = TP_pos/ (TP_pos + FP_pos)
        recall_pos = TP_pos / (TP_pos + FN_pos)
        precision_neg = TP_neg  / (TP_neg  + FP_neg )
        recall_neg = TP_neg  / (TP_neg  + FN_neg )
        F1_pos = 2*precision_pos*recall_pos / (precision_pos+recall_pos)
        F1_neg = 2*precision_neg*recall_neg / (precision_neg+recall_neg)
        
        return {'accuracy':accuracy/len(predict_label) , 'F1_pos':F1_pos, 'F1_neg':F1_neg}
    
    def combined_accuracy(self, combined_prediction):
        aspect_review = []
        aspect_polarity = []
        for i in range(len(self.test_review)):
            if self.aspect in self.golden[i]:
                if self.golden[i][self.aspect] == 'positive' or self.golden[i][self.aspect] == 'negative':
                    aspect_review.append(self.test_review[i])
                    aspect_polarity.append(self.golden[i][self.aspect])
        
        TN_pos  = 0
        TN_neg  = 0 
        accuracy = 0
        for sentence, polarity in combined_prediction.items():
            if sentence in aspect_review and polarity == aspect_polarity[aspect_review.index(sentence)]:
                accuracy += 1
                if polarity == 'positive':
                    TN_pos +=1
                else:
                    TN_neg +=1
        prediction_polarity = [i for i in combined_prediction.values()]
        pos_prediction =     prediction_polarity.count('positive')
        neg_prediction =     prediction_polarity.count('negative')
       
        pos_true = aspect_polarity.count('positive')
        neg_true = aspect_polarity.count('negative')
        
        output =  {'pos':[TN_pos,pos_prediction,pos_true],'neg':[TN_neg,neg_prediction,neg_true]}
        return output, [accuracy,len(combined_prediction)]
       
       
       
       
       
       
       
       
                      
                


    
