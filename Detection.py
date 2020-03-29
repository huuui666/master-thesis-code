# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:16:53 2019

@author: hui.wang
"""

from gensim.matutils import softcossim
import gensim
from gensim.corpora import Dictionary
import numpy as np
from sklearn.cluster import KMeans
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize   
from tqdm import tqdm
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
import pickle
import re
import pandas as pd
#from googletrans import Translator
import time
from lemma import lemmatizer



class detection:
    def __init__(self,embedding_matrix,similarity_matrix,overall_dict,vocab,reviews=None,emb_dim=300,k=14,translate=None):
#        self.model=gensim.models.Word2Vec.load(embeddingsource)
        self.embedding_matrix = embedding_matrix
        self.similarity_matrix = similarity_matrix
        self.overall_dict = overall_dict
        self.vocab = vocab
        self.reviews = reviews
        self.seed_word = {'food':['food','beef','dessert','noodle','pizza','oyster','cooked','drink'],
                        'service':['service','staff','server','attitude','tip','request'],
                        'price':['price','inexpensive','overpriced','bill','money'],
                        'ambience':['ambience','atmosphere','comfortable','music']
                        }
#        self.seed_word = {'service':[['help'],['staff'],['service'],['communication'],['feedback']],
#           'specific_quality':[['wheel','broken'],['defective']],
#           'general_quality':[['quality'],['satisfied','bike']], #it is possible to take phrase, or even sentence as seed word
#           'price':[['price'],['discount']],
#           'delivery':[['deliver'],['delivery','time'],['wait','delivery']], 
#           'stock':[['stock'],['available']],
#           'website':[['website']],
#           'repair':[['repair'],['warranty']]
#                             }     
#                        
                        
        self.k = k
        self.emb_dim = emb_dim
        self.alpha={'service':0.7,'price':0.7,'food':0.7, 'ambience':0.7  }
        self.threshold={'service':0.0441,'price':0.069,'food':0.016,'ambience':0.055}
#        self.threshold={'service':0.083,'price':0.161,'specific_quality':0.104, 'general_quality':0.27 ,'repair':0.219,'stock':0.23,
#                        'website':0.265,'delivery':0.136}
#        self.alpha={'service':0.75,'price':0.75,'specific_quality':0.75, 'general_quality':0.75  ,'repair':0.75,'stock':0.75,'website':0.75,'delivery':0.75}
        self.translate = translate
       
#    def translate_review(self, reviews,translate):
#        if translate == None:
#            return reviews
#        else:
#            output = []
#            for review in tqdm(reviews):
#                if review:
#                    output.append(Translator().translate(review,dest='en').text)
#                    time.sleep(1)
#            return output        
#	
     
        
    def transform_review(self,input_review=None):
        stopword = stopwords.words('english')
        if input_review == None:
            reviews1 = list(map(lambda x: sent_tokenize(x),self.reviews))
        else:
            reviews1 = list(map(lambda x: sent_tokenize(x),input_review))
        reviews2 = reviews1.copy()
        for review in range(len(reviews1)):
            for sentence in range(len(reviews1[review])):
                reviews2[review][sentence] = (re.sub(r'\W+', ' ',str(reviews1[review][sentence].lower() ))).strip()
                reviews2[review][sentence] = lemmatizer().lemmatize_sentence(reviews2[review][sentence])
                reviews2[review][sentence] = [j for j in reviews2[review][sentence] if j not in stopword]
                reviews2[review][sentence] = [j for j in reviews2[review][sentence] if not j.isdigit()]
#                reviews2[review][sentence]=[lmtzr.lemmatize(j) for j in reviews2[review][sentence]]
                reviews2[review][sentence] = [j for j in reviews2[review][sentence] if j in self.vocab]
                if not reviews2[review][sentence]:
                    reviews2[review][sentence].append('<pad>')
        all_in_sentence = []
        for review in range(len(reviews2)):
            for sentence in range(len(reviews2[review])):
                all_in_sentence.append(reviews2[review][sentence])
#        all_in_sentence=list(map(lambda x: word_tokenize(x),all_in_sentence))        
#        for i in range(len(all_in_sentence)):
#            all_in_sentence[i]=[j for j in all_in_sentence[i] if j not in stopword]
#            all_in_sentence[i]=[j for j in all_in_sentence[i] if not j.isdigit()]
#        self.all_in_sentence=all_in_sentence
       
        return all_in_sentence, reviews2
       
    def cluster_init(self):
        k = self.k
        data,data1 = self.transform_review()
        vocab = self.vocab
        for i in range(len(data)):
            for j in range(len(data[i])):
               data[i][j] = vocab[data[i][j]]
        sentence_average = np.empty((len(data),self.emb_dim))
        i = -1
        for sentence in data:
            i+= 1
            sentence_vector = np.zeros((1,self.emb_dim))
            for word in sentence:
                sentence_vector+= self.embedding_matrix[word]
            sentence_average[i] = sentence_vector/len(sentence)    

        k = KMeans(n_clusters=k)
        k.fit(sentence_average)
        clusters = k.cluster_centers_
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True) 
        init_centeriod = norm_aspect_matrix.astype(np.float32)     
        return init_centeriod,sentence_average
           

    def get_distance(self,centeriod, sentence_average):
        distance = np.empty((sentence_average.shape[0],centeriod.shape[0]))
        for i in range(centeriod.shape[0]):
            a = sentence_average-centeriod[i]
            distance[:,i] = np.apply_along_axis(np.linalg.norm,1,a)
        output = {k:np.argmin(distance[k]) for k in range(sentence_average.shape[0])}
        return output
    

    def soft_cosin(self):
        seed_word = self.seed_word
        similarity_matrix = self.similarity_matrix
        overall_dict = self.overall_dict
        reviews1,reviews2 = self.transform_review()
#        overall_dict=gensim.corpora.Dictionary(reviews1)
#        similarity_index = WordEmbeddingSimilarityIndex(self.model.wv)
#        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, overall_dict)
        similarity_output = {i:{k:[] for k in seed_word} for i in range(len(reviews1))}
        for i in tqdm(range(len(reviews1))):
            review_doc = overall_dict.doc2bow(reviews1[i])
            for key in seed_word:
                similarity = 0
                for j in range(len(seed_word[key])): #change for semeval?
                    seed_word_doc = overall_dict.doc2bow([seed_word[key][j]])
#                for j in seed_word[key]:
#                    seed_word_doc = overall_dict.doc2bow(j)    
                    similarity+= similarity_matrix.inner_product(review_doc,seed_word_doc, normalized=True)
                similarity_output[i][key] = '%.3f'% (similarity/len(seed_word[key]))
        return similarity_output, overall_dict        

    def score(self,input_centeriod=[],input_cluster_score=None):
        seed_word = self.seed_word
        reviews1,reviews2 = self.transform_review()
            
        if len(input_centeriod) != 0: 
            centeriod = input_centeriod
            vocab = self.vocab
            for i in range(len(reviews1)):
                for j in range(len(reviews1[i])):
                   reviews1[i][j] = vocab[reviews1[i][j]]
            sentence_average = np.empty((len(reviews1),self.emb_dim))
            i = -1
            for sentence in reviews1:
                i+= 1
                sentence_vector = np.zeros((1,self.emb_dim))
                for word in sentence:
                    sentence_vector+= self.embedding_matrix[word]
                sentence_average[i] = sentence_vector/len(sentence)
                
        else:
            centeriod,sentence_average = self.cluster_init()
        cluster_index = self.get_distance(centeriod,sentence_average) 
        similarity_output,d = self.soft_cosin()
        
        if input_cluster_score == None:
            cluster_score = {k:{aspect:0 for aspect in seed_word } for k in range(centeriod.shape[0])}        
            for i in range(centeriod.shape[0]):
                cluster_select = [key for key in cluster_index if cluster_index[key]==i]
                for aspect in seed_word:
                    aspect_cluster_score = 0
                    for j in range(len(cluster_select)):
                        aspect_cluster_score+= float(similarity_output[cluster_select[j]][aspect])
                    try:
                        cluster_score[i][aspect] = aspect_cluster_score/len(cluster_select)
                    except:
                        cluster_score[i][aspect] = 0.0
        else:
            cluster_score = input_cluster_score
        finial_output = {k:{aspect:0 for aspect in seed_word } for k in range(len(reviews1))}    
        for k in similarity_output:
            for aspect in similarity_output[k]:
                finial_output[k][aspect] = self.alpha[aspect]*float(similarity_output[k][aspect])+(1-self.alpha[aspect])*cluster_score[cluster_index[k]][aspect]
        return finial_output, similarity_output, cluster_score,centeriod
    
    def aspect_detection(self,input_centroid,cluster_score, input_threshold=None):
        if input_threshold:
            threshold = input_threshold
        else:
            threshold = self.threshold
        all_in_sentence, reviews2 = self.transform_review()
        finial_score, similarity_output,cluster_score,centeriod = self.score(input_centroid,cluster_score)
        detection = {k:{a:0 for a in self.seed_word} for k in range(len(all_in_sentence))}
        for key in finial_score:
            for aspect in finial_score[key]:
                if round(finial_score[key][aspect],2) > threshold[aspect]:
                    detection[key][aspect] = 1
        detection_output = {k:{a:0 for a in self.seed_word} for k in range(len(reviews2))}
        start = 0
        for i in range(len(reviews2)):
            finish = start+len(reviews2[i])
            aux = {}
            aux_list = list(range(start,finish))
            for j in aux_list:
                aux.update({j:detection[j]})
            for value in aux.values():
                for aspect in self.seed_word:
                    if value[aspect] == 1:
                        detection_output[i][aspect] += 1
            start = finish   
            
        detection_dataframe = pd.DataFrame()
        detection_dataframe['reviews'] = self.reviews
        for aspect in self.seed_word:
            aspect_list = []
            for value in detection_output.values():
                aspect_list.append(value[aspect])
            detection_dataframe[aspect] = aspect_list
        
        return detection_dataframe      
    
        
    def find_threshold(self,truth_label,finial_score,prefer_precision=True): # truth_label as dataframe with reviews different aspects
        test_all_sentence,test2 = self.transform_review(list(truth_label['reviews']))
        reviews1,reviews2 = self.transform_review()
        choices = []
        for i in test_all_sentence:
            choices.append(reviews1.index(i))
        prediction = {i:finial_score[choices[i]] for i in range(len(choices))}
        output = {}
        for aspect in self.seed_word:
            threshold = 0.005
            result = 4 * [0.0]# f1, precision, recall, threshold
            while threshold < 0.3:
                TP = 0
                FP = 0
                FN = 0
                for i in range(len(truth_label)):
                    if prediction[i][aspect] > threshold:
                        if list(truth_label[aspect])[i] > 0:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if list(truth_label[aspect])[i] > 0:
                            FN += 1
                try:
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    if prefer_precision:
                        f1 = (1+0.25)*(precision*recall) / (0.25*precision+recall) #prefer precision
                    else:
                        f1 = 2*(precision*recall) / (precision+recall)
                except ZeroDivisionError:
                    f1 = 0.0
                if f1 > result[0]:
                    result = [f1, precision, recall, threshold]
                threshold += 0.001
                threshold = round(threshold,3)
            output.update({aspect:result})     
        return output






















