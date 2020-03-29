# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:01:57 2019

@author: huuui
"""
import os
import pickle
import embeddingreader
from joint_train import all_for_one
from dataread import Read_data
from keras.preprocessing import sequence
import copy
from sentiment_train import Train
from seedupdate import seed_word_extraction
from new_eva import evaluate


def saveFile(filename,file):
    pickle_file=open(filename,'wb+')
    pickle.dump(file,pickle_file)
    pickle_file.close()
    print('Saved')

def readFile(filename):
    pickle_file = open(filename,'rb')
    output = pickle.load(pickle_file)
    pickle_file.close()
    return output


fileDir = os.path.dirname(os.path.realpath('__file__'))
vocab_path = os.path.join(fileDir,'resources','bike_vocab.pkl')
vocab = readFile(vocab_path)

centroid_path = os.path.join(fileDir,'resources','bike_centroid')
centroid = readFile(centroid_path)

overall_dict_path = os.path.join(fileDir,'resources','bike_cluster_score')
cluster_score = readFile(overall_dict_path)



overall_dict_path = os.path.join(fileDir,'resources','bike_overall_dict.pkl')
overall_dict = readFile(overall_dict_path)

similarity_matrix_path = os.path.join(fileDir,'resources','bike_similarity_matrix.pkl')
similarity_matrix = readFile(similarity_matrix_path)

embedding_source = os.path.join(fileDir,'resources','bike_w2v_embedding')


review1_path = os.path.join(fileDir,'Data','bike_data.pkl')
bike_review =  readFile(review1_path)


review_dict_path = os.path.join(fileDir,'Data','bike_review_dict')
bike_review_dict=  readFile(review_dict_path)

bike_part_of_speech_vocab_path = os.path.join(fileDir,'resources','bike_part_of_speech_vocab')
bike_part_of_speech_vocab = readFile(bike_part_of_speech_vocab_path)


bike_test_review_path =  os.path.join(fileDir,'Data','bike_test_review')
bike_test_review = readFile(bike_test_review_path)


bike_gold_path =  os.path.join(fileDir,'Data','bike_gold')
bike_gold = readFile(bike_gold_path)

emb = embeddingreader.Embeddingreader(embedding_source,vocab)

sentiment_seedword={
                'quality':{'positive':['perfect','beautiful','happy'],
                         'negative':['broken','defective','damage']} ,
                 'service':{'positive':['great','attentive','help'],
                         'negative':['rude','terrible','poor']},
                'delivery':{'positive':['fast','roadworthy','neat'],
                         'negative':['delay','terrible','poor']}

                }
# changing sentiment_seedword to positive as great and negative as terrible will lead to the result of Table 5.9             
threshold={'service':0.15,'quality':0.15,'delivery':0.15}
a = all_for_one(bike_review,vocab,emb,threshold,sentiment_seedword,centroid) 
#review_dict,t1,t2,cluster_score = a.init_detect(similarity_matrix,overall_dict)    
z2,z3 = a.sentiment_split(bike_review_dict)


class Arguement:
    def __init__(self,classnumber, batch_size, epoch_size,opposite_size):
        self.classnumber = classnumber
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.opposite_size = opposite_size
        
args = Arguement(2,50,10,50)



accuracy = {}
accuracy_low_boundary = {}
aspect_positive = {}
aspect_negative = {}
for aspect in sentiment_seedword:
    r1,r2 = Read_data(z2, aspect,vocab,stopword=True).sentence_convert()        

    train_x = sequence.pad_sequences(r1, maxlen= r2)
    train_x_copy = copy.deepcopy(train_x)
    opposite_pool1 = Read_data(z2, aspect,vocab,stopword=True).link_grade(train_x,z3[aspect])
    probability,sentence_vector,weight,_ = Train(sentiment_seedword[aspect],train_x,train_x_copy, r2,
                        vocab,emb,args,opposite_pool1).train_model()

    sent_pos, sent_neg = seed_word_extraction(bike_review,vocab).new_sentiment_word(weight,probability,train_x_copy,bike_part_of_speech_vocab,emb)
    sent_pos_list = [i for i in sent_pos]
    sent_neg_list = [i for i in sent_neg]
    
    aspect_positive.update({aspect:sent_pos_list})
    aspect_negative.update({aspect:sent_neg_list})
    
    
    probability,sentence_vector,weight, model = Train({'positive':sent_pos_list,'negative':sent_neg_list},train_x,train_x_copy, r2,
                        vocab,emb,args,opposite_pool1).train_model()
    
    
    accuracy.update({aspect: evaluate(aspect,bike_gold, bike_test_review,vocab, model ).sentiment_accuracy()})
    accuracy_low_boundary.update({aspect: evaluate(aspect,bike_gold, bike_test_review,vocab, model ).sentiment_accuracy(boundary = 0.4)})
    
    name = aspect+'_bike_model'
    path = os.path.join(fileDir,'output_model',name)
    model.save(path)

updated_seed_word = {'positive':aspect_positive, 'negative':aspect_negative}
updated_seed_path = os.path.join(fileDir,'output_model','updated__bike_seed_word')
saveFile(updated_seed_path, updated_seed_word)
    
print ('Accuracy when set 0.5 as boundary: ')
print (accuracy)
print ('Accuracy when set 0.4 as boundary: ')
print(accuracy_low_boundary)
    
