# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:43:31 2020

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
from Detection import detection
from aspect_centroid import centroid_generate
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
vocab_path = os.path.join(fileDir,'resources','vocab.pkl')
vocab = readFile(vocab_path)

overall_dict_path = os.path.join(fileDir,'resources','centroid2014.pkl')
centroid = readFile(overall_dict_path)

overall_dict_path = os.path.join(fileDir,'resources','cluster_score2014.pkl')
cluster_score = readFile(overall_dict_path)



overall_dict_path = os.path.join(fileDir,'resources','overall_dict2014.pkl')
overall_dict = readFile(overall_dict_path)

similarity_matrix_path = os.path.join(fileDir,'resources','similarity_matrix2014.pkl')
similarity_matrix = readFile(similarity_matrix_path)

embedding_source = os.path.join(fileDir,'resources','w2v_embedding')


review1_path = os.path.join(fileDir,'Data','train_review2014.pkl')
review2014 = readFile(review1_path)


part_of_speech_vocab_path = os.path.join(fileDir,'resources','part_of_speech_vocab')
part_of_speech_vocab = readFile(part_of_speech_vocab_path)


emb = embeddingreader.Embeddingreader(embedding_source,vocab)

gold_path = os.path.join(fileDir,'Data','gold_test2014.pkl')
gold2014=readFile(gold_path)

test_review_path = os.path.join(fileDir,'Data','test_review2014')
test_review2014=readFile(test_review_path)


class Arguement:
    def __init__(self,classnumber, batch_size, epoch_size,opposite_size):
        self.classnumber = classnumber
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.opposite_size = opposite_size


args_large = Arguement(2,50,10,30) #20 ,10
args_small = Arguement(2,20,10,10)


threshold = {'food':0.016,'service':0.048,'ambience':0.062,'price':0.099} #2014 new resource


sentiment_seedword={'food':{'positive':['great'],'negative':['terrible']} ,
                 'service':{'positive':['great'],'negative':['terrible']},
                 'ambience':{'positive':['great'],'negative':['terrible']},
                 'price':{'positive':['great'],'negative':['terrible']}
                }

print('First roughly detection')
a = all_for_one(review2014,vocab,emb,threshold,sentiment_seedword,centroid) 
review_dict,t1,t2,cluster_score = a.init_detect(similarity_matrix,overall_dict)    
z2,z3 = a.sentiment_split(review_dict)


aspect_category_large = ['food', 'service']
aspect_category_small = ['ambience','price']

center = []
accuracy = {}
aspect_positive = {}
aspect_negative = {}
for aspect in aspect_category_large:
    print ('train '+ aspect+' model')
    r1,r2 = Read_data(z2, aspect,vocab,stopword=True).sentence_convert()        

    train_x = sequence.pad_sequences(r1, maxlen= r2)
    train_x_copy = copy.deepcopy(train_x)
    opposite_pool1 = Read_data(z2, aspect,vocab,stopword=True).link_grade(train_x,z3[aspect])
    probability,sentence_vector,weight,_ = Train(sentiment_seedword[aspect],train_x,train_x_copy, r2,
                        vocab,emb,args_large,opposite_pool1).train_model()

    sent_pos, sent_neg = seed_word_extraction(review2014,vocab).new_sentiment_word(weight,probability,train_x_copy,part_of_speech_vocab,emb)
    sent_pos_list = [i for i in sent_pos]
    sent_neg_list = [i for i in sent_neg]
    
    aspect_positive.update({aspect:sent_pos_list})
    aspect_negative.update({aspect:sent_neg_list})
    
    
    probability,sentence_vector,weight, model = Train({'positive':sent_pos_list,'negative':sent_neg_list},train_x,train_x_copy, r2,
                        vocab,emb,args_large,opposite_pool1).train_model()
    
    
    accuracy.update({aspect: evaluate(aspect,gold2014, test_review2014,vocab, model ).sentiment_accuracy()})
    
    name = aspect+'_model'
    path = os.path.join(fileDir,'output_model',name)
    model.save(path)
    
    _,sentence_average=detection(emb.embedding_matrix,similarity_matrix,overall_dict,vocab,
                          review_dict[aspect],emb_dim=200).cluster_init()
    center.append( centroid_generate(probability, sentence_average).generate())
    


for aspect in aspect_category_small:
    print ('train '+ aspect+' model')
    r1,r2 = Read_data(z2, aspect,vocab,stopword=True).sentence_convert()        

    train_x = sequence.pad_sequences(r1, maxlen= r2)
    train_x_copy = copy.deepcopy(train_x)
    opposite_pool1 = Read_data(z2, aspect,vocab,stopword=True).link_grade(train_x,z3[aspect])
    probability,sentence_vector,weight,_ = Train(sentiment_seedword[aspect],train_x,train_x_copy, r2,
                        vocab,emb,args_small,opposite_pool1).train_model()

    sent_pos, sent_neg = seed_word_extraction(review2014,vocab,threshold=0).new_sentiment_word(weight,probability,train_x_copy,part_of_speech_vocab,emb,RB=True)
    sent_pos_list = [i for i in sent_pos]
    sent_neg_list = [i for i in sent_neg]
    
    aspect_positive.update({aspect:sent_pos_list})
    aspect_negative.update({aspect:sent_neg_list})
    
    
    probability,sentence_vector,weight, model = Train({'positive':sent_pos_list,'negative':sent_neg_list},train_x,train_x_copy, r2,
                        vocab,emb,args_small,opposite_pool1).train_model()
    
    accuracy.update({aspect: evaluate(aspect,gold2014, test_review2014,vocab, model ).sentiment_accuracy()})
    
    name = aspect+'_model'
    path = os.path.join(fileDir,'output_model',name)
    model.save(path)
    
    _,sentence_average=detection(emb.embedding_matrix,similarity_matrix,overall_dict,vocab,
                          review_dict[aspect],emb_dim=200).cluster_init()
    if aspect == 'price':                      
        center.append( centroid_generate(probability, sentence_average, threshold=0.7).generate())
    else:
        center.append( centroid_generate(probability, sentence_average).generate())

updated_seed_word = {'positive':aspect_positive, 'negative':aspect_negative}
updated_seed_path = os.path.join(fileDir,'output_model','updated_seed_word')
saveFile(updated_seed_path, updated_seed_word)
    
print ('Start evaulation')
# detection evaulation
import numpy as np
gold_dataframe_path= os.path.join(fileDir,'Data','gold2014_dataframe')
gold2014_dataframe=readFile(gold_dataframe_path)
def find_threshold(prediction, truth_label):
    threshold=0.01
    result=4*[0.0]# f1, precision, recall, threshold
    while threshold<0.3:
        TP=0
        FP=0
        FN=0
        prediction_label=copy.deepcopy(prediction)
        for i in range(len(truth_label)):
            if prediction[i]>threshold:
                prediction_label[i]=1
                if truth_label[i] > 0:
                    TP+=1
                else:
                    FP+=1
            else:
                prediction_label[i]=0
                if truth_label[i] > 0:
                    FN+=1
        try:
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            f1=2*(precision*recall)/(precision+recall)
        except:
            f1=0.0
        if f1>result[0]:
            result=[f1,precision,recall,threshold]
        threshold+=0.001
        threshold=round(threshold,3)
    return result    

def evaulate(prediction, golden_label): # prediction is the prediction dataframe from aspect.detection(), golden_label is the test set dataframe
    assert len(prediction) == len(golden_label)

    prediction = prediction.reset_index(drop=True)
    golden_label = golden_label.reset_index(drop=True)
    prediction_evaulation = {aspect:[0.0] * 4 for aspect in list(prediction)[1:]}
    False_Positive_sentence = {aspect:[] for aspect in list(prediction)[1:] }
    False_Negative_sentence = {aspect:[] for aspect in list(prediction)[1:] }
    for i in prediction_evaulation:
        TP = 0
        FP = 0
        FN = 0
        for j in range(len(prediction)):
            if prediction[i][j] > 0:
                if golden_label[i][j] > 0:
                    TP += 1
                else:
                    FP += 1
                    False_Positive_sentence[i].append(prediction['reviews'][j])
            else:
                if golden_label[i][j] > 0:
                    FN += 1
                    False_Negative_sentence[i].append(prediction['reviews'][j])
        try:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2*(precision*recall) / (precision+recall)
        except:
            f1 = 0.0
        prediction_evaulation[i] = [f1, precision, recall]
    return prediction_evaulation, False_Positive_sentence, False_Negative_sentence

all_center = np.vstack(center)
new_score,_,new_cluster_score,_=detection(emb.embedding_matrix,similarity_matrix,overall_dict,vocab,
                          review2014,emb_dim=200).score(all_center)

centroid_path = os.path.join(fileDir,'output_model','centroid')
saveFile(centroid_path, all_center)


#threshold determination
train_gold_path= os.path.join(fileDir,'Data','train_gold2014.pkl')
train_gold=readFile(train_gold_path)
threshold={}
for aspect in new_score[0]:
    truth=[1  if aspect in train_gold[i] else 0 for i in range(len(new_score))] 
    result=find_threshold([i[aspect] for i in new_score.values()],truth)[-1]
    threshold.update({aspect:result})
 
prediction_dataframe=detection(emb.embedding_matrix,similarity_matrix,overall_dict,vocab,
                          test_review2014,emb_dim=200).aspect_detection(all_center,new_cluster_score, threshold)

eva_re1, diff_FP1, diff_FN1=evaulate(prediction_dataframe, gold2014_dataframe)

#joint
print ('Start joint perfromance evaulation')
from keras.models import load_model
import keras.backend as K
from mylayer import Attention, WeightedSum, Reconstruction, distance_loss
def my_loss( y_true,y_pred):
        return K.mean(y_pred)  


aspect_number = {}   
aspect_acc = {}

for aspect in sentiment_seedword:
    new_test_review2014 = []
    for i in range(len(test_review2014)):
        if aspect in gold2014[i]:
            if gold2014[i][aspect] == 'positive' or gold2014[i][aspect] == 'negative':
                new_test_review2014.append(test_review2014[i])
                    
    prediction_dataframe=detection(emb.embedding_matrix,similarity_matrix,overall_dict,vocab,
                          new_test_review2014,emb_dim=200).aspect_detection(all_center,new_cluster_score, threshold)
    
    name = aspect+'_model'
    path = os.path.join(fileDir,'output_model',name)
    model = load_model(path,custom_objects={'Attention':Attention(),'WeightedSum':WeightedSum(),
                                                      'Reconstruction':Reconstruction(2,200),'distance_loss':distance_loss(),
                                                      'my_loss':my_loss})
    aspect_prediction = list(prediction_dataframe[aspect])
    predicted_aspect_review =[new_test_review2014[i] for i in range(len(new_test_review2014)) if aspect_prediction[i] !=0 ]
    test_r1,test_r2=Read_data({aspect:predicted_aspect_review}, aspect,vocab,stopword=True).sentence_convert()            
    test_train_x = sequence.pad_sequences(test_r1, maxlen= model.layers[0].output.get_shape()[1])
    class_output_model=K.function([model.get_layer('sentence_input').input],[model.get_layer('class').output])
    probability_test=np.asarray(class_output_model([test_train_x]))    

    polarity_result=['positive' if probability_test[0][i][0]>0.5 else 'negative'  for i in range(len(predicted_aspect_review)) ]
    
    combined_prediction = {predicted_aspect_review[i] : polarity_result[i] for i in  range(len(predicted_aspect_review)) }
    number, acc = evaluate(aspect,gold2014, test_review2014,vocab, model ).combined_accuracy(combined_prediction)
    aspect_number.update({aspect:number})
    aspect_acc.update({aspect:acc})
    
TN_pos = 0
TN_neg = 0
pre_pos = 0
pre_neg = 0
rec_pos = 0
rec_neg = 0
acc = 0
total = 0
for aspect in aspect_number:
    TN_pos += aspect_number[aspect]['pos'][0]
    TN_neg += aspect_number[aspect]['neg'][0]
    pre_pos += aspect_number[aspect]['pos'][1]
    pre_neg += aspect_number[aspect]['neg'][1]
    rec_pos += aspect_number[aspect]['pos'][2]
    rec_neg += aspect_number[aspect]['neg'][2]
    acc += aspect_acc[aspect][0]
    total += aspect_acc[aspect][1]
    
Precision_pos =   TN_pos/pre_pos
Precision_neg = TN_neg/pre_neg
Recall_pos = TN_pos/rec_pos
Recall_neg = TN_neg/rec_neg
F1_positive=2*(Precision_pos*Recall_pos)/(Precision_pos+Recall_pos)
F1_negative=2*(Precision_neg*Recall_neg)/(Precision_neg+Recall_neg)




import pandas as pd
detection_frame = pd.DataFrame(eva_re1)
detection_frame.insert(0, "Aspect", ['F1','Preision','Recall'], True)
print('Aspect Category Detection:')
print( detection_frame)
print('Sentiment Accuracy' )
print(accuracy)
print('Joint Performance')
print({'Accuracy':acc/total, 'F1-positive':F1_positive,'F1_negative':F1_negative})
