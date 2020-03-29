# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:02:53 2019

@author: hui.wang
"""
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input, Lambda
from keras.models import Model
import copy
from tqdm import tqdm
import numpy as np
import random
from mylayer import Attention, WeightedSum, Reconstruction, distance_loss, advanced_distance_loss
from keras.models import load_model

class Train:
    def __init__(self,seed_word, train_x,train_x_copy, maxlen,vocab,emb,args,opposite_pool):
        self.seed_word=  seed_word
        self.train_x = train_x
        self.train_x_copy = train_x_copy
        self.maxlen = maxlen
        self.vocab = vocab
        self.emb = emb
        self.args = args
        self.opposite_pool = opposite_pool
        
        
    def sentence_average(self,data,emb):
        sentence_average = np.empty((len(data),emb.emb_dim))
        i = -1
        for sentence in data:
            i+= 1
            sentence_vector = np.zeros((1,emb.emb_dim))
            for word in sentence:
                sentence_vector+= emb.embedding_matrix[word]
            sentence_average[i] = sentence_vector/len(sentence)    

        return sentence_average
    

    def create_model(self,T_init=None):
        maxlen = self.maxlen
        emb = self.emb
        args = self.args
        
        def ortho_reg(weight_matrix):
            w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)),
                                     K.floatx())
            reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0]))) #.vlaue afrer value
            return 0.1 * reg
         
        vocab_size = len(self.vocab)
        sentence_input = Input(shape=(maxlen,),dtype='int32',name='sentence_input')
        opposite_input1 = Input(shape=(args.opposite_size,maxlen),dtype='int32',name='opposite_input1')
#        same_input1=Input(shape=(args.opposite_size,maxlen),dtype='int32',name='same_input1')# new add
        
        sentence_embedding_layer = Embedding(vocab_size,emb.emb_dim,weights=[emb.embedding_matrix],trainable=False,mask_zero=True, name='sentence_emb')
        e_w = sentence_embedding_layer(sentence_input)
        e_n = sentence_embedding_layer(opposite_input1)
#        e_s=sentence_embedding_layer(same_input1)# new add
        
        s_w = Lambda(lambda x: K.mean(x,axis=1))(e_w)  # s_w as sentence average
        z_n = Lambda(lambda x: K.mean(x,axis=1))(e_n)
#        z_sn=Lambda(lambda x: K.mean(x,axis=1))(e_s) # new add
        
        weight = Attention(name='weight')([e_w,s_w])
        z_s = WeightedSum(name='weight_sum')([e_w,weight]) # only can use layer or for simplfy to use Lambda
        p_t = Dense(args.classnumber,activation='softmax',name='class')(z_s)  # for 3 need to further make a class for every manually pre-defined parameter
        dim = emb.emb_dim
        r_s = Reconstruction(args.classnumber,dim,T_regularizer=ortho_reg, name='sentiment_matrix')(p_t)  # tensor shape: (?,300)
        #loss
        loss = distance_loss()([z_s,r_s,z_n])
#        loss=advanced_distance_loss()([z_s,r_s,z_n,z_sn]) #new add
        
        model = Model(inputs=[sentence_input,opposite_input1], outputs=loss)
#        model=Model(inputs=[sentence_input,opposite_input1,same_input1], outputs=loss) # new add
    
        if T_init is not None: 
            K.set_value(model.get_layer('sentiment_matrix').T, T_init)
    
        return model
        
    def my_loss(self, y_true,y_pred):
        return K.mean(y_pred)        
        
    def batch_generator(self,sentence,batch_size):
        batch_number = len(sentence)/batch_size
        np.random.shuffle(sentence)
        sentence2 = copy.deepcopy(sentence)
        batch_count = 0
        while True:
            if batch_count >= batch_number:
                np.random.shuffle(sentence)
                sentence2 = copy.deepcopy(sentence)
                batch_count = 0
        
            batch = sentence2[batch_count*batch_size: (batch_count+1)*batch_size]
            batch_count += 1
            yield batch
        
    def opposite_input(self,batch_input, opposite_size,pool):
        batch_size = batch_input.shape[0]
        dim = batch_input.shape[1]
        output = []
        for i in range(len(batch_input)):
            if tuple(batch_input[i]) in pool['positive']:
                choice = random.choices(pool['negative'],k=opposite_size)
            else:
                choice = random.choices(pool['positive'],k=opposite_size)
            output.extend(choice)
        opposite_output = np.asarray(output).reshape(batch_size,opposite_size,dim)        
        return(opposite_output)   
    

    def same_pool_input(self,batch_input,opposite_size,pool):
        batch_size = batch_input.shape[0]
        dim = batch_input.shape[1]
        output = []
        for i in range(len(batch_input)):
            if tuple(batch_input[i]) in pool['positive']:
                choice = random.choices(pool['positive'],k=opposite_size)
            else:
                choice = random.choices(pool['negative'],k=opposite_size)
            output.extend(choice)
        opposite_output = np.asarray(output).reshape(batch_size,opposite_size,dim)        
        return(opposite_output)
            
        
        
        
    def train_model(self,save=None,aspect=None):
        vocab = self.vocab
        emb = self.emb
        args = self.args
        np.random.seed(1234)
        seed_word1 = self.seed_word['positive']
        seed_word2 = self.seed_word['negative']
        input1 = []
        for word in seed_word1:
            input1.append([vocab[word]])

        input2 = []
        for word in seed_word2:
            input2.append([vocab[word]])
        s1 = self.sentence_average(input1,emb)
        s11 = np.mean(s1,axis=0).reshape((1,self.emb.emb_dim))
        s2 = self.sentence_average(input2,emb)
        s22 = np.mean(s2,axis=0).reshape((1,self.emb.emb_dim))
        init_centroid = np.concatenate((s11,s22))
            

        model = self.create_model(init_centroid)            
        model.compile(optimizer='sgd',loss=self.my_loss, metrics=[self.my_loss])
        
        sentence_batch = self.batch_generator(self.train_x,self.args.batch_size)
        batch_per_epoch = len(self.train_x)//self.args.batch_size + 1
        
        min_loss = float('inf')
        
        for i in range(self.args.epoch_size):
            loss = 0
            distance_loss = 0
            for j in tqdm(range(batch_per_epoch)):
                batch_input = next(sentence_batch)
                opposite_batch_input = self.opposite_input(batch_input,self.args.opposite_size,self.opposite_pool)
#                same_batch_input=self.same_pool_input(batch_input,self.args.opposite_size,self.opposite_pool)
                batch_loss, batch_distance_loss = model.train_on_batch([batch_input,opposite_batch_input],np.ones((len(batch_input), 1)) )
                loss+= batch_loss/batch_per_epoch
                distance_loss+= batch_distance_loss/batch_per_epoch
        
        if loss < min_loss:
            min_loss = loss
            class_output_model = K.function([model.get_layer('sentence_input').input],[model.get_layer('class').output])
            probability = np.asarray(class_output_model([self.train_x_copy]))
            
            sentence_embedding_model = K.function([model.get_layer('sentence_input').input],[model.get_layer('weight_sum').output])
            sentence_vector = np.asarray(sentence_embedding_model([self.train_x_copy]))
            
            weight_model = K.function([model.get_layer('sentence_input').input],[model.get_layer('weight').output])
            weight = np.asarray(weight_model([self.train_x_copy]))
        if save == True:
            model.save('food_model1031.h5') 
            print('model saved')
        return probability, sentence_vector, weight, model
        
        
 
        