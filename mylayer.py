# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:32:36 2019

@author: hui.wang
"""

import keras.backend as K
from keras import constraints
from keras import initializers as initializations
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers import Dot
from keras.models import Sequential
from sklearn.cluster import KMeans



class Attention(Layer):
    def __init__(self, bias=True, **kwargs):
        self.supports_masking = True # support masking 
        self.init = initializations.get('glorot_uniform')
        self.bias = bias
        super(Attention,self).__init__(**kwargs)
        
    def build(self, input_shape): # input_shape is how the input for the layer should be look like
        assert type(input_shape) == list # make sure input as wanted, a list first is e_w, then s_w
        assert len(input_shape) == 2
        
        self.sentence_length = input_shape[0][1]
        self.M = self.add_weight(  name='weight',
                shape=(input_shape[0][-1],input_shape[1][-1]), # 300*300 dimension in this case
                               initializer=self.init
                              
                )
        if self.bias:
            self.b = self.add_weight(    name='bias',
                    shape=(1,),
                               initializer='zero'
                        
                    )
        self.built = True # need for build the weight
        
     
    def compute_mask(self, input_tensor, mask=None): # not need to pass mask to a further layer
        return None   
    
    def call(self,input_tensor,mask=None):
        e_w = input_tensor[0]
        s_w = input_tensor[1]
        mask = mask[0]
        
        y = K.dot(self.M, K.transpose(s_w)) #300*1 dimension
        y = K.transpose(y) # for further computation
        y = K.expand_dims(y,-2) # extend tensor dimension
        y = K.repeat_elements(y,self.sentence_length,axis=1) #actually do not need
        d_i = K.sum(e_w*y,axis=-1) # 173 weights for every word in sentence,  maximum length equal to 173, 
                                 #at last tensor dimension(axis=-1)
        if self.bias: # add bias term
            b = K.repeat_elements(self.b,self.sentence_length,axis=0)
            d_i+= b
            
        d_i = K.tanh(d_i)
        d_i = K.exp(d_i)
        
        if mask is not None:
            d_i*= K.cast(mask, K.floatx())
        
        d_i/= K.cast(K.sum(d_i,axis=1,keepdims=True)+K.epsilon(),K.floatx())# keep dimension, still 173, for further weightsum
        return d_i 
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1]) # as None,173
        

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)
        
    def call(self, input_tensor,mask=None):
        assert type(input_tensor) == list
        assert len(input_tensor) == 2
        
        e_w = input_tensor[0]
        weight = input_tensor[1]
        weight = K.expand_dims(weight)
        weighted = K.sum(e_w*weight,axis=1)
        self.weighted = weighted   # maybe need for output
        return(weighted)
        
    def compute_mask(self,x,mask=None):
        return None
        
    def compute_output_shape(self,input_shape):
        return(input_shape[0][0],input_shape[0][-1])
        
#
#class Kmeans(Layer):# as alternative method layer
#    def __init__(self,classnumber,**kwargs):
#        self.supports_masing=True
#        self.class_number=classnumber
#        super(Kmeans,self).__init__(**kwargs)
#        
        

        
        
class Reconstruction(Layer):
    def __init__(self, class_number,emb_dim,T_centroid='uniform', T_regularizer=None, weights=None, **kwargs):
        self.supports_masking = True
        self.emb_dim = emb_dim
        self.class_number = class_number
        self.T_init = initializations.get(T_centroid)
        self.T_regularizer = regularizers.get(T_regularizer)
        self.init_weight = weights
        super(Reconstruction, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        self.T = self.add_weight(name='sentiment_weightmatrix',
                              shape=(self.class_number, self.emb_dim),
                              initializer=self.T_init,
                              regularizer=self.T_regularizer
                              )
        if self.init_weight is not None:
            self.set_weights(self.init_weight)
        self.built = True
        
    
    def call(self, input_tensor,mask=None):
        return(K.dot(input_tensor,self.T)) #none, 300 tensor
    
    def compute_mask(self,x,mask=None):
        return None
        
    def compute_output_shape(self,input_shape):
        return(input_shape[0],self.emb_dim)
    
    def get_config(self):    
        config = {
            'class_number': self.class_number,
            'emb_dim': self.emb_dim}    
        base_config = super(Reconstruction, self).get_config()    
        return dict(list(base_config.items()) + list(config.items()))

            
        

class cosin_loss(Layer):
    def __init__(self,  **kwargs):
        super(cosin_loss, self).__init__(**kwargs)
        
    def call(self, input_tensor,mask=None):
        z_s = input_tensor[0]
        r_s = input_tensor[1]
        cosin_proimity = Dot(axes=-1,normalize=True)([z_s,r_s])
        loss =-(K.cast(K.sum(K.log((K.minimum(10**-7,cosin_proimity))),axis=-1,keepdims=True),K.floatx()))
        return loss
    
    def compute_mask(self,x,mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0][0],1)
        
class distance_loss(Layer):
    def __init__(self, **kwargs):
        super(distance_loss,self).__init__(**kwargs)
        
    def call(self,input_tensor, mask=None):
        z_s = input_tensor[0]
        r_s = input_tensor[1]
        z_n = input_tensor[2]
        
        z_s = K.l2_normalize(z_s, axis=-1)
        r_s = K.l2_normalize(r_s, axis=-1)
        z_n = K.l2_normalize(z_n, axis=-1)
        
        length = z_n.shape[1]#.value
        
        pos = K.sum(z_s*r_s, axis=-1,keepdims=True)
        pos = K.repeat_elements(pos,length,axis=1)
        r_s = K.expand_dims(r_s,axis=-2)
        r_s = K.repeat_elements(r_s,length,axis=1)
        neg = K.sum(z_n*r_s,axis=-1)
        loss = K.cast(K.sum(K.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), K.floatx())
        
        return loss
    
    def compute_mask(self,x,mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0][0],1)
        
    


class advanced_distance_loss(Layer):
    def __init__(self, **kwargs):
        super(advanced_distance_loss,self).__init__(**kwargs)
        
    def call(self,input_tensor, mask=None):
        z_s = input_tensor[0]
        r_s = input_tensor[1]
        z_n = input_tensor[2]
        r_n = input_tensor[3]
        
        z_s = K.l2_normalize(z_s, axis=-1)
        r_s = K.l2_normalize(r_s, axis=-1)
        z_n = K.l2_normalize(z_n, axis=-1)
        r_n = K.l2_normalize(r_n, axis=-1)
        
        length = z_n.shape[1]
        
        pos = K.sum(z_s*r_s, axis=-1,keepdims=True)
        pos = K.repeat_elements(pos,length,axis=1)
        r_s = K.expand_dims(r_s,axis=-2)
        r_s = K.repeat_elements(r_s,length,axis=1)
        neg = K.sum(z_n*r_s,axis=-1)
        pos_rest = K.sum(r_n*r_s,axis=-1)
        
        loss = K.cast(K.sum(K.maximum(0., (1. - pos + neg-pos_rest)), axis=-1, keepdims=True), K.floatx())
        
        return loss
    
    def compute_mask(self,x,mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0][0],1)
        




