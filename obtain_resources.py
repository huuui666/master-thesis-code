# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:01:17 2020

@author: huuui
"""


import gensim
from gensim.models import Word2Vec
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk import pos_tag  
from lemma import lemmatizer
import operator
import re
from nltk.corpus import stopwords
import pickle
import os


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
review1_path = os.path.join(fileDir,'Data','train_review2014.pkl')
review2014 = readFile(review1_path)

Citysearch_path = os.path.join(fileDir,'Data','Citysearch.txt')
with open(Citysearch_path) as f:
    content = f.readlines()
Citysearch_review = [x.strip() for x in content] 

def create_all_in_sentence(reviews, stopword=False):
    if stopword:
        stopword = stopwords.words('english')
    else:
        stopword=[]
    reviews1 = list(map(lambda x: sent_tokenize(x),reviews))
    reviews2 = reviews1.copy()
    for review in range(len(reviews1)):
        for sentence in range(len(reviews1[review])):
            reviews2[review][sentence] = (re.sub(r'\W+', ' ',str(reviews1[review][sentence].lower() ))).strip()
            reviews2[review][sentence] = lemmatizer().lemmatize_sentence(reviews2[review][sentence])
            reviews2[review][sentence] = [j for j in reviews2[review][sentence] if j not in stopword]
            reviews2[review][sentence] = [j for j in reviews2[review][sentence] if not j.isdigit()]
            if not reviews2[review][sentence]:
                reviews2[review][sentence].append('<pad>')#if empty after clean, add '<pad>' which is zero in vocab
    all_in_sentence = []
    for review in range(len(reviews2)):
        for sentence in range(len(reviews2[review])):
            all_in_sentence.append(reviews2[review][sentence])
       
    return all_in_sentence


def build_vocab(all_in_sentence):# build vocabulary, based on the frequency of word in corpus
    word_freq = {}
    all_in_sentence = [' '.join(all_in_sentence[i]) for i in range(len(all_in_sentence))]    
    for sentence in all_in_sentence:
        all_word = [i for i in word_tokenize(sentence) ]
        all_word = [i for i in all_word if not i.isdigit()]
        for word in all_word:
            if word not in word_freq :
                word_freq[word] = 1
            else:
                word_freq[word] = word_freq[word]+all_word.count(word)
    sorted_word_freq = sorted(word_freq.items(),key=operator.itemgetter(1),reverse=True)
    vocab = {'<pad>':0}
    index = len(vocab)
    for word, _ in sorted_word_freq:
        vocab[word] = index
        index += 1 
   
    return(vocab)
    

def create_embedding(all_in_sentence, save_source):
    model = Word2Vec(min_count = 5,# minimum count for word in corpus
                     window = 5,  #window size of how many words to be considered around the targert word
                     size=200,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=50) # negative sampling which increases the performance significantly
    model.build_vocab(all_in_sentence, progress_per=10000)
    model.train(all_in_sentence, total_examples=model.corpus_count, epochs=30, report_delay=1)      
    model.save(save_source)

def create_softcosine_resourse(model_source,all_in_sentence): # create resources for soft cosine
    overall_dict = gensim.corpora.Dictionary(all_in_sentence)
    model = gensim.models.Word2Vec.load(model_source)
    similarity_index = WordEmbeddingSimilarityIndex(model.wv)

    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, overall_dict)
    return overall_dict, similarity_matrix

def build_part_of_speech(review):
    part_of_speech_vocab={}
    for i in range(len(review)):
        for word, pos in pos_tag(lemmatizer().lemmatize_sentence(review[i])):
            if word not in part_of_speech_vocab:
                part_of_speech_vocab.update({word:[pos]})
            else:
                part_of_speech_vocab[word].append(pos)
    return part_of_speech_vocab            

all_in_sentence = create_all_in_sentence(review2014)
vocab = build_vocab(all_in_sentence)

embedding_save_path = os.path.join(fileDir,'new_resources','w2v_embedding')
all_in_sentence_citysearch = create_all_in_sentence(Citysearch_review)
create_embedding(all_in_sentence_citysearch, embedding_save_path)

#embedding_path = os.path.join(fileDir,'resources','w2v_embedding')
overall_dict, similarity_matrix = create_softcosine_resourse(embedding_save_path,all_in_sentence)
part_of_speech = build_part_of_speech(review2014)


vocab_path = os.path.join(fileDir,'new_resources','vocab.pkl')
saveFile(vocab_path,vocab)

overall_dict_path = os.path.join(fileDir,'new_resources','overall_dict2014.pkl')
saveFile(overall_dict_path,overall_dict)

similarity_matrix_path = os.path.join(fileDir,'new_resources','similarity_matrix2014.pkl')
saveFile(similarity_matrix_path,similarity_matrix)

part_of_speech_path = os.path.join(fileDir,'new_resources','part_of_speech_vocab')
saveFile(part_of_speech_path,part_of_speech)
    