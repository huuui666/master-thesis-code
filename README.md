# master-thesis-code
In order to run the code, there should be tqdm, scikit-learn, NLTK, Keras and Gensim installed. (pip install -r requirements.txt)

The restaurant_main is to train the model and give the evaulation based on the data of Sem Eval 2014 restaurant data, just run it in Anaconda prompt. (python restaurant_main.py) (there may appears a warning: The graph couldn't be sorted in 
topological order about tensorflow, which is no need to worry. This warning may due to the data set size or tensorflow backend, given that running model on  bike dataset hardly appears this warning.) 
The aspect category detection performance, sentiment classification accuracy and joint performance would be given in the end.

The saved sentiment models are saved in output_model folder, as well as new centroid for detection and the updated seed words.

The bike_main will give the sentiment classification performance on the Trustpilot data. The sentiment model will be saved as well as the updated sentiment seed words into the output_model folder.

With Data folder, there are several restaurant data, including train data review (train_review2014), train data gold label (train_gold2014), test data review (test_review2014), test data gold label (gold_test2014) and test 
data gold label dataframe (gold2014_dataframe).

The bike data includes bike train data (bike_data), bike test data (bike_test_review), bike test data gold label (bike_gold). 

There is also a data for restaurant word embedding (Citysearch).

Within resources folder, there are all resources used for training, obtained from Data. All resources could be re-obtained by running obtain_resources.py and then saved into new_resource folder.


