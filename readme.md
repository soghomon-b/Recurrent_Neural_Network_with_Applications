A repository that contains classes needed to train and use an RNN to identify categories, there are classes specially designed to handle Arabic data and others specially designed to handle long English sentences. // 

Here is an explanation of each class: 
## ArabicDataProcessor
All classes extend the AbstractArabicDataProcessor, the file has different fields and methods to facilitate processing Arabic data into tensors for machine learning. The only abstract method is the word_to_tensor method. This one is processed differently in each child's class. 
# SimpleArabicDataProcessor 
uses the one-hot encoding method in the word_to_tensor method
# PositionEmbeddingArabicDataProcessor 
uses the one-hot encoding method and replaces the 1 in the tensor with a 2 for characters identified as important positions in a word that signal a specific scale. 

the classification is as follows: 
* if the second letter of the word is an "ا", it is likely a subject scale for 3-lettered verbs. 
* if the word's first letter is a "م",  it is likely an object scale for 3-lettered verbs.
* if the word contains a "kasrah" diacritic, it is likely a subject scale for 4 or more lettered verbs.
* if the word contains a "fatha" diacritic, it is likely a subject scale for 4 or more lettered verbs.

## SentenceProcessor
extends the SimpleArabicDataProcessor, it is in a different file since it has the same functionality as the SimpleArabicDataProcessor, but works on data that contains long English sentences instead of Arabic words. 

Process each sentence as follows: 
* converts the sentence into one word by removing words in the sentence that likely do not signal specific meaning, then the sentence gets processed by the stopwords in the NLTK library and turned into one word.
* The word uses the word_to_tensor in the SimpleArabicDataProcessor.
## RNN 
the recurrent neural network itself. Code based on the following tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

## Training 
used the same training process in https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
Refactored the process in the tutorial into one class, which if you instantiate then call, will start the process on an RNN by procesing the data using the SimpleArabicDataProcessor. If a different processor is needed, instance.set_model() can be called, before instance()


