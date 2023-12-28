import torch
from torch._tensor import Tensor 
import torch.nn as nn 
import os
from abc import ABC, abstractmethod
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ArabicDataProcessor import SimpleArabicDataProcessor


#processes sentences to shorter words to be feeded into the RNN
class SentenceProcessor(SimpleArabicDataProcessor): 
     def __init__(self, file_directories : list):
        super.__init__(file_directories) 

    #uses NLTK to process the sentence, then removes words that likely don't add meaning
     def sentenceTokenizer(self, sentence : str) -> str: 

         words = word_tokenize(sentence)
         no_words = ["was", "is", "are", "were", "am", "this", "these", "that", "those", "I",
                      "you", "we", "they", "he", "she", "it", "'", "!", "and", "our", "your",
                        "their", "me", "the", "a", "an", "to", "has", "been", "have", "everything",
                          "something", "from", "very", "much", "super", "," ]
         stop_words = set(stopwords.words('english'))
         filtered_words = [word for word in words if word.lower() not in stop_words]
         for word in filtered_words: 
             filtered_words.remove(word) if word in no_words else None
        
         filtered_review = ' '.join(filtered_words)
         print(filtered_review) if len(filtered_words) > 3 else None
         return filtered_review
    
    #uses the one-hot encoding method to turn the word into a tensor 
    #after being processed with the sentenceTokenizer
     def word_to_tensor(self, word: str) -> torch.Tensor:
         filtered_word = self.sentenceTokenizer(word)
         return super().word_to_tensor(filtered_word)