import torch 
import torch.nn as nn 
import os
from abc import ABC, abstractmethod


#exception classes
class DictionaryNotEmpty(Exception): 
    pass



class CharacterNotInVocab(Exception): 
    pass

class RunTimeException(Exception): 
    pass

#actual classes 
class AbstractArabicDataProcessor(ABC): 
    
    def __init__(self, file_directories : list):
        self.files : list = file_directories
        self.data_dict : dict = {}
        self.tensor_dict : dict = {}
        self.vocab : list = []
        self.biggest_word_count : int = 0
        self.initiate_vocab()
        self.categories = []
        self.files_to_dict()
     
    #MODIFIES: self.vocab
    #initiates the vocabulary list
    def initiate_vocab(self): 
        for content in self.readFiles(): 
            letters_list = [char for char in content]
            for word in letters_list: 
                list_of_letters =[char for char in word]
                for letter in list_of_letters: 
                    if letter not in self.vocab and letter != "\n" or " ": 
                        self.vocab.append(letter)
                    
    #returns list of the content of txt file in self.files
    def readFiles(self) -> list:
        all_files_contents = []
        for file in self.files: 
            with open(file, "r", encoding='utf-8') as content:
                file_content = content.readlines()
                all_files_contents.append(file_content)
        return all_files_contents
    
    #converts the content of each txt file into a dictionary with the title as key and list of words as value
    #returns dict(String, List(String))
    #MODIFIES: self.data_dict
    #if self.data_dict is not empty, raises DictionaryNotEmpty
    def files_to_dict(self) -> dict: 
        if (len(self.data_dict) != 0):
            raise DictionaryNotEmpty("Data dictionary is not empty")
        for file in self.files: 
            word_list = []
            with open(file, "r", encoding='utf-8') as f:
                file_name = os.path.basename(file).replace(".txt", "")
                self.categories.append(file_name)
                for line in f: 
                    filtered_line = line.replace("/n", "").replace(" ", "")
                    self.biggest_word_count = len(filtered_line) if len(filtered_line) > self.biggest_word_count else self.biggest_word_count
                    word_list.append(filtered_line)
                self.data_dict[file_name] = word_list
        return self.data_dict

    #converts the values in data_dict into tensors
    #returns dict(String, torch.Tensor)
    #MODIFIES: self.tensor_dict
    #if self.tensor_dict is not empty, raises DictionaryNotEmpty
    def value_to_tensor(self) -> dict:
        if (len(self.tensor_dict) != 0):
            raise DictionaryNotEmpty("Tensor dictionary is not empty")
        for key,value in self.data_dict.items(): 
            self.tensor_dict[key] = self.to_list_tensor(value)
        return self.tensor_dict
    
    #empties dictionary to refill it without methods above throwing exception, if which_dict is not one of "data"
    # or "tensor" throws RunTimeException
    def empty_dict(self, which_dict : str): 
        if (which_dict != "data" or "tensor"): 
            raise RunTimeException("word passed should be data or tensor")

        if (which_dict == "data"): 
            self.data_dict = {}
        if (which_dict == "tensor"): 
            self.tensor_dict = {}
            
    
    #converts words in a list into a tensor, then returns the list as a tensor        
    def to_list_tensor(self, list_of_string : list) -> torch.Tensor:
        tensor_list = []
        for word in list_of_string: 
            word_tensor = self.word_to_tensor(word)
            tensor_list.append(word_tensor)
        stacked_tensor = torch.stack(tensor_list)
        return stacked_tensor
    

    @abstractmethod
    #converting the word into numbers understandable by the computer, process varies. 
    def word_to_tensor(self, word : str) -> torch.Tensor: 
        pass



class SimpleArabicDataProcessor(AbstractArabicDataProcessor): 
    
    def __init__(self, file_directories : list):
        super().__init__(file_directories)

    #uses simple one-hot encoding method 
    def word_to_tensor(self, word : str) -> torch.Tensor: 
        tensor = torch.zeros(self.biggest_word_count, 1, len(self.vocab))
        vocab_list = [letter for letter in word]
        letters_to_numbers = [self.vocab.index(letter) if letter in self.vocab else -4 for letter in vocab_list]
        for i in letters_to_numbers:
            if   i == -4: 
                raise CharacterNotInVocab("Character " + vocab_list[letters_to_numbers.index(i)] + " not in vocab list")
        
        for i in range(len(word)):
            index_to_update = (i, 0, letters_to_numbers[i])
            new_value = 1
            tensor[index_to_update] = new_value
            
        return tensor
    



class PositionEmbeddingArabicDataProcessor(AbstractArabicDataProcessor): 
    
    def __init__(self, file_directories : list):
        super.__init__(file_directories) 
    
    #along with one-hot encoding, implements Positional Embedding 
    def word_to_tensor(self, word : str) -> torch.Tensor: 
        case : int = self.word_analyser(word)
        pass #stub

    #set criteria for Positional Embedding, 
    # divide the cases into n number of cases and make this method return the case needed.
    def word_analyser(self, word : str) -> int:
        return 0 #stub
    
