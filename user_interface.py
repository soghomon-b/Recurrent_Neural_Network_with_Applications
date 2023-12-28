from  ArabicDataProcessor import *
from RNN import *
from training import *
from SentenceProcessor import SentenceProcessor

# change files to your data, make sure files are named the same as the category name
files = ["F:\\Python_Testing\\positive_words.txt", 
                "F:\\Python_Testing\\negative_words.txt",
                "F:\\Python_Testing\\neutral_words.txt"
                ]

trainer = Trainer(files) #instantiation of the training class

# uncomment if a different processor is needed
#model = SentenceProcessor(files)
#trainer.set_model(model)


trainer() #training 


#using the trained model
while True: 
    input = input("enter word: ")
    if (input == "-1"):
        break
    trainer.predict(input)
 
