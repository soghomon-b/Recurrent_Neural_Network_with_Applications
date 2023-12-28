from  ArabicDataProcessor import *
from RNN import *
from training import *
from SentenceProcessor import SentenceProcessor
#comment out either 1 or 2 to run the other. If either is not commented, 1 will run until -1 is entered in the 
# input then 2 will run

#1. predicting the scale of the word
files = ["F:\اسم فاعل.txt","F:\اسم مفعول.txt" ]  #enter the files for training 
trainer = Trainer(files) #initialising the trianer
print("training starting now")
trainer() #training
print("training ended, model ready to use")


while True: 
    input = input(" :أدخل الكلمة")
    if (input == "-1"):
        break
    trainer.predict(input)
 


#2. training to identify word notion (positive, negative, neutral)
notion_files = ["F:\\Python_Testing\\positive_words.txt", 
                "F:\\Python_Testing\\negative_words.txt",
                "F:\\Python_Testing\\neutral_words.txt"
                ]
trainer = Trainer(notion_files)
model = SentenceProcessor(notion_files)
trainer.set_model(model)
trainer() #training 


while True: 
    input = input("enter word: ")
    if (input == "-1"):
        break
    trainer.predict(input)
 
