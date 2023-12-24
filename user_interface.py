from  ArabicDataProcessor import *
from RNN import *
from training import *

files = ["F:\اسم فاعل.txt","F:\اسم مفعول.txt" ]  #enter the files for training 
trainer = Trainer(files) #initialising the trianer
print("training starting now")
trainer() #training
print("training ended, model ready to use")


while True: 
    input = input(" :أدخل الكلمة")
    trainer.predict(input)
    
