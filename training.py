
import random
from typing import Any 
from RNN import RNN
from ArabicDataProcessor import SimpleArabicDataProcessor
import torch 
from torch import nn 
import time 
import math


#training class for the RNN model
class Trainer():
    def __init__(self, files : list, learning_rate = 0.05, criterion = nn.NLLLoss(), n_hidden = 128, 
                 n_iters = 20000, print_every = 800, plot_every = 100  ):
        self.model = SimpleArabicDataProcessor(files)
        self.rnn = RNN(len(self.model.vocab), n_hidden, len(self.model.files))
        self.learning_rate = learning_rate 
        self.criterion = criterion
        self.n_hidden = n_hidden
        self.f_category_tensor = torch.tensor([1], dtype=torch.long)
        self.m_category_tensor = torch.tensor([2], dtype=torch.long)
        self.n_iters = n_iters
        self.print_every = print_every
        self.plot_every = plot_every
        self.current_loss = 0
        self.all_losses = []
        self.model_number = 0
        
    def set_model(self, model): 
        self.model = model

    #returns a random integer
    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]
    
    #returns a random category
    def randomTrainingExample(self):
        category = self.randomChoice(self.model.categories)
        line = self.randomChoice(self.model.data_dict[category])
        category_tensor = torch.tensor([self.model.categories.index(category)], dtype=torch.long)
        line_tensor = self.model.word_to_tensor(line)
        return category, line, category_tensor, line_tensor
    
    #returns the category of the ouput based on categories in the RNN
    def categoryFromOutput(self, output : torch.Tensor):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.model.categories[category_i], category_i


   


    #given a category tensor and a line tensor, it trains it using the RNN
    def train(self, category_tensor : torch.Tensor, line_tensor : torch.Tensor):
        hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.rnn.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)

        return output, loss.item()


    
    # calculates the time since "since"
    def timeSince(self, since : int):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    # performs the training self.n_iters times
    def perform_training(self):
        start = time.time() 
        for iter in range(1, self.n_iters + 1):
            category, line, category_tensor, line_tensor = self.randomTrainingExample()
            output, loss = self.train(category_tensor, line_tensor)
            self.current_loss += loss

            if iter % self.print_every == 0:
                guess, guess_i = self.categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                # Example iteration and logging to a text file
                to_print = '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / self.n_iters * 100, self.timeSince(start), loss, line, guess, correct)
                #with open('logfile.txt', 'a', encoding='utf-8') as file:
                    # Log a message for each iteration and append a new line
                    #file.write(f"{to_print}\n")
                print(to_print)
            if iter == self.n_iters:
                self.model_number += 1
                self.save_model('model'+ str(self.model_number) + ".pth")

            # Add current loss avg to list of losses
            if iter % self.plot_every == 0:
                self.all_losses.append(self.current_loss / self.plot_every)
                self.current_loss = 0
        

    # evaluates the model trained 
    def evaluate(self, line_tensor : torch.Tensor):
        hidden = self.rnn.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)

        return output

    #saves the model to use later
    def save_model(self, path : str):
        torch.save(self.rnn.state_dict(), path)
        print(f'Model saved to {path}')

    #reloads the model
    def load_model(self, path : str):
        # Assuming your model is an instance of a torch.nn.Module
        self.rnn.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')

    #given a model, it predits the output
    def predict(self, input_line : str, n_predictions=3):
        input_line = input_line.replace(" ", "")  # Fix: assign the result back to input_line
        print('\n> %s' % input_line)

        with torch.no_grad():
            output = self.evaluate(self.model.word_to_tensor(input_line))

            if output.size(0) == 0:
                print("Error: self.Model output is empty.")
                return

            # Ensure n_predictions is within the valid range
            n_predictions = min(n_predictions, output.size(0))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, self.model.categories[category_index]))
                predictions.append([value, self.model.categories[category_index]])
    
    
    def __call__(self) -> Any:
        self.perform_training()



#training a model and saving it
#trainer = Trainer(["F:\اسم فاعل.txt","F:\اسم مفعول.txt" ])
#trainer()