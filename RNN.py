import torch 
import torch.nn as nn 
import os
from abc import ABC, abstractmethod
from ArabicDataProcessor import *
import random
import time
import math

model = SimpleArabicDataProcessor(["F:\اسم فاعل.txt","F:\اسم مفعول.txt" ])
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


#prep for training 
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(model.categories)
    line = randomChoice(model.data_dict[category])
    category_tensor = torch.tensor([model.categories.index(category)], dtype=torch.long)
    line_tensor = model.word_to_tensor(line)
    return category, line, category_tensor, line_tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return model.categories[category_i], category_i

#training 

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.NLLLoss() 
n_hidden = 128
rnn = RNN(len(model.vocab), n_hidden, len(model.files))
f_category_tensor = torch.tensor([1], dtype=torch.long)
m_category_tensor = torch.tensor([2], dtype=torch.long)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


n_iters = 20000
print_every = 800
plot_every = 100



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        # Example iteration and logging to a text file
        to_print = '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct)
        #with open('logfile.txt', 'a', encoding='utf-8') as file:
            # Log a message for each iteration and append a new line
            #file.write(f"{to_print}\n")
        print(to_print)

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    input_line = input_line.replace(" ", "")  # Fix: assign the result back to input_line
    print('\n> %s' % input_line)

    with torch.no_grad():
        output = evaluate(model.word_to_tensor(input_line))

        if output.size(0) == 0:
            print("Error: Model output is empty.")
            return

        # Ensure n_predictions is within the valid range
        n_predictions = min(n_predictions, output.size(0))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, model.categories[category_index]))
            predictions.append([value, model.categories[category_index]])


# Assuming evaluate function and model.categories are defined elsewhere
print(all_losses)
while True:
    try:
        input_word = input("أدخل الكلمة: ")
        if input_word == "-1":
            break
        predict(input_word)
    except (Exception):
        pass
