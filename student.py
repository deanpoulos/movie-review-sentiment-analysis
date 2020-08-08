#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch.nn as tnn
import torch.optim as toptim
from torch import round as rnd
from torch import transpose, Tensor, device, argmax, cuda, sqrt
from torchtext.vocab import GloVe
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
        `sample`: a list of words (as strings)
    # remove stopwords from sample

    sw = set(stopwords.words('english'))
    sample = [s for s in sample if s not in sw]

    # improve vocabulary by stemming, i.e., reduce 'jumping' to 'jump'
    ps = PorterStemmer()
    sample = [ps.stem(s) for s in sample]

    #print(" ".join(sample))

    """
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
        `vocab`: torchtext.vocab.Vocab object
    """

    return batch

stopWords = set(stopwords.words('english'))
wordVectors = GloVe(name='6B', dim=300)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """

    # Use a one-hot encoding
    newLabels = []
    for elem in datasetLabel:
        newLabel = [0,0,0,0,0]
        newLabel[int(elem - 1)] = 1
        newLabels.append(newLabel)

    return Tensor(newLabels).to(device)

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """

    # Take max-probability and return it
    newOutput = []
    for elem in netOutput:
        newOutput.append((elem == max(elem)).nonzero() + 1)

    newOutput = Tensor(newOutput).to(device)

    return newOutput

###########################################################################
################### The following determines the model ####################
###########################################################################

class lstm_network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    def __init__(self):
        super(network, self).__init__()
        # params
        self.vocab = 50
        self.layers = 2
        self.hidden = 256
        self.dropout = 0.1
        self.output = 5
        # layers
        self.lstm = tnn.LSTM(self.vocab, self.hidden, self.layers, 
                             batch_first=True, bidirectional=True, dropout=self.dropout)
        self.dropout = tnn.Dropout(self.dropout)
        self.fc = tnn.Linear(self.hidden, self.output)
        self.sigmoid = tnn.Sigmoid()

    def forward(self, input, length):
        batch_size = len(length)
                                                 # Input: [batch_size, seq_length, n_vocab ]
        lstm_out, h = self.lstm(input)                  # [batch_size, seq_length, n_hidden]
        lstm_out = self.dropout(lstm_out)       
        lstm_out = lstm_out.contiguous().view(-1, self.hidden)
                                                        # [batch_size*seq_length, n_hidden]
        fc_out = self.fc(lstm_out)                      # [batch_size*seq_length, n_output]
        sigmoid_out = self.sigmoid(fc_out)              # [batch_size*seq_length, n_output] 
        sigmoid_out = sigmoid_out.view(batch_size, -1)  # [batch_size, seq_length*n_output] 
        sigmoid_last = sigmoid_out[:,-5:]               # [batch_size, 5]

        return sigmoid_last

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(300, 625, 2, batch_first=True)
        self.fc1 = tnn.Linear(625, 125)
        self.r1 = tnn.ReLU()
        self.fc2 = tnn.Linear(125, 25)
        self.r2 = tnn.ReLU()
        self.fc3 = tnn.Linear(25, 5)
        self.sm = tnn.Softmax()

    def forward(self, input, length):
        # Apply LSTM and make timesteps sequential
        data, states = self.lstm(input)
        data = data.contiguous().view(-1, 625)

        # Propogate through dense layer and ReLU
        data = self.fc1(data)
        data = self.r1(data)

        # Propogate through dense layer and ReLU
        data = self.fc2(data)
        data = self.r2(data)

        # Propogate through dense layer and ReLU
        data = self.fc3(data)
        data = self.sm(data)

        # Ignore all timesteps except the last
        data = data.view(len(length), -1)
        data = data[:,-5:]

        return data

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        targetStars = argmax(target) + 1
        BCELoss = tnn.functional.binary_cross_entropy(output, target).to(device)
#        MSELoss = []
#        for i in range(len(target)):
#            ratingLoss = 0
#            for j in range(5):
#                x = target[i][j]
#                y = output[i][j]
#                k = argmax(target[i]+1).data
#                ratingLoss += sqrt((x-y)**2)  * abs(j - k)
#            MSELoss.append(ratingLoss)
#
#        MSELoss = Tensor(MSELoss).to(device)

        return BCELoss #+ MSELoss

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
device = device('cuda:0' if cuda.is_available() else 'cpu')
#lossFunc = tnn.BCELoss(Tensor([1,0.65,0.4,0.65,1]).to(device))
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.001)