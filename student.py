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
import torch
from torch import round as rnd
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
    """

    """
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
wordVectors = GloVe(name='6B', dim=50)

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

    # zeros = np.zeros((*datasetLabel.shape, 5), dtype=np.float)

    # device = datasetLabel.device
    zeros = torch.zeros((*datasetLabel.shape, 5), dtype=torch.float)
    ints = datasetLabel.to(int) - 1
    zeros[np.arange(len(ints)), ints] = 1.0
    # zeros.index_put(datasetLabel, 1.0)
    return zeros.to(datasetLabel.device)

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """

    # return rnd(netOutput)

    return netOutput.argmax(axis=1).to(torch.float) + 1.0

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(50, 256, 2, batch_first=True)
        # self.fc0 = tnn.Linear(256*120, 256)
        self.fc1 = tnn.Linear(256, 64)
        # self.act = tnn.Tanh()
        self.act = tnn.ReLU()
        self.fc2 = tnn.Linear(64*120, 50)
        self.fc3 = tnn.Linear(50, 5)
        self.sm = tnn.Softmax(dim=1)

    def forward(self, iput, length):
        n_b = len(iput)
        padded_input = tnn.functional.pad(
            iput, [0, 0, 0, 120 - iput.shape[1]]
        ) # has shape [#batches, 120, 50]
        data, states = self.lstm(padded_input)
        # data = data.contiguous().view(data.shape[0], -)
        data = data.contiguous().view(n_b, 120, -1)
        # data = self.act(data)
        # data = self.fc0(data)
        data = self.fc1(data)
        data = self.act(data)
        data = data.view(n_b, -1)
        data = self.fc2(data)
        data = self.act(data)
        data = self.fc3(data)
        # data = data.view(len(length), 5)
        data = self.sm(data)
        return data

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, output, target):
        # targets = convertLabel(target)
        sqr_err = (output - target)**2
        mse = sqr_err.mean()
        return mse

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = tnn.BCELoss()
# lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
# optimiser = toptim.SGD(net.parameters(), lr=0.02)
optimiser = toptim.Adam(net.parameters(), lr=0.005)
