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
import torch
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

def read_stopwords():
    """ Attempts to read stop words from `my_stopwords.txt`, 
    if this fails a hard-coded set of words will be loaded.

    Returns the stopwords as a list.
    """
    words = []
    try:
        with open("my_stopwords.txt", "r") as f:
            for line in f:
                words.extend(line.split())
    except:
        words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
            'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'did', 'doing', 'a', 'an', 'the', 'and', 
            'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 
            'by', 'for', 'with', 'about', 'between', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 
            'out', 'on', 'off', 'over', 'under', 'then', 'once', 'here', 
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
            'can', 'will', 'just', 'don', "don't", 'should', "should've", 
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
            "aren't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'ma', 'mightn', "mightn't", 'needn', "needn't", 'shan', "shan't", 
            'shouldn', "shouldn't", 'wasn', 'weren', "weren't", 'won']

    return words

stopWords = set(read_stopwords())
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
    zeros = torch.zeros((*datasetLabel.shape, 5), 
                        dtype=torch.float)          # [[0., 0., 0., 0., 0.],...]
    ints = datasetLabel.to(int) - 1 
    zeros[np.arange(len(ints)), ints] = 1.0         # [[0., 0., 1., 0., 0.],...]
    return zeros.to(datasetLabel.device)

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # Find where the highest element is in each row
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
        # Use an LSTM to scan over the temporal input
        self.lstm = tnn.LSTM(300, 625, 2, batch_first=True)
        # Pass the output of the LSTM through
        # 3 fully layers (2 hidden + output layer)
        self.fc1 = tnn.Linear(625, 125)
        self.fc2 = tnn.Linear(125, 25)
        self.fc3 = tnn.Linear(25, 5)
        self.relu = tnn.ReLU()
        self.sm = tnn.Softmax(dim=1)

    def forward(self, input, length):
        # Apply LSTM and make timesteps sequential
        data, states = self.lstm(input)
        data = data.contiguous().view(-1, 625)

        # Propogate through dense layers and ReLU
        data = self.fc1(data)
        data = self.relu(data)
        data = self.fc2(data)
        data = self.relu(data)

        # Propogate through dense layer and Softmax
        data = self.fc3(data)
        data = self.sm(data)

        # Ignore all timesteps except the last
        data = data.view(len(length), -1)
        data = data[:,-5:]

        return data

class BCEPlus(tnn.BCELoss):
    """
    Combines BCE loss with a MSE loss weighed by the squared distance
    of each predicted probability from the target star. 
    ie, equivalent to:
        BCEPlus = BCELoss + ( [[  2,   1,   2,   3,   4], ...]**2
                            * [[0.2, 0.8, 0.5, 0.1, 0.2], ...] )

    This attempts to allocate higher loss to predictions further from 
    the correct star, aswell as use binary cross-entropy since this
    is also a classification problem.
    """
    def __init__(self):
        super(BCEPlus, self).__init__()

    def forward(self, output, target):
        # Calculate the normal BCE loss
        BCEloss = super(BCEPlus, self).forward(output, target)

        # Find the distance of stars from the target star,
        # ie, dist_from_target will be something like:
        #     [[3, 2, 1, 2, 3], [4, 3, 2, 1, 2], ...]**2
        # Where the 1 is at the position of the target prediction.
        target_idx = target.argmax(dim=1, keepdims=True)
        arange = torch.arange(5).repeat([len(target), 1]).to(output.device)
        dist_from_target = target_idx - arange
        dist_from_target = (dist_from_target.abs() + 1)**2
        dist_from_target = dist_from_target.to(torch.float)
        sqr_err = (output - target)**2
        # Weigh what will become the mse loss by the distances
        product = (dist_from_target * sqr_err)
        loss = product.mean() + BCEloss
        return loss

    def __str__(self):
        return super(BCEPlus, self).__str__() + \
            " [Uses dist**2 * sqr_err + BCE loss]"

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
device = device('cuda:0' if cuda.is_available() else 'cpu')
lossFunc = BCEPlus()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 1
batchSize = 32
epochs = 15
optimiser = toptim.Adam(net.parameters(), lr=0.0001)
