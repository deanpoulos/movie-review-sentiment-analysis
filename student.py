#!/usr/bin/env python3
"""
`student.py`
UNSW COMP9444 Neural Networks and Deep Learning
group/PhiTorch: Dean Poulos & Leo Carnovale
                (z5122508)    (z5159960)
"""

"""
###########################################################################
######################## How the program works ############################
###########################################################################

This implementation works as follows:
    1. Removes custom stopwords from review text input data
    2. Establishes pre-trained 300-dimension word-vectors using GloVe 6B.
    3. Converts rating label data into probability distribution vectors using 
       a one-hot encoding.
    4. Propagates batches of word-vector reviews through an LSTM network, 
       followed by multiple dense layers with ReLU activation and then outputs 
       the last timestep of each review through a Softmax activation.
    5. Backpropogates to train the network using a custom loss function, 
       consisting of a cross-entropy metric and a weighted MSE metric using an 
       AdaM optimiser.
    6. Iterates over data for 15 epochs using the entire training set.

###########################################################################
############################# Preprocessing ###############################
###########################################################################

We determined empirically that the removal of stopwords and stemming of words 
did not significantly increase performance. One major drawback of removing 
stopwords was the omission of important negation words which are common in 
reviews and imply the negation of a predicted sentiment. So, the implementation 
includes `nltk`’s english stopword list with the exclusion of many negation 
words. The `preprocessing` function also implements word stemming however 
it is commented out due to an observed decrease in network accuracy.

###########################################################################
########################### Network Structure #############################
###########################################################################

--> Why use an LSTM implementation of RNN?

Long short-term memory (LSTM) is an artificial recurrent neural network 
(RNN) architecture LSTM has feedback connections beyond simple feedforward 
networks. So, LSTMs are a great candidate for processing sequences of 
words as in a sentiment analysis classification problem. 

Furthermore, a significant issue in classification tasks like these is the 
vanishing gradient problem. Activation architectures in RNNs such as LSTMs 
and GRUs have been shown to successfully overcome the vanishing gradient 
problem [Hochreiter et al., 1997], [Gers et al., 2002], [Cho et al., 2014]. 

Both the LSTM and the GRU solve the vanishing gradient problem by 
re-parameterizing the RNN; The input to the LSTM cell is multiplied by the 
activation of an input gate, and the previous values are multiplied by a 
forget gate, the network only interacts with the LSTM unit via gates. GRU 
simplifies the LSTM architecture by combining the forget and input gates 
into an update gate and merging the cell state with the hidden state. Thus, 
we have chosen LSTM as the preferred choice of RNN.

The actual structure of our network was the result of an evolution from an 
LSTM layer followed by 2 or 3 fully connected layers. Many different 
configurations were tested, the shaping of the data throughout the model as 
well as parameters of the model itself (number of hidden nodes in layers, etc) 
were all changed to see if an improvement was possible. We found that the 
current implementation achieved the highest weighted score. Initially, 
one-hot encoding was quickly found to produce better results than the 
original regression based network we implemented. 

We then found further improvements from a custom loss function, detailed below. 
The vast majority of significant changes in the model itself did not provide a 
large improvement in accuracy, and it was the result of many small changes to 
LSTM parameters, hidden layer node counts, and the number of hidden layers, 
that brought the model to its current accuracy. We also experimented with 
dropout layers, in both the lstm output and the inputs during training. We 
hoped that these would help prevent the network from over-fitting, and allow 
the network to better identify various features in the data. As with most 
other changes, these made little improvement and usually resulted in a lower 
accuracy after 10-20 epochs.

--> Why use a Stacked RNN Architecture?

Research on stacked RNN architectures has shown it to be remarkably successful 
in NLP tasks in comparison to single RNNs [Irsoy et al., 2014]. Multi-stacked 
architecture operates on different time scales; the lower level layer captures 
short term interaction while the aggregate effects are captured by the high 
level layers. [Hermans et al.,2013]

--> Why we use softmax?

Softmax function takes an N-dimensional vector and maps it to a vector of 
floats in the range [0,1] which add up to 1. This vector satisfies the condition 
for to a probability distribution which is suitable for probabilistic 
interpretation of a classification task like sentiment analysis.

###########################################################################
########################### Optimiser Choice ##############################
###########################################################################

Some research suggests LSTM benefit from loss functions which closely emulate 
the behaviour of LSTMs themselves, i.e., by using a Stochastic Gradient Descent 
(SGD) optimiser [Andrychowicz et al., 2016]. SGD also known as incremental 
gradient descent tries to determine minimum error via iteration, however 
suffers from convergence to a local minimum when the objective function is not 
convex or pseudo convex. Since SGD is insufficient in escaping these troughs in
the loss landscape, where they oscillate across slopes and slowly progress to 
the bottom, we will not be using this optimizer for our LSTM network. 

Other research has shown that when compared amongst various optimiser choices, 
adaptive momentum is a favourable strategy for descending complex objective 
functions, and more specifically, the AdaM variant.
[Bikal Basnet, DeepDataScience Wordpress, 2016]

###########################################################################
############################# Loss Function ###############################
###########################################################################

Given a probability distribution as `output` and integer star quantity as 
our `target`, we convert the target into a probability distribution by using
a one-hot encoding. This enables us to use a probabilistic approach which is 
more similar to a regression problem with a weighted score than a simple 
classification problem. A natural measure of the difference between two 
probability distributions is their entropy, and so a cross-entropy loss 
function was chosen to be a major component of our loss function, implemented 
via PyTorch’s `nn.BCELoss()`. 

However, since the weighted score rewards predictions that are one star away,
then the classification accuracy is not sufficient for maximising this score. 
We have chosen to use a custom loss function which adds a scaled MSE to the 
BCE loss. The scaled MSE loss component of our loss function multiplies the 
standard MSE with the squared distance of the star from the target star. 
This component now increases the amplitude of the weight adjustments associated
with rating predictions which are far away from the target star. This 
incentivises the network to learn patterns which do not just lead it to the 
correct classification (BCE), but away from the incorrect score. 

This choice is suitable for making predictions about classes which are not 
independent.

"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torch import round as rnd
from torch import transpose, Tensor, device, argmax, cuda, sqrt
from torchtext.vocab import GloVe
import nltk
from nltk.stem import PorterStemmer
import numpy as np

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    """
    # remove stopwords from sample
    sw = set(stopwords.words('english'))
    sample = [s for s in sample if s not in sw]

    # improve vocabulary by stemming, i.e., reduce 'jumping' to 'jump'
    ps = PorterStemmer()
    sample = [ps.stem(s) for s in sample]
    """
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    return batch

def read_stopwords():
    """ Attempts to read stop words from `my_stopwords.txt`, 
    if this fails a hard-coded set of words will be loaded.

    Returns the stopwords as a list.
    """
    words = []
    try:
        # read from file for local execution
        with open("my_stopwords.txt", "r") as f:
            for line in f:
                words.extend(line.split())
    except:
        # otherwise, use hard-coded stopwords, based on nltk.stopwords.
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
    # transform simple classification to vector using a one-hot encoding.
    zeros = torch.zeros((*datasetLabel.shape, 5), dtype=torch.float)          
                                                    # [0., 0., 0., 0., 0.]
    ints = datasetLabel.to(int) - 1 
    zeros[np.arange(len(ints)), ints] = 1.0         # [0., 0., 1., 0., 0.]

    return zeros.to(datasetLabel.device)

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # find where the highest element is in each row and return that index.
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
    # Parameters
    n_hidden    = 625
    n_vocab     = wordVectors.dim # 300
    n_stacked   = 2
    n_output    = 5

    def __init__(self):
        super(network, self).__init__()

        self.lstm = tnn.LSTM(n_vocab, n_hidden, n_stacked, batch_first=True)
        self.fc1 = tnn.Linear(n_hidden, n_hidden/5)
        self.fc2 = tnn.Linear(n_hidden/5, n_hidden/25)
        self.fc3 = tnn.Linear(n_hidden/25, n_output)
        self.relu = tnn.ReLU()
        self.sm = tnn.Softmax(dim=1)

    def forward(self, input, length):

        batch_size = len(length) # `input` [batch_size, seq_length, n_vocab]

        # Use LSTM to scan over temporal input
        data, states = self.lstm(input) # [batch_size, seq_length, n_hidden]
        data = data.contiguous()
        data = data.view(-1, n_hidden)  # [batch_size*seq_length,  n_hidden]

        # Pass output of the LSTM through 3 dense layers (2 hidden, 1 output)
        data = self.relu(fc1(data))       # [batch_size*seq_length, n_hidden/5]
        data = self.relu(fc2(data))       # [batch_size*seq_length, n_hidden/25]
        data = self.sm(fc3(data))         # [batch_size*seq_length, n_output]

        # Output the final timestep of the LSTM
        data = data.view(batch_size, -1) # [batch_size, seq_length*n_output]  
        data = data[:,-5:]               # [batch_size, 5]

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
        target_idx = target.argmax(dim=1, keepdims=True)
        arange = torch.arange(5).repeat([len(target), 1]).to(output.device)

        # i.e., dist_from_target will be something like:
        #     [[3, 2, 1, 2, 3], [4, 3, 2, 1, 2], ...]**2
        # where the 1 is at the position of the target prediction.
        dist_from_target = target_idx - arange
        dist_from_target = (dist_from_target.abs() + 1)**2
        dist_from_target = dist_from_target.to(torch.float)

        # Calculate the standard MSE
        sqr_err = (output - target)**2

        # Weigh what will become the mse loss by the distances
        product = (dist_from_target * sqr_err)

        # Add our weighted MSE to BCE
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
