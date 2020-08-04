''' `torchtext` is a library which can do the following:
        - File loading: load in the corpus from various formats
        - Tokenization: break sentences into lists of words
        - Vocab: generate a vocabulary list
        - Numericalize: map words into integer numbers for the entire corpus
        - Word Vector: initialize vocabulary randomly
        - Batching: generate batches of training samples (with padding)
        - embedding lookup: map each sentence to fixed dimension word vectors

   -> (tokenization)
        ["The", "quick", ..., "dog" ]
   -> (vocab)
        {"The" -> 0,
         "quick" -> 1,
         ...
         "dog" -> 7
         }
   -> (numericalize)
        ["The", "quick", ..., "dog" ] -> [0, 1, ..., 7]
   -> (embedding lookup)
        [
           [0.3, 0.2, 0.5],
           [0.6, 0. , 0.1].
           ...
        ]
'''
import torch
from torchtext import data
from torchtext.vocab import GloVe
from torch import nn as tnn
import torch.optim as toptim

def main():
     ''' Steps:
          1. Preprocess into fields
          2. Use TabularDataset  to load .json
          3. Batch and pad using BucketIterator
          4. Define network, loss function, optimiser
          5. Train network
     '''
     # 0. Initialize PyTorch stuff
     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

     # 1. Preprocess - tokenize into fields
     tokenize = lambda x: x.split()
     textField = data.Field(lower=True, include_lengths=True, batch_first=True)
     labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

     # 2. Load .json
     fields = { 'reviewText':   ('reviewText', textField),
                    'rating':       ('rating',    labelField)  
          }
     dataset = data.TabularDataset('train.json', 'json', fields) 
     print("\n\n======== PRINTING dataset[0].__dict__.keys() =========")
     print(dataset[0].__dict__.keys())
     print(dataset[0].__dict__.values())
     print("======================================================\n\n")
     wordVectors = GloVe(name='6B', dim=50)
     textField.build_vocab(dataset, vectors=wordVectors)
     print("\n\n================ PRINTING textField.vocab ==============")
     print("The vocab is a dictionary which maps words to 50-dimension vectors.")
     print("Think of each dimension as a quality of the word, i.e. one parameter")
     print("could be `gender`. then, king - queen = 0 in all dimensions")
     print("as they are the the same thing, except for in the parameter")
     print("which described gender.")
     print(textField.vocab.vectors)
     print("======================================================\n\n")

     # 3. Batch and pad
     batchSize = 2
     trainLoader = data.BucketIterator(dataset, shuffle=True,
                                             batch_size=batchSize,
                                             sort_key=lambda x: len(x.reviewText),
                                             sort_within_batch=True)
     print("\n\n======= PRINTING batch.field for each batch in trainLoader =======")
     for batch in trainLoader:
          print(batch.reviewText, end=' --- ')
          print(batch.rating)
     print("======================================================\n\n")

     # 4. Define network, loss function, optimiser
     net = network().to(device)
     criterion = tnn.MSELoss()
     optimiser = toptim.SGD(net.parameters(), lr=0.01)
     epochs = 100

     # 5. Train.
     for epoch in range(epochs):
          runningLoss = 0
          for i, batch in enumerate(trainLoader):
               # Get a batch and potentially send it to GPU memory.
               inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
               length = batch.reviewText[1].to(device)
               labels = batch.rating.type(torch.FloatTensor).to(device)

               optimiser.zero_grad()

               # Forward pass through the network.
               output = net(inputs, length)
               loss = criterion(output, labels)

               # Calculate gradients.
               loss.backward()

               # Minimise the loss according to the gradient.
               optimiser.step()

               runningLoss += loss.item()

               if i % 32 == 31:
                    print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, runningLoss / 32))
                    runningLoss = 0

     net.eval()

     closeness = [0 for _ in range(5)]
     with torch.no_grad():
          for batch in trainLoader:
               # Get a batch and potentially send it to GPU memory.
               inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
               length = batch.reviewText[1].to(device)
               labels = batch.rating.type(torch.FloatTensor).to(device)

               # Convert network output to integer values.
               outputs = net(inputs, length).flatten()

               for i in range(5):
                    closeness[i] += torch.sum(abs(labels - outputs) == i).item()

     accuracy = [x / len(trainLoader) for x in closeness]
     score = 100 * (accuracy[0] + 0.4 * accuracy[1])

     print("\n"
          "Correct predictions: {:.2%}\n"
          "One star away: {:.2%}\n"
          "Two stars away: {:.2%}\n"
          "Three stars away: {:.2%}\n"
          "Four stars away: {:.2%}\n"
          "\n"
          "Weighted score: {:.2f}".format(*accuracy, score))

class network(tnn.Module):

    def __init__(self):
        super(network, self).__init__()

        self.lstm = tnn.LSTM(50, 256, 2, batch_first=True)
        self.fc1 = tnn.Linear(256, 64)
        self.fc2 = tnn.Linear(64, 16)
        self.fc3 = tnn.Linear(16, 1)
        self.sigmoid = tnn.Sigmoid()

    def forward(self, iput, length):
        data, states = self.lstm(iput)
        data = data.contiguous().view(-1, 256)
        data = self.fc1(data)
        data = self.fc2(data)
        data = self.fc3(data)
        data = self.sigmoid(data)
        data = data.view(len(length), -1)
        return data[:,-1]

if __name__ == '__main__':
    main()