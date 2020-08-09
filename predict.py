import torch
import torch.nn as tnn
from torchtext import data
from torchtext.vocab import GloVe
import numpy as np
import student
import json

def predict(net, review, seq_length = 200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    words = preprocess(review)
    encoded_words = [vocab_to_int[word] for word in words]
    padded_words = pad_text([encoded_words], seq_length)
    padded_words = torch.from_numpy(padded_words).to(device)
    
    if(len(padded_words) == 0):
        "Your review must contain at least 1 word!"
        return None
    
    net.eval()
    h = net.init_hidden(1)
    output, h = net(padded_words, h)
    pred = torch.round(output.squeeze())
    msg = "This is a positive review." if pred == 0 else "This is a negative review."
    
    return msg

def main():
    while True:
        custom_input = input("Type your review!\n--> ")
        custom_input = student.preprocessing(custom_input)

        text = {'reviewText': custom_input, 'rating': 1.0}

        # 0. Initialize PyTorch stuff
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        with open('.custom_review.json', 'w') as f:
            json.dump(text, f)
        
        # 1. Preprocess - tokenize into fields
        textField = data.Field(lower=True, include_lengths=True, batch_first=True)
        labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

        # 2. Load .json
        fields = { 'reviewText':   ('reviewText', textField),
                       'rating':   ('rating',     labelField)  }
        dataset = data.TabularDataset('.custom_review.json', 'json', fields) 
        wordVectors = GloVe(name='6B', dim=300)
        textField.build_vocab(dataset, vectors=wordVectors)

        # 3. Batch and pad
        batchSize = 1
        trainLoader = data.BucketIterator(dataset, shuffle=True,
                                                batch_size=batchSize,
                                                sort_key=lambda x: len(x.reviewText),
                                                sort_within_batch=True)

        net = student.network().to(device)
        net.load_state_dict(torch.load('submitted.pth'))
        net.eval()

        with torch.no_grad():
          for batch in trainLoader:
               # Get a batch and potentially send it to GPU memory.
               inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
               length = batch.reviewText[1].to(device)
               labels = batch.rating.type(torch.FloatTensor).to(device)

               # Convert network output to integer values.
               outputs = net(inputs, length).flatten()
               rating = int(torch.argmax(outputs).cpu().data.numpy()) + 1

               print("==> " + str(rating) + " Star" + (rating != 1)*"s" + "!\n")



if __name__ == '__main__':
    main()