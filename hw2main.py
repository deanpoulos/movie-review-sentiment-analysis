#!/usr/bin/env python3
"""
hw2main.py
UNSW COMP9444 Neural Networks and Deep Learning
DO NOT MODIFY THIS FILE
"""

import torch
from torchtext import data
import sys

import student

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    student.device = device
    print("Using device: {}"
          "\n".format(str(device)))

    # Load the training dataset, and create a dataloader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           preprocessing=student.preprocessing,
                           postprocessing=student.postprocessing,
                           stop_words=student.stopWords)
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

    dataset = data.TabularDataset('train.json', 'json',
                                 {'reviewText': ('reviewText', textField),
                                  'rating': ('rating', labelField)})

    textField.build_vocab(dataset, vectors=student.wordVectors)

    # Allow training on the entire dataset, or split it for training and validation.
    if student.trainValSplit == 1:
        trainLoader = data.BucketIterator(dataset, shuffle=True,
                                          batch_size=student.batchSize,
                                          sort_key=lambda x: len(x.reviewText),
                                          sort_within_batch=True)
    else:
        train, validate = dataset.split(split_ratio=student.trainValSplit,
                                        stratified=True, strata_field='rating')

        trainLoader, valLoader = data.BucketIterator.splits(
            (train, validate), shuffle=True, batch_size=student.batchSize,
             sort_key=lambda x: len(x.reviewText), sort_within_batch=True)

    # Get model and optimiser from student.
    net = student.net.to(device)
    criterion = student.lossFunc
    optimiser = student.optimiser

    # Train.
    try:
        for epoch in range(student.epochs):
            runningLoss = 0

            for i, batch in enumerate(trainLoader):
                # Get a batch and potentially send it to GPU memory.
                inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                length = batch.reviewText[1].to(device)
                labels = batch.rating.type(torch.FloatTensor).to(device)

                # PyTorch calculates gradients by accumulating contributions
                # to them (useful for RNNs).
                # Hence we must manually set them to zero before calculating them.
                optimiser.zero_grad()

                # Forward pass through the network.
                output = net(inputs, length)
                loss = criterion(output, student.convertLabel(labels))

                # Calculate gradients.
                loss.backward()

                # Minimise the loss according to the gradient.
                optimiser.step()

                runningLoss += loss.item()

                if i % 32 == 31:
                    print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                        % (epoch + 1, i + 1, runningLoss / 32))
                    runningLoss = 0
    except KeyboardInterrupt:
        print("Stopping training.")
        pass
    # Save model.
    torch.save(net.state_dict(), 'savedModel.pth')
    print("\n"
          "Model saved to savedModel.pth")

    # Test on validation data if it exists.
    if student.trainValSplit != 1:
        net.eval()

        dist = [0, 0, 0, 0, 0]
        closeness = [0 for _ in range(5)]
        with torch.no_grad():
            for batch in valLoader:
                # Get a batch and potentially send it to GPU memory.
                inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                length = batch.reviewText[1].to(device)
                labels = batch.rating.type(torch.FloatTensor).to(device)

                # Convert network output to integer values.
                outputs = student.convertNetOutput(net(inputs, length)).flatten()
                for output in outputs:
                  dist[int(output.cpu().data.numpy())-1] += 1
                for i in range(5):
                    closeness[i] += torch.sum(abs(labels - outputs) == i).item()

        accuracy = [x / len(validate) for x in closeness]
        score = 100 * (accuracy[0] + 0.4 * accuracy[1])

        scores = ("\n"
              "Correct predictions: {:.2%}\n"
              "One star away: {:.2%}\n"
              "Two stars away: {:.2%}\n"
              "Three stars away: {:.2%}\n"
              "Four stars away: {:.2%}\n"
              "\n"
              "Weighted score: {:.2f}").format(*accuracy, score)
        print(scores)

        numOutputs = sum(dist)
        print(dist)
        print(numOutputs)
        dist = [x / numOutputs for x in dist]

        prediction_distribution = ("\n"
              "5 Stars: {:.2%}\n"
              "4 Stars: {:.2%}\n"
              "3 Stars: {:.2%}\n"
              "2 Stars: {:.2%}\n"
              "1 Star : {:.2%}\n").format(*dist)
            
        print(prediction_distribution)
              
    # Add to log
    if '--nolog' not in sys.argv:
        with open("log.txt", "a") as f:
            f.write("Model:\n")
            f.write(str(net))
            f.write("\n\nEpochs: {}\n".format(student.epochs))
            f.write("Word vec dim: {}\n".format(student.wordVectors.dim))
            f.write("\nOptimiser: {}\n".format(student.optimiser))
            f.write("\nResults:\n")
            f.write(scores)
            f.write("\n===========\n")


if __name__ == '__main__':
    main()