# -*- coding: utf-8 -*-
"""
GRU POS Tagger.ipynb
Implementation of a GRU based POS Tagger.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext.datasets as datasets

import string
punctuations = string.punctuation

device = "cuda" if torch.cuda.is_available() else "cpu"

class Vocab():
  def __init__(self):
    self.train_iter = datasets.CoNLL2000Chunking(split='train')
    self.word2index = {}
    self.vocabulary = set()
    self.n_words = 0

    self.tag2index = {}
    self.index2tag = {}
    self.tags = set()
    self.n_tags = 0

    self.__buildVocab(self.train_iter)

  def removePunctuations(self, sample):
    observation_sequence, state_sequence = sample[0], sample[1]
    new_observation_sequence, new_state_sequence = [], []

    for i in range(len(state_sequence)):
      if (state_sequence[i] in punctuations) or (state_sequence[i] in ['$', "''", '(', ')', ':', '``']):
        continue
      else:
        new_observation_sequence.append(observation_sequence[i])
        new_state_sequence.append(state_sequence[i])
    
    return new_observation_sequence, new_state_sequence

  def __buildVocab(self, train_iter):
    print(f"Number of samples: {len(train_iter)}")

    for sample in train_iter:
      observation_sequence, state_sequence = self.removePunctuations(sample)

      for i, word in enumerate(observation_sequence):
        if state_sequence[i] not in self.tag2index:
          self.tag2index[state_sequence[i]] = self.n_tags
          self.index2tag[self.n_tags] = state_sequence[i]
          self.n_tags += 1
          self.tags.add(state_sequence[i])

        if word not in self.word2index:
          self.word2index[word] = self.n_words
          self.n_words += 1
          self.vocabulary.add(word)
  
  def sequence2tensor(self, sample):
    sequence_tensor = []
    target_tensor = []

    sequence, target = self.removePunctuations(sample)

    for i, word in enumerate(sequence):
      if word not in self.vocabulary:
        continue

      sequence_tensor.append(self.word2index[word])
      target_tensor.append(self.tag2index[target[i]])
    
    sequence_tensor = torch.tensor(sequence_tensor, dtype = torch.long, device = device)
    target_tensor = torch.tensor(target_tensor, dtype = torch.long, device = device)

    return sequence_tensor, target_tensor

class GRU_POS_Tagger(nn.Module):
  def __init__(self, vocab_size, hidden_size, tag_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)

    self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
    self.linear = nn.Linear(hidden_size, tag_size)
    self.softmax = nn.Softmax(dim = -1)
  
  def forward(self, input, hidden):
    input = self.embedding(input).unsqueeze(0)
    output, hidden = self.gru(input, hidden)
    output = self.softmax(self.linear(output))
    return output, hidden
  
  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size).to(device)

class BiGRU_POS_Tagger(nn.Module):
  def __init__(self, vocab_size, hidden_size, tag_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)

    self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True, bidirectional = True)
    self.linear = nn.Linear(2*hidden_size, tag_size)
    self.softmax = nn.Softmax(dim = -1)
  
  def forward(self, input, hidden):
    input = self.embedding(input).unsqueeze(0)
    output, hidden = self.gru(input, hidden)
    output = self.softmax(self.linear(output))
    return output, hidden
  
  def initHidden(self):
    return torch.zeros(2, 1, self.hidden_size).to(device)

def train(input_tensor, target_tensor, model, optimizer, criterion):
  model.train()
  hidden = model.initHidden()
  predicted, _ = model(input_tensor, hidden)
  loss = criterion(predicted[0], target_tensor)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss

def trainIter(model, lr, epochs, print_every, is_validate = False):
  optimizer = optim.SGD(model.parameters(), lr = lr)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(epochs):
    loss = 0
    train_iter = datasets.CoNLL2000Chunking(split='train')
    len_train_iter = len(train_iter)

    for i, sample in enumerate(train_iter):
      input_tensor, target_tensor = vocab.sequence2tensor(sample)
      loss += train(input_tensor, target_tensor, model, optimizer, criterion)

      if (i+1)%print_every == 0:
        print(f"Epoch : {epoch}/{epochs}, i: {i+1}/{len_train_iter}, Training Loss: {loss/print_every}")
        loss = 0
    
    if is_validate:
      validate(model, criterion)

def validate(model, criterion):
  model.eval()
  hidden = model.initHidden()
  valid_iter = datasets.CoNLL2000Chunking(split='train')
  len_valid_iter = len(valid_iter)
  loss = 0
  
  for sample in valid_iter:
    input_tensor, target_tensor = vocab.sequence2tensor(sample)
    predicted, _ = model(input_tensor, hidden)
    loss += criterion(predicted[0], target_tensor)
  
  print(f"Validation Loss: {loss/len_valid_iter}")
  print()

vocab = Vocab()
GRUTagger = GRU_POS_Tagger(vocab.n_words, 256, vocab.n_tags).to(device)
BiGRUTagger = BiGRU_POS_Tagger(vocab.n_words, 256, vocab.n_tags).to(device)

trainIter(BiGRUTagger, lr = 0.5, epochs = 10, print_every = 4000, is_validate = False)

import numpy as np

def evaluate(sample, model, is_print = False):
  model.eval()

  input_tensor, target_tensor = vocab.sequence2tensor(sample)

  if(input_tensor.size(0) == 0):
    return 100

  hidden = model.initHidden()
  output, hidden = model(input_tensor, hidden)

  _, max_indices = torch.max(output[0], dim = -1)

  predicted_output = []
  target_state = []

  for index in target_tensor:
    target_state.append(vocab.index2tag[index.item()])

  for index in max_indices:
    predicted_output.append(vocab.index2tag[index.item()])

  accuracy = 0
  for i in range(len(target_state)):
    if target_state[i] == predicted_output[i]:
      accuracy += 1

  accuracy = accuracy * 100/len(target_state)

  if is_print:
    print(predicted_output)
    print(target_state)
    print()

  return accuracy

# Evaluating BiGRU over testing dataset. Acheived accuracy ~85%
test_iter = datasets.CoNLL2000Chunking(split='test')
num_samples = len(test_iter)
num_samples_copy = num_samples
accuracy_track = []

for sample in test_iter:
  accuracy_track.append(evaluate(sample, BiGRUTagger, False))
  num_samples -= 1
  if num_samples <=0:
    break

mean_accuracy = np.array(accuracy_track).mean()
print(f"Mean accuracy of BiGRU_POS_Tagger over {num_samples_copy} samples is {mean_accuracy}%")

# Evaluating GRU over testing dataset. Acheived accuracy ~83%
test_iter = datasets.CoNLL2000Chunking(split='test')
num_samples = len(test_iter)
num_samples_copy = num_samples
accuracy_track = []

for sample in test_iter:
  accuracy_track.append(evaluate(sample, GRUTagger, False))
  num_samples -= 1
  if num_samples <=0:
    break

mean_accuracy = np.array(accuracy_track).mean()
print(f"Mean accuracy of GRUTagger over {num_samples_copy} samples is {mean_accuracy}%")