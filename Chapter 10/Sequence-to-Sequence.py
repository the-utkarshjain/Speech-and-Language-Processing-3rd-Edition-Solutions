# -*- coding: utf-8 -*-
"""Sequence to Sequence

This is the implementation of "Sequence to Sequence Learning with Neural Networks" paper by Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014).

Paper available at: https://arxiv.org/abs/1409.3215

This paper was the first one to propose using encoder-decoder model translate a source sentence. Although the paper was breakthrough in its time, it has a serious flaw. It tries to encode all the information in a fixed-size context vector. And although it works on smaller sequences, its performance drops while translating longer sequences (because it is unable to compress all that information in a small context vector). Soon after this a paper named "Neural Machine Translation" built on the proposed model and used an attention mechanism to deliver improved performance on longer sequences.

Framework used: PyTorch

Reference:
1. https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining some default tokens
SOS_token = 0
EOS_token = 1

# Class to store the word2index and index2word dictionary built on a corpus of some specific lanaguage
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Processing corpus to build the populate the "Lang" class
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('/content/drive/MyDrive/data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# Removing all pairs that have source or target sentence of lenght greater than MAX_LENGTH
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

# Implementation of Encoder
class EncoderRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1,1,-1)
    output, hidden = self.gru(embedded, hidden)
    return output, hidden
  
  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device = device)

# Implementation of Decoder
class DecoderRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(DecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.relu = nn.ReLU()
    self.gru  = nn.GRU(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax(dim = 1)
  
  def forward(self, input, hidden):
    input = self.relu(self.embedding(input).view(1,1,-1))
    output, hidden = self.gru(input, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden
  
  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device = device)

def tensorFromSentence(lang, input):
  sentence_tensor = [lang.word2index[word] for word in input.split(' ')]
  sentence_tensor.append(EOS_token)
  return torch.tensor(sentence_tensor, dtype =torch.long, device = device).view(-1, 1)

# Given a pair (source_sentence, target_sentence), convert them to corresponsing indicies to send into the encoder and decoder
def tensorsFromPair(pair):
  input_tensor = tensorFromSentence(input_lang, pair[0])
  output_tensor = tensorFromSentence(output_lang, pair[1])
  return (input_tensor, output_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
  encoder_hidden = encoder.initHidden()

  input_size = input_tensor.size(0)

  for i in range(input_size):
    encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)

  decoder_hidden = encoder_hidden
  decoder_input = torch.tensor([SOS_token], device = device)
  target_size = target_tensor.size(0)
  overall_loss = 0

  for i in range(target_size):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    loss = criterion(decoder_output, target_tensor[i])
    overall_loss += loss
    decoder_input = target_tensor[i]

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  overall_loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()

  return overall_loss.item()/target_size

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01):
  encoder_optimizer = optim.SGD(encoder.parameters(), lr= learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr= learning_rate)
  criterion = nn.NLLLoss()
  training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

  plot_losses = []
  print_total_loss = 0
  plot_total_loss = 0

  for iter in range(1, n_iters+1):
    input_tensor, target_tensor = training_pairs[iter - 1]

    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_total_loss += loss
    plot_total_loss += loss

    if iter%print_every == 0:
      print(f"iter: {iter}/{n_iters} \t loss: {print_total_loss/print_every}")
      print_total_loss = 0
    
    if iter%plot_every == 0:
      plot_losses.append(plot_total_loss / plot_every)
      plot_total_loss = 0 
    
  showPlot(plot_losses)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(output_lang.n_words, hidden_size).to(device)

trainIters(encoder1, decoder1, 75000, print_every=5000)

def evaluate(encoder, decoder, sentence, max_length = 10):
  input_tensor = tensorFromSentence(input_lang, sentence)
  input_size = input_tensor.size(0)
  encoder_hidden = encoder.initHidden()

  for i in range(input_size):
    encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
  
  decoder_hidden = encoder_hidden
  decoder_input = torch.tensor([SOS_token], device = device)
  decoder_words = []

  for i in range(MAX_LENGTH):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.topk(1)

    if topi.item() == EOS_token:
      decoder_words.append('<EOS>')
      break
    else:
      decoder_words.append(output_lang.index2word[topi.item()])
    
    decoder_input = torch.tensor([topi.item()], device = device)
  
  return decoder_words
    

def evaluateRandomly(encoder, decoder, n=10):
  for i in range(n):
      pair = random.choice(pairs)
      print('>', pair[0])
      print('=', pair[1])
      output_words = evaluate(encoder, decoder, pair[0])
      output_sentence = ' '.join(output_words)
      print('<', output_sentence)
      print('')

evaluateRandomly(encoder1, decoder1)

sentences = ["vous etes une fille interessante ."]

for sentence in sentences:
  print(sentence)
  print(evaluate(encoder1, decoder1, sentence))
  print()

