# -*- coding: utf-8 -*-
"""Neural Machine Translation

This is the implementation of "Neural Machine Translation by Jointly Learning to Align and Translate" paper by Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014).

Paper available at: https://arxiv.org/abs/1409.0473

This paper was the first one to propose using attention mechanism in a basic encoder-decoder based translator to give more importance to relevant information in the source sentence. Although the authors were not able to state-of-the-arts results by a high margin, this paper did set the grounds for the "Attention is All You Need" paper (published in 2017). 

Some parts of the code was picked up from PyTorch's official tutorials, but it still different because this is the exact implementation of the proposed architecture. In PyTorch's tutorial, they have used a completely different attention mechanism. 

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
PAD_token = 2

# Class to store the word2index and index2word dictionary built on a corpus of some specific lanaguage
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "<PAD>"}
        self.n_words = 3  # Count SOS and EOS and <PAD>

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

# Function to lowercase, trim, and remove non-letter characters from a sentence
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

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence1(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    pad = (0, MAX_LENGTH - len(indexes))
    indexes = torch.tensor(indexes, dtype=torch.long, device=device)
    indexes = F.pad(indexes, pad, "constant", 2)
    return indexes

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Given a pair (source_sentence, target_sentence), convert them to corresponsing indicies to send into the encoder and decoder
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence1(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# Implementation of Encoder
class EncoderRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, bidirectional = True, batch_first = True)
  
  def forward(self, input, hidden):
    input = self.embedding(input).unsqueeze(0)
    output, hidden = self.gru(input, hidden)
    return output, hidden

  def initHidden(self):
    return torch.zeros(2, 1, self.hidden_size, device = device)

# Implementation of Decoder
class AttnDecoderRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size, dropout_p = 0.1, max_length = MAX_LENGTH):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.max_length = max_length
    self.vocab_size = vocab_size
    self.dropout_p = dropout_p

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(3*hidden_size, hidden_size)

    self.linear1 = nn.Linear(self.hidden_size*3, self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size, 1)
    self.softmax = nn.Softmax(dim = 0)

    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax1 = nn.LogSoftmax(dim=1)

  def forward(self, input, hidden, encoder_hidden):
    hidden_stack = torch.vstack([hidden[0] for i in range(self.max_length)])
    concat = torch.cat((hidden_stack, encoder_hidden[0]), 1)
    attn_weights = self.softmax(self.linear2(torch.tanh(self.linear1(concat))))
    context_vector = torch.bmm(torch.transpose(attn_weights, 0, 1).unsqueeze(0), encoder_hidden)

    input = self.embedding(input)
    input_concat = torch.cat((input, context_vector[0]), 1).unsqueeze(0)
    
    output, hidden = self.gru(input_concat, hidden)
    output = self.softmax1(self.out(output[0]))

    return output, hidden, attn_weights
  
  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device = device)

# Class to initialize the hidden vector at zeroth time step for decoder 
class encode_hidden(nn.Module):
  def __init__(self, hidden_size):
    super(encode_hidden, self).__init__()
    self.hidden_size = hidden_size
    self.linear = nn.Linear(hidden_size, hidden_size)
  
  def forward(self, hidden):
    hidden = torch.tanh(self.linear(hidden))
    return hidden

def AttnTrain(input_tensor, target_tensor, encoder, decoder, hidden_encoder, encoder_optimizer, decoder_optimizer, hidden_optimizer, criterion, max_length=MAX_LENGTH):
  encoder_hidden = encoder.initHidden()

  encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

  decoder_hidden = decoder.initHidden()

  decoder_hidden[0][0] = hidden_encoder(encoder_hidden[0])[0]
  
  decoder_input = torch.tensor([SOS_token], device = device)
  target_size = target_tensor.size(0)
  overall_loss = 0

  for i in range(target_size):
    decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
    loss = criterion(decoder_output, target_tensor[i])
    overall_loss += loss
    decoder_input = target_tensor[i]

  hidden_optimizer.zero_grad()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  overall_loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()
  hidden_optimizer.step()

  return overall_loss.item()/target_size

def trainIters(encoder, decoder, hidden_encoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01):
  encoder_optimizer = optim.SGD(encoder.parameters(), lr= learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr= learning_rate)
  hidden_optimizer = optim.SGD(decoder.parameters(), lr= learning_rate)

  criterion = nn.NLLLoss()
  training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

  plot_losses = []
  print_total_loss = 0
  plot_total_loss = 0

  for iter in range(1, n_iters+1):
    input_tensor, target_tensor = training_pairs[iter - 1]

    loss = AttnTrain(input_tensor, target_tensor, encoder, decoder, hidden_encoder, encoder_optimizer, decoder_optimizer, hidden_optimizer, criterion)
    print_total_loss += loss
    plot_total_loss += loss

    if iter%print_every == 0:
      print(f"iter: {iter}/{n_iters} \t loss: {print_total_loss/print_every}")
      print_total_loss = 0
    
    if iter%plot_every == 0:
      plot_losses.append(plot_total_loss / plot_every)
      plot_total_loss = 0
  
  return plot_losses

encoder = EncoderRNN(input_lang.n_words, 256).to(device)
decoder = AttnDecoderRNN(output_lang.n_words, 256).to(device)
hidden_encoder = encode_hidden(256).to(device)

loss_track = trainIters(encoder, decoder, hidden_encoder, 75000, print_every = 5000, plot_every = 1000)

def evaluate(encoder, decoder, hidden_encoder, sentence, max_length=MAX_LENGTH):
  with torch.no_grad():
    input_tensor = tensorFromSentence1(input_lang, sentence)

    encoder_hidden = encoder.initHidden()

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_hidden = decoder.initHidden()
    decoder_hidden[0][0] = hidden_encoder(encoder_hidden[0])[0]
    decoder_input = torch.tensor([SOS_token], device = device)

    decoded_words = []

    decoder_attentions = torch.zeros(max_length, max_length, device = device)

    for i in range(max_length):
      decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
      topv, topi = decoder_output.data.topk(1)

      decoder_attentions[i]  = torch.t(attn_weights)[0]

      if topi.item() == EOS_token:
          decoded_words.append('<EOS>')
          break
      else:
          decoded_words.append(output_lang.index2word[topi.item()])

      decoder_input = torch.tensor([topi.squeeze().detach()], device = device)
  
    return decoded_words, decoder_attentions

def evaluateRandomly(encoder, decoder, hidden_encoder, n=50):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attn_weights = evaluate(encoder, decoder, hidden_encoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print()
        print('')

evaluateRandomly(encoder, decoder, hidden_encoder)

evaluateRandomly(encoder, decoder, hidden_encoder)

torch.save(encoder.state_dict(), '/content/drive/MyDrive/data/NMT_encoder_weights.pth')
torch.save(decoder.state_dict(), '/content/drive/MyDrive/data/NMT_decoder_weights.pth')
torch.save(hidden_encoder.state_dict(), '/content/drive/MyDrive/data/NMT_hidden_encoder_weights.pth')