# -*- coding: utf-8 -*-
"""Attention Is All You Need

This is the implementation of "Attention Is All You Need" paper by Vaswani et. al (2017).

Paper available at: https://arxiv.org/abs/1706.03762

This is the paper that revolutionized the neural machine translation task and is widely used in tranlation system today. It not only delivers state-of-the-art results, but do so while avoiding the slow and sequential processing of a RNN. The proposed architecture eschews using RNNs at all, and only uses Scaled-Dot-Product Attention Mechanism. Because of its parallel processing, it only requires a fraction of training time wrt previous state-of-the-art models to acheive better results. This is one of the best papers I have implemented so far and I hope you implement it too.

PS: I have not used the proposed model exactly. For example, I have used SGD optimizer rather than Adam. And I did not use label smoothing. Also, because of limited computational resources, I could not train the model for long enough to get good results.

Framework used: PyTorch

Reference:
1. https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context = "talk")
# %matplotlib inline

device = "cuda" if torch.cuda.is_available() else "cpu"

#Defining Start-of-Statement (SOS), End-of-Statement (EOS), Pad (PAD) tokens
SOS_token = 0
EOS_token = 1
PAD_token = 2

#Class for storing the word2index and index2word dictionaries given a corpus from a language. Taken from PyTorch tutorial for NLP (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model)
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "<PAD>"}
        self.n_words = 3  # Count SOS and EOS

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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    function to lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

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

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    pad = (0, MAX_LENGTH - len(indexes))
    indexes = torch.tensor(indexes, dtype=torch.long, device=device)
    indexes = F.pad(indexes, pad, "constant", 2)
    return indexes

def tensorsFromPair(pair):
    """
    Given a pair of input and target tensors, generate the corresponding tensor indexes representation.
    """
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor.unsqueeze(0), target_tensor.unsqueeze(0))

def attention(query, key, value, mask = None):
  """
  input:
    query, key, value, mask: tensors of 4 dimensional space
  return:
    attention: tensor of 4D space
    attn_weights: tensor of 4D space
  """
  d_k = query.size(-1)
  softmax_arg = torch.matmul(query, torch.transpose(key, -2, -1))/math.sqrt(d_k)

  if mask is not None:
    attn_weights = F.softmax(softmax_arg.masked_fill(mask == 0, -1e9), dim = -1)
  else:
    attn_weights = F.softmax(softmax_arg, dim = -1)
  
  attention_applied = torch.matmul(attn_weights, value)
  return attention_applied, attn_weights

def subsequent_mask(n_rows):
  """
  input : (int) number of words in the input
  return: a tensor of shape (1, n_rows, n_rows)
  """
  shape = (1, n_rows, n_rows)
  mask = torch.ones(shape, device = device)
  mask = torch.tril(mask, diagonal = 0)
  return mask

def clone(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout = 0.1, n_rows = MAX_LENGTH):
    super(PositionalEncoding, self).__init__()
    self.n_rows = n_rows
    self.d_model = d_model
    self.positional_encoding = torch.zeros((self.n_rows, self.d_model), device = device)
    self.positions = torch.arange(0, self.n_rows, device = device).unsqueeze(1)
    self.dropout = nn.Dropout(dropout)

    i = torch.arange(0, d_model, step =  2, device = device)

    self.pos_denominator = self.positions*torch.exp(-1*i*math.log(10000)/self.d_model)
    self.positional_encoding[:, 0::2] = torch.sin(self.pos_denominator)
    self.positional_encoding[:, 1::2] = torch.cos(self.pos_denominator)
    self.positional_encoding.unsqueeze_(0) #(1, words, d_model)

  def forward(self, embedding):
    """
    embedding: (batch, n_words, embed_dim)
    returns: (batch, n_words, embed_dim)
    """
    return self.dropout(self.positional_encoding + embedding)

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, d_model, dropout = 0.1):
    super(MultiHeadAttention, self).__init__()
    self.n_heads = n_heads
    self.d_k = d_model // self.n_heads
    self.layers = clone(nn.Linear(d_model, d_model), 4)
    self.dropout = nn.Dropout(dropout)
    self.layernorm = nn.LayerNorm(d_model, elementwise_affine = True)

  def forward(self, query, key, value, mask = None):
    query_residual = query
    n_batches, n_rows, n_cols = query.size()
    n_cols = n_cols // self.n_heads
    
    if mask is not None:
      mask =  mask.unsqueeze(0)

    query, key, value = [l(x).view(n_batches, -1, self.n_heads, n_cols).transpose(1, 2) for l,x in zip(self.layers, (query, key, value))]
    attention_applied, attn_weights = attention(query, key, value, mask)
    attention_applied = attention_applied.contiguous().view(n_batches, -1, self.n_heads*self.d_k)

    output = query_residual + self.dropout(self.layers[-1](attention_applied))
    return self.layernorm(output)

class Linear(nn.Module):
  def __init__(self, d_model, dropout = 0.1):
    super(Linear, self).__init__()
    self.layers = nn.ModuleList([ nn.Linear(d_model, d_model*4),
                                  nn.ReLU(),
                                  nn.Linear(d_model*4, d_model),
                                  nn.Dropout(dropout) ])
    self.layernorm = nn.LayerNorm(d_model, elementwise_affine = True)

  def forward(self, input):
    input_residual = input
    for layer in self.layers:
      input = layer(input)
    
    output = input_residual + input
    return self.layernorm(output)

class EncoderLayer(nn.Module):
  def __init__(self, n_heads, d_model):
    super(EncoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(n_heads, d_model)
    self.linear = Linear(d_model)
  
  def forward(self, input):
    return self.linear(self.self_attention(input, input, input, mask = None))

class DecoderLayer(nn.Module):
  def __init__(self, n_heads, d_model):
    super(DecoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(n_heads, d_model)
    self.multiheadattention = MultiHeadAttention(n_heads, d_model)
    self.linear = Linear(d_model)
  
  def forward(self, input, encoder_output):
    mask = subsequent_mask(input.size(-2))
    output = self.self_attention(input, input, input, mask = mask)
    output = self.multiheadattention(output, encoder_output, encoder_output, mask = None)
    return self.linear(output)

class Encoder(nn.Module):
  def __init__(self, Nx, n_heads, vocab_size, d_model):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positionalencoding = PositionalEncoding(d_model)
    self.encoderlayers = clone(EncoderLayer(n_heads, d_model), Nx)

  def forward(self, input):
    #accepts input in form of (batch, words)
    input = self.positionalencoding(self.embedding(input))
    for encoderlayer in self.encoderlayers:
      input = encoderlayer(input)
    return input

class Decoder(nn.Module):
  def __init__(self, Nx, n_heads, vocab_size, d_model):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positionalencoding = PositionalEncoding(d_model)
    self.decoderlayers = clone(DecoderLayer(n_heads, d_model), Nx)
    self.linear = nn.Linear(d_model, vocab_size)
    self.softmax = nn.LogSoftmax(dim = -1)
  
  def forward(self, input, encoder_output):
    input = self.positionalencoding(self.embedding(input))
    for decoderlayer in self.decoderlayers:
      input = decoderlayer(input, encoder_output)
    input = self.softmax(self.linear(input))
    return input

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
  encoder_output = encoder(input_tensor)
  decoder_output = decoder(target_tensor, encoder_output)

  loss = criterion(decoder_output.view(-1, output_lang.n_words), target_tensor.view(-1))

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss

def TrainIters(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.0001):
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
  
  return plot_losses

Nx = 6
n_heads = 8
d_model = 512

encoder = Encoder(Nx, n_heads, input_lang.n_words, d_model).to(device)
decoder = Decoder(Nx, n_heads, output_lang.n_words, d_model).to(device)

loss_track = TrainIters(encoder, decoder, 75000, print_every = 5000, plot_every = 1000)