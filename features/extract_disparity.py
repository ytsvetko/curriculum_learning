#!/usr/bin/env python3

import os
import argparse
import glob
from scipy import spatial

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--in_vocab", default="../data/baseline/vocab.txt")
parser.add_argument("--in_baseline_vectors", default="../data/baseline/baseline.wiki.cbow.vectors")
parser.add_argument("--out_feature_data_dir", default="./data/train")
args = parser.parse_args()

FEATURE_NAME = "disparity"

WORD_VECTORS = {}

def LoadVocab(vocab_file):
  vocab = {}
  total = 0
  for line in open(vocab_file):
    #the 4831127
    word, freq = line.split()
    freq = float(freq)
    vocab[word] = freq
    total += freq
  # normalize to probabilities
  for w, f in vocab.items():
    w_prob = f/total
    vocab[w] = w_prob
  return vocab

def Cosine(word1, word2):
  w1 = WORD_VECTORS[word1]
  w2 = WORD_VECTORS[word2]
  return spatial.distance.cosine(w1, w2)
  
def LoadVectors(vectors_file):
  vectors = {}
  for line in open(vectors_file):
    tokens = line.split()
    if len(tokens) < 3:
      continue
    word = tokens[0]
    vector = [float(n) for n in tokens[1:]]
    vectors[word] = vector
  return vectors
    
def ExtractFeature(in_file, out_file, vocab):
  for line in in_file:
    feature_val = 0
    seen_pairs = set()
    tokens = line.split()
    for word1 in tokens:
      for word2 in tokens:
        if word1 not in vocab or word2 not in vocab:
          continue
        if word1 != word2 and (word1, word2) not in seen_pairs and (word2, word1) not in seen_pairs:
          seen_pairs.add((word1, word2))
          seen_pairs.add((word2, word1))
          feature_val += vocab[word1]*vocab[word2]*Cosine(word1, word2)
    out_file.write(str(feature_val) + "\n")

def main():
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  vocab = LoadVocab(args.in_vocab)
  global WORD_VECTORS 
  WORD_VECTORS = LoadVectors(args.in_baseline_vectors)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), vocab)
    

if __name__ == '__main__':
  main()
