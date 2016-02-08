#!/usr/bin/env python3

import os
import argparse
import glob
import math

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--in_vocab", default="../data/baseline/vocab.txt")
parser.add_argument("--out_feature_data_dir", default="./data/train")
args = parser.parse_args()

FEATURE_NAME = "balance_shannon"

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
    vocab[w] = w_prob * math.log1p(w_prob)
  ln_N = math.log1p(total)
  return vocab, ln_N
  
def ExtractFeature(in_file, out_file, vocab, ln_N):
  for line in in_file:
    feature_val = 0
    tokens = line.split()
    for word in tokens:
      if word in vocab:
        feature_val += vocab[word]
    feature_val = -feature_val #/ ln_N
    out_file.write(str(feature_val) + "\n")

def main():
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  vocab, ln_N = LoadVocab(args.in_vocab)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), vocab, ln_N)
    

if __name__ == '__main__':
  main()
