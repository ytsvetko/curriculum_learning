#!/usr/bin/env python3

import sys
import argparse
import json
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--in_vocab", default="../vocab.txt")
parser.add_argument("--oracle_matrix", default="/usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en")
parser.add_argument("--out_closures", default="./supersense_closures.txt")
parser.add_argument("--out_relative_freqs", default="./supersense_relative_freqs.txt")
parser.add_argument("--oracle_matrix_thresh", type=float, default=0.1)
args = parser.parse_args()

def LoadVocab(vocab_file):
  vocab = {}
  for line in open(vocab_file, "r"):
    word, freq = line.split()
    word = word.lower()
    if word in vocab:
      vocab[word] += float(freq)
    else:
      vocab[word] = float(freq)
  return vocab
 
def ReadOracleMatrix(filename, vocab, thresh):
  #filename format: headache  {"WN_noun.cognition": 0.5, "WN_noun.state": 0.5}
  supersense_words = collections.defaultdict(set)    
  supersense_freqs = collections.defaultdict(float)
  vocab_supersenses = {}
  for line in open(filename):
    word, json_line = line.strip().split("\t")
    if word not in vocab:
      continue
    word_freq = vocab[word]
    json_features = json.loads(json_line) 
    remove = set()  
    for k, v in json_features.items():
      if v < thresh:
        remove.add(k)
    for k in remove:
      del json_features[k]
    max_supersense_prob = 0
    max_supersense_name = None
    for supersense_name, supersense_val in json_features.items():
      supersense_words[supersense_name].add(word) 
      supersense_freqs[supersense_name] += supersense_val*word_freq
      if supersense_val > max_supersense_prob:
        max_supersense_prob = supersense_val
        max_supersense_name = supersense_name
    vocab_supersenses[word] = (max_supersense_name, max_supersense_prob)
  return vocab_supersenses, supersense_words, supersense_freqs
 
def main():
  out_file_freqs = open(args.out_relative_freqs, "w")
  out_file_closures = open(args.out_closures, "w")
  vocab = LoadVocab(args.in_vocab)
  vocab_supersenses, supersense_words, supersense_freqs = ReadOracleMatrix(args.oracle_matrix, vocab, args.oracle_matrix_thresh)
  print("Oracle matrix loaded")
  for supersense_name, words in supersense_words.items():
    out_file_closures.write("{}\t{}\t{}\n".format(supersense_name, str(supersense_freqs[supersense_name])," ".join(sorted(words))))
  print("Closures written")
    
  for line in open(args.in_vocab):
    word, freq = line.split()
    if word in vocab_supersenses:
      relative_freq = vocab_supersenses[word][1]*float(freq)/supersense_freqs[vocab_supersenses[word][0]]
      out_file_freqs.write("{}\t{}\t{}\n".format(word, str(relative_freq), vocab_supersenses[word][0]))
    

if __name__ == '__main__':
  main()
