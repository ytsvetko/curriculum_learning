#!/usr/bin/env python

from __future__ import division
import sys
import argparse
from nltk.corpus import wordnet as wn
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--in_vocab", default="../vocab.txt")
parser.add_argument("--out_closures", default="./synset_relative_freqs.txt")
parser.add_argument("--aoa_filename", default="/usr1/home/ytsvetko/projects/curric/data/aoa/AoA.txt")
args = parser.parse_args()

POS_SET = set([wn.NOUN, wn.VERB, wn.ADJ, wn.ADV])
def LoadPOStags(aoa_file):
  result = {}
  for line in aoa_file:
    tokens = line.split()
    word = tokens[0]
    pos = tokens[2]
    if pos == "Adjective":
      pos = wn.ADJ
    elif pos == "Noun":
      pos = wn.NOUN
    elif pos == "Verb":
      pos = wn.VERB
    elif pos == "Adverb":
      pos = wn.ADV
    elif pos == "Name":
      pos = wn.NOUN
    if pos in POS_SET:
      result[word] = pos
  return result

def LoadVocab(vocab_file):
  vocab = {}
  for line in vocab_file:
    word, freq = line.split()
    word = word.lower()
    if word in vocab:
      vocab[word] += float(freq)
    else:
      vocab[word] = float(freq)
  return vocab
  
def GetClosure(word, vocab, pos_tags):
  def relative_frequency(closure):  
    total_freq = 0.0
    for w in closure:
      if w in vocab:
        total_freq += vocab[w]
    if total_freq > 0 :
      return 1.0*vocab[word]/total_freq
    return 0.0
    
  #nltk get closure
  closure = set([word])
  synsets = set([])
  try:
    if word in pos_tags:
      synsets = set(wn.synsets(word, pos_tags[word]))
  except:
    print word
    return set(), 0.0
  hypo = lambda s: s.hyponyms()
  hyper = lambda s: s.hypernyms()
  for synset in synsets:
    for w in synset.lemma_names():
      if w.isalpha():
        closure.add(w)
    for synset_hypo in synset.closure(hypo, depth=1):
      for w in synset_hypo.lemma_names():
        if w.isalpha():
          closure.add(w)
    for synset_hyper in synset.closure(hyper, depth=1):
      for w in synset_hyper.lemma_names():
        if w.isalpha():
          closure.add(w)
  return closure, relative_frequency(closure)
  
def main():
  out_file = codecs.open(args.out_closures, "w", "utf-8")
  vocab = LoadVocab(codecs.open(args.in_vocab, "r", "utf-8"))
  pos_tags = LoadPOStags(codecs.open(args.aoa_filename, "r", "utf-8"))
  for line in codecs.open(args.in_vocab, "r", "utf-8"):
    word, _ = line.split()
    closure, relative_freq = GetClosure(word.lower(), vocab, pos_tags)
    if relative_freq > 0.0:
      out_file.write(u"{}\t{}\t{}\n".format(word, str(relative_freq), " ".join(closure)))
    

if __name__ == '__main__':
  main()
