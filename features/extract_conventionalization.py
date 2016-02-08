#!/usr/bin/env python3

import os
import argparse
import glob
import collections 

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--wiki_titles_filename", default="/usr1/home/ytsvetko/projects/curric/data/wiki-titles/titles.txt")
parser.add_argument("--vocab_filename", default="/usr1/home/ytsvetko/projects/curric/data/vocab.txt")

args = parser.parse_args()

FEATURE_NAME = "conventionalization"

def LoadVocab(vocab_file):
  vocab = set()
  for line in vocab_file:
    vocab.add(line.lower().split()[0])
  return vocab
  
def LoadTitlesFile(titles_file, vocab):
  result = collections.defaultdict(set)
  for i, line in enumerate(titles_file):
    if i % 10000 == 0:
      print(i)
    title = (line.lower().strip().split('" title="')[1]).split('">')[0].strip().lower()
    title_words = title.split()
    if not (set(title_words) - vocab):
      for word in title_words:
        result[word].add(title)
  return result
  
def ExtractFeature(in_file, out_file, titles):
  for line in in_file:
    line = line.lower()
    line_vocab = set(line.split())
    line_titles = set()
    for word in line_vocab:
      line_titles.update(titles[word])
    feature_val = 0.0
    for title in line_titles:
      if title in line:
        feature_val += 1
    out_file.write(str(feature_val) + "\n")

def main():
  vocab = LoadVocab(open(args.vocab_filename))
  titles = LoadTitlesFile(open(args.wiki_titles_filename), vocab)
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for i, in_file_name in enumerate(glob.glob(os.path.join(args.in_training_data_dir, "*"))):
    #if i % 1000 == 0:
    print(i)
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), titles)
    

if __name__ == '__main__':
  main()
