#!/usr/bin/env python3

import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--concreteness_filename", default="/usr1/home/ytsvetko/projects/curric/data/concreteness/ratings.txt")
args = parser.parse_args()

FEATURE_NAME = "concreteness"

def LoadConcretenessFile(concreteness_file):
  result_unigrams = {}
  result_bigrams = {}
  for line in concreteness_file:
    tokens = line.strip().split("\t")
    word = tokens[0]
    is_bigram = int(tokens[1])
    concreteness_rating = float(tokens[2])
    if is_bigram:
      assert word not in result_bigrams, line
      result_bigrams[word] = concreteness_rating
    else:
      assert word not in result_unigrams, line
      result_unigrams[word] = concreteness_rating
  return result_unigrams, result_bigrams

def ExtractFeature(in_file, out_file, concreteness_dict_unigrams, concreteness_dict_bigrams):
  def next_bigram(line):
    tokens = line.split()
    for bigram in zip(*[tokens[i:] for i in range(2)]):
      yield " ".join(bigram)
      
  for line in in_file:
    concreteness = 0.0
    total = 0   
    new_line = line
    for bigram in next_bigram(line):
      if bigram in concreteness_dict_bigrams:
        concreteness += concreteness_dict_bigrams[bigram]
        total += 1
        new_line = new_line.replace(bigram, "")
    for word in new_line.split():
      if word in concreteness_dict_unigrams:
        concreteness += concreteness_dict_unigrams[word]
        total += 1
    feature_val = 0.0
    if total > 0.0:
      feature_val = concreteness / total
    out_file.write(str(feature_val) + "\n")

def main():
  concreteness_dict_unigrams, concreteness_dict_bigrams = LoadConcretenessFile(open(args.concreteness_filename))
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), 
                   concreteness_dict_unigrams, concreteness_dict_bigrams)
    

if __name__ == '__main__':
  main()
