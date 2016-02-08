#!/usr/bin/env python3

import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--relative_freq_filename", default="/usr1/home/ytsvetko/projects/curric/data/wordnet/synset_relative_freqs.txt")
args = parser.parse_args()

FEATURE_NAME = "synset_relative_freq"

def LoadFreqFile(relative_freq_file):
  result = {}
  for line in relative_freq_file:
    tokens = line.split()
    word = tokens[0]
    relative_freq = float(tokens[1])
    assert word not in result, line
    result[word] = relative_freq
  return result
  
def ExtractFeature(in_file, out_file, relative_freq_dict):
  for line in in_file:
    total = 0.0
    words = line.split()
    for word in words:
      if word in relative_freq_dict:
        total += relative_freq_dict[word]
    feature_val = 0.0
    if total > 0.0:
      feature_val = total / len(words)
    out_file.write(str(feature_val) + "\n")

def main():
  relative_freq_dict = LoadFreqFile(open(args.relative_freq_filename))
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), relative_freq_dict)
    

if __name__ == '__main__':
  main()
