#!/usr/bin/env python

import os
import argparse
import glob
import collections
import sys
import kenlm

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--lm", default="/usr0/home/ytsvetko/usr1/projects/curric/data/lm/char.5grams.binary")
args = parser.parse_args()

FEATURE_NAME = "char_lm_score"

def ExtractFeature(in_file, out_file, lm):
  for line in in_file:
    new_line = []
    for word in line.split():
      if word == "UNK":
        new_line.append(word)
      else:
        new_line.extend(" ".join(word))
    char_line = " ".join(new_line)
    feature_val = lm.score(char_line)
    out_file.write(str(feature_val) + "\n")

def main():
  lm = kenlm.LanguageModel(args.lm)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), lm)
    

if __name__ == '__main__':
  main()
