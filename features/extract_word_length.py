#!/usr/bin/env python3

import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
args = parser.parse_args()

FEATURE_NAME = "word_length"

def ExtractFeature(in_file, out_file):
  for line in in_file:
    num_tokens = len(line.split())
    num_characters = 0
    for ch in line:
      num_characters += 1
    feature_val = num_characters/num_tokens
    out_file.write(str(feature_val) + "\n")

def main():
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"))
    

if __name__ == '__main__':
  main()
