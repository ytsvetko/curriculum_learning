#!/usr/bin/env python3

import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
args = parser.parse_args()

FEATURE_NAME = "type_token_ratio"

def ExtractFeature(in_file, out_file):
  for line in in_file:
    tokens = line.split()
    num_tokens = len(tokens)
    num_types = len(set(tokens))
    if num_tokens:
      feature_val = 1.0 * num_types / num_tokens
    else:
      feature_val = 0.0
    out_file.write(str(feature_val) + "\n")

def main():
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"))
    

if __name__ == '__main__':
  main()
