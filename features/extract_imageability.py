#!/usr/bin/env python3

import os
import argparse
import glob
import json 

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--imageability_filename", default="/usr1/home/ytsvetko/projects/curric/data/mrc/imageability.predictions")
args = parser.parse_args()

FEATURE_NAME = "imageability"

def LoadImageabilityFile(imageability_file):
  result = {}
  for line in imageability_file:
    tokens = line.strip().split("\t")
    word = tokens[0]
    imageability_rating = float(json.loads(tokens[2])["I"])
    assert word not in result, line
    result[word] = imageability_rating
  return result

def ExtractFeature(in_file, out_file, imageability_dict):
  #tweaks	U	{"U": 0.58666666666666667, "I": 0.41333333333333333}
  for line in in_file:
    imageability = 0.0
    total = 0   
    for word in line.split():
      if word in imageability_dict:
        imageability += imageability_dict[word]
        total += 1
    feature_val = 0.0
    if total > 0.0:
      feature_val = imageability / total
    out_file.write(str(feature_val) + "\n")

def main():
  imageability_dict = LoadImageabilityFile(open(args.imageability_filename))
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), imageability_dict)
    

if __name__ == '__main__':
  main()
