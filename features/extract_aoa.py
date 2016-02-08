#!/usr/bin/env python3

import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--aoa_filename", default="/usr1/home/ytsvetko/projects/curric/data/aoa/AoA.txt")
args = parser.parse_args()

FEATURE_NAME = "aoa"

def LoadAoAFile(aoa_file):
  result = {}
  for line in aoa_file:
    tokens = line.split()
    word = tokens[0]
    aoa = tokens[-1]
    try: 
      aoa = float(aoa)
    except:
      aoa = 25.0
    assert word not in result, line
    result[word] = aoa
  return result
  
def ExtractFeature(in_file, out_file, aoa_dict):
  for line in in_file:
    aoa = 0.0
    words = line.split()
    for word in words:
      if word in aoa_dict:
        aoa += aoa_dict[word]
      else:
        aoa += 25.0
    feature_val = 0.0
    if aoa > 0.0:
      feature_val = aoa / len(words)
    out_file.write(str(feature_val) + "\n")

def main():
  aoa_dict = LoadAoAFile(open(args.aoa_filename))
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), aoa_dict)
    

if __name__ == '__main__':
  main()
