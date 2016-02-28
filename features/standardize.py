#!/usr/bin/env python3

import os
import argparse
import glob
import math 

parser = argparse.ArgumentParser()
parser.add_argument("--in_feature_data_dir", default="./data/train")
parser.add_argument("--out_feature_data_dir", default="./data-standardized/train")
parser.add_argument("--in_feature_name", default="degree")
args = parser.parse_args()

def ExtractSum(in_file):
  lines = in_file.readlines()
  file_elems = [float(n) for n in lines]
  file_sum = sum(file_elems)
  file_len = len(lines)
  return file_sum, file_elems, file_len
  
def Standardize(in_file, out_file, mean, std):
  for n in in_file:
    standardized = (float(n)-mean)/std
    out_file.write("{}\n".format(standardized))
    
def main():
  total_sum = 0
  total_len = 0
  total_elems = []
  for in_file_name in glob.glob(os.path.join(args.in_feature_data_dir, "*." + args.in_feature_name)): 
    file_sum, file_elems, file_len = ExtractSum(open(in_file_name))
    total_sum += file_sum
    total_len += file_len
    total_elems.extend(file_elems)
  
  mean = total_sum / total_len
  std = math.sqrt(sum([(n - mean)**2 for n in total_elems])/total_len)
      
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_feature_data_dir, "*." + args.in_feature_name)):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name))
    Standardize(open(in_file_name), open(out_file_name, "w"), mean, std)
    

if __name__ == '__main__':
  main()
