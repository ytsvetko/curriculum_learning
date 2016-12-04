#!/usr/bin/env python3

"""
./parse_output.py --tpe_dir hyperopt_august2013_mod_13_2016-1-29--qvec-cbow-50

"""

import argparse
import json
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--tpe_dir")
args = parser.parse_args()

def ParseOutFile(filename, qvec_filename):
  timestamp = os.path.getctime(filename)
  lines = open(filename).readlines()
  _, param_str, param_hash = eval(lines[0].strip())[:3]
  params = json.loads(param_str)
  qvec_lines = open(qvec_filename).readlines()
  qvec_scores = qvec_lines[-1].split()
  test_score = 0.0
  dev_score = 0.0
  if len(qvec_scores) == 4: 
    _, test_score, _, dev_score = qvec_lines[-1].split()
  return timestamp, params, param_hash, float(dev_score), float(test_score) 
  
def OutIter(dirname):
  for filename in glob.iglob(os.path.join(dirname, "*.out")):
    basename = os.path.basename(filename)
    if "_" in basename or len(basename.split(".")) > 2:
      continue
    qvec_filename = filename.replace(".out", ".qvec.out")
    yield ParseOutFile(filename, qvec_filename)

def main():
  last_best_dev = None
  last_best_test = None
  for filetimestamp, param_dict, param_hash, dev_score, test_score in sorted(OutIter(args.tpe_dir)):
    if last_best_dev is None:
      last_best_dev = dev_score
      last_best_test = test_score
    if dev_score > last_best_dev:
      last_best_dev = dev_score
      last_best_test = test_score
    print("\t".join([str(last_best_dev), str(last_best_test), str(dev_score), str(test_score), param_hash, str(param_dict)]))

if __name__ == "__main__":
  main()
