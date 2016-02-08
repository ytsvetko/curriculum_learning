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

def ParseOutFile(filename):
  timestamp = os.path.getctime(filename)
  lines = open(filename).readlines()
  _, param_str, param_hash = eval(lines[0].strip())
  params = json.loads(param_str)
  return timestamp, params, param_hash, float(lines[-1].strip())
  
def OutIter(dirname):
  for filename in glob.iglob(os.path.join(dirname, "*.out")):
    basename = os.path.basename(filename)
    if "_" in basename or len(basename.split(".")) > 2:
      continue
    yield ParseOutFile(filename)

def main():
  last_best = None
  for filetimestamp, param_dict, param_hash, result_score in sorted(OutIter(args.tpe_dir)):
    if last_best is None:
      last_best = result_score
    if result_score < last_best:
      last_best = result_score
    print("\t".join([str(last_best), str(result_score), param_hash, str(param_dict)]))

if __name__ == "__main__":
  main()
