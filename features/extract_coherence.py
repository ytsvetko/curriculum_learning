#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--distance_matrix", default="/usr1/home/ytsvetko/projects/curric/data/coherence-graph/distances.npy")
parser.add_argument("--distance_threshold", type=float, default=0.8)
args = parser.parse_args()


DEGREE_CENTRALITY_FEATURE_NAME = "degree_centrality"
CLOSENESS_CENTRALITY_FEATURE_NAME = "closeness_centrality"
EIGEN_CENTRALITY_FEATURE_NAME = "eigen_centrality"


def LoadCorpusIndex(corpus_dir):
  result = {}
  index = 0
  for in_file_name in sorted(glob.iglob(os.path.join(corpus_dir, "*"))):
    basename = os.path.basename(in_file_name)
    print("Loading:", in_file_name)
    for line_num in enumerate(open(in_file_name)):
      result[index] = (basename, line_num)
      index += 1
  return result


def main():
  print("Loading matrix")
  density_matrix = np.load(args.distance_matrix)
  print("Loading corpus index")
  corpus_index = LoadCorpusIndex(args.in_training_data_dir)
  print("Applying threshold")

  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    out_file_name = os.path.join(args.out_feature_data_dir, os.path.basename(in_file_name) + "." + FEATURE_NAME)
    ExtractFeature(open(in_file_name), open(out_file_name, "w"), aoa_dict)
    

if __name__ == '__main__':
  main()
