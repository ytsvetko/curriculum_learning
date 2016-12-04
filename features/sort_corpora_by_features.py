#!/usr/bin/env python3

import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--in_feature_data_dir", default="./data-standardized/train")
parser.add_argument("--out_sorted_corpora_dir", default="./sorted-corpora/train")
args = parser.parse_args()

def GetFeatureNames(feature_dir):
  feature_names = set()
  for filename in glob.glob(os.path.join(feature_dir, "*")):
    feature = os.path.basename(filename).split(".", 1)[1]
    feature_names.add(feature)
  return feature_names

def LoadCorpus(corpus_dir):
  corpus = []
  wiki_filenames = []
  for filename in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
    wiki_filenames.append(os.path.basename(filename))
    corpus.extend(open(filename).readlines())
  return corpus, wiki_filenames

def LoadFeature(feature_dir, wiki_filenames, feature_name):
  feature_map = []
  index = 0
  for filename in wiki_filenames:
    for line in open(os.path.join(feature_dir, filename + "." + feature_name)):
      feature_map.append((float(line.strip()), index))
      index += 1
  return sorted(feature_map, reverse=True)

def main():
  os.makedirs(args.out_sorted_corpora_dir, exist_ok=True)
  print("Loading corpus")
  corpus, wiki_filenames = LoadCorpus(args.in_training_data_dir)
  print("Loading list of features")
  for feature_name in sorted(GetFeatureNames(args.in_feature_data_dir)):
    print("Loading feature:", feature_name)
    feature_map = LoadFeature(args.in_feature_data_dir, wiki_filenames, feature_name)
    assert len(feature_map) == len(corpus), (len(feature_map), len(corpus))
    print("Storing corpus ordered by feature:", feature_name)
    out_file = open(os.path.join(args.out_sorted_corpora_dir, "wiki." + feature_name), "w")
    for feature_val, corpus_line_index in feature_map:
      out_file.write(corpus[corpus_line_index])

if __name__ == '__main__':
  main()
