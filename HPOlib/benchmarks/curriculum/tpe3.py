#!/usr/bin/env python3

import time
import subprocess
import json
import sys
import glob
import collections
import os

W2V_MIN_COUNT="10"
FEATURE_DIR="/usr1/home/ytsvetko/projects/curric/features/data/train" #data-standardized
TRAIN_DATA_DIR="/usr1/home/ytsvetko/projects/curric/data/train"
W2V="/usr0/home/ytsvetko/tools/word2vec" #"/usr1/home/ytsvetko/tools/wang2vec/word2vec"
QVEC="/usr0/home/ytsvetko/usr1/projects/qvec/qvec_cca.py" # TODO other optimization functions
QVEC_ORACLE="/usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en"

def LoadScoredData(params):
  # key = (train_filename, line_num), value = {feature: feature_value * feature_weight}
  feature_dict = collections.defaultdict(dict)
  for train_filename in glob.glob(os.path.join(TRAIN_DATA_DIR, "*")):
    train_base_filename = os.path.basename(train_filename)
    feature_filebase = os.path.join(FEATURE_DIR, train_base_filename)
    for feature, weight in params.items():
      for line_num, line in enumerate(open(feature_filebase + "." + feature)):
        feature_dict[(train_base_filename, line_num)][feature] = float(line.strip()) * weight

  # Sum up the feature values
  score_dict = {}
  for key, weighted_features_dict in feature_dict.items():
    score_dict[key] = sum(weighted_features_dict.values())

  result = []  # (score, line)
  for train_filename in glob.glob(os.path.join(TRAIN_DATA_DIR, "*")):
    train_base_filename = os.path.basename(train_filename)
    for line_num, line in enumerate(open(train_filename)):
      result.append((score_dict[(train_base_filename, line_num)], line))
  return result
      
  
def SortTrainingData(scored_data, sorted_training_data_path):
  out_file = open(sorted_training_data_path, "w")
  for score, line in sorted(scored_data, reverse = True):
    out_file.write(line)

def PrepareVectors(sorted_training_data_path, word_vectors_path, params_hash):
  # Run word-to-vec
  w2v_command = [W2V, "-train", sorted_training_data_path, "-output", word_vectors_path,
                 "-cbow", "1", "-size", "100", "-window", "5", "-negative", "10", "-hs", "0",
                 "-sample", "1e-4", "-threads", "1", "-iter", "1", "-min-count", W2V_MIN_COUNT,
                 "-binary", "0", ] #"-cbow", "0", "-type", "3"
  stdout_filename = params_hash + ".w2v.out"
  stderr_filename = params_hash + ".w2v.err"
  with open(stdout_filename, "w") as stdout_file:
    with open(stderr_filename, "w") as stderr_file:
      print(" ".join(w2v_command))
      exit_code = subprocess.call(w2v_command, stdout=stdout_file, stderr=stderr_file)
      if exit_code != 0:
        print("Error running: w2v")
        raise subprocess.CalledProcessError(exit_code, " ".join(w2v_command))


def RunEval(word_vectors_path, params_hash):
  # Run QVec
  qvec_command = [QVEC, "--in_vectors", word_vectors_path, "--in_oracle", QVEC_ORACLE]
  stdout_filename = params_hash + ".qvec.out"
  stderr_filename = params_hash + ".qvec.err"
  with open(stdout_filename, "w") as stdout_file:
    with open(stderr_filename, "w") as stderr_file:
      print(" ".join(qvec_command))
      exit_code = subprocess.call(qvec_command, stdout=stdout_file, stderr=stderr_file)
      if exit_code != 0:
        print("Error running: qvec")
        raise subprocess.CalledProcessError(exit_code, " ".join(qvec_command))
  output = open(stdout_filename).readlines()[-1] 
  return -float(output.split()[-1])


def Eval(params, params_hash):
  sorted_training_data_path = os.path.join(".", params_hash, "sorted_training_data")
  word_vectors_path = os.path.join(".", params_hash, "word_vectors")
  
  print("Loading data")
  scored_data = LoadScoredData(params)
  print("Sorting data")
  SortTrainingData(scored_data, sorted_training_data_path)
  print("word2vec")
  PrepareVectors(sorted_training_data_path, word_vectors_path, params_hash)
  print("QVEC")
  eval_result = RunEval(word_vectors_path, params_hash)
  return eval_result


def main(args):
  print(args)
  assert len(args) == 3, (len(args), args)
  str_params = json.loads(args[1])
  params = {k: float(v) for k,v in str_params.items()}
  params_hash = args[2]
  os.makedirs(params_hash, exist_ok=True)
  result = Eval(params, params_hash)
  print(result)


if __name__ == "__main__":
  main(sys.argv)
