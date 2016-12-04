#!/usr/bin/env python3

import time
import subprocess
import json
import sys
import glob
import collections
import os
import numpy as np

W2V_MIN_COUNT="10"
FEATURE_DIR="/usr1/home/ytsvetko/projects/curric/features/data-standardized/train" 
TRAIN_DATA_DIR="/usr1/home/ytsvetko/projects/curric/data/train"
W2V="/usr0/home/ytsvetko/tools/word2vec" 
WANG2V="/usr1/home/ytsvetko/tools/wang2vec/weightedWord2vec" 
QVEC="/usr0/home/ytsvetko/usr1/projects/qvec/qvec_cca.py" 
QVEC_ORACLE="/usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en"
NG="/usr1/home/ytsvetko/projects/curric/downstream/newsgroups/eval.sh"
NER="/usr1/home/ytsvetko/projects/curric/downstream/ner/eval.sh"
POSTAG="/usr1/home/ytsvetko/projects/curric/downstream/postag/eval.sh"
SENTI="/usr1/home/ytsvetko/projects/curric/downstream/sentiment-analysis/eval.sh"
PARSE="/usr1/home/ytsvetko/projects/curric/downstream/internal-lstm-parser/eval.sh"

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
      

def NormalizeScores(scores):
  scores = np.array(scores)
  scores = scores - scores.mean()
  scores = scores / scores.std()
  scores = 1.0/(1.0+np.exp(-scores)) # Sigmoid
  #scores = np.tanh(scores)
  scores = scores + 0.5
  return scores
   
def SortTrainingData(scored_data, sorted_training_data_path, weighted_w2v):
  out_file = open(sorted_training_data_path, "w")
  lines = []
  scores = []
  for score, line in sorted(scored_data, reverse = True):
    lines.append(line)
    scores.append(score)
  scores = NormalizeScores(scores)
  if weighted_w2v:
    for score, line in zip(scores, lines):
      out_file.write("{}\t{}".format(str(score), line))
  else:
    for line in lines:
      out_file.write(line)
    score_file = open(sorted_training_data_path + ".scores", "w")
    for score in scores:
      score_file.write(str(score) + "\n")

def PrepareVectors(sorted_training_data_path, word_vectors_path, params_hash, weighted_w2v):
  # Run word-to-vec
  if weighted_w2v:
    w2v_command = [WANG2V, "-train", sorted_training_data_path, "-output", word_vectors_path,
                 "-type", "0", "-size", "100", "-window", "5", "-negative", "10", "-hs", "0",
                 "-sample", "1e-4", "-threads", "1", "-iter", "1", "-min-count", W2V_MIN_COUNT,
                 "-binary", "0", ]
  else:
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


def RunEval(word_vectors_path, params_hash, eval_mode):
  # Run Qvec_CCA
  if eval_mode == "QVEC":
    eval_command = [QVEC, "--in_vectors", word_vectors_path, "--in_oracle", QVEC_ORACLE]
  elif eval_mode == "SENTI":
    eval_command = [SENTI, word_vectors_path]
  elif eval_mode == "NG":
    eval_command = [NG, word_vectors_path]
  elif eval_mode == "PARSE":
    eval_command = [PARSE, word_vectors_path]
  elif eval_mode == "NER":
    eval_command = [NER, word_vectors_path]
  elif eval_mode == "POS":
    eval_command = [POSTAG, word_vectors_path]
  stdout_filename = params_hash + ".qvec.out"
  stderr_filename = params_hash + ".qvec.err"
  with open(stdout_filename, "w") as stdout_file:
    with open(stderr_filename, "w") as stderr_file:
      print(" ".join(eval_command))
      exit_code = subprocess.call(eval_command, stdout=stdout_file, stderr=stderr_file)
      if exit_code != 0:
        print("Error running: evaluation")
        raise subprocess.CalledProcessError(exit_code, " ".join(eval_command))
  output = open(stdout_filename).readlines()[-1] 
  return -float(output.split()[-1])


def Eval(params, params_hash, eval_mode, weighted_w2v):
  sorted_training_data_path = os.path.join(params_hash, "sorted_training_data")
  word_vectors_path = os.path.join(params_hash, "word_vectors")
  
  print("Loading data")
  scored_data = LoadScoredData(params)
  print("Sorting data")
  SortTrainingData(scored_data, sorted_training_data_path, weighted_w2v)
  print("word2vec")
  PrepareVectors(sorted_training_data_path, word_vectors_path, params_hash, weighted_w2v)
  print("QVEC")
  eval_result = RunEval(word_vectors_path, params_hash, eval_mode)
  return eval_result


def main(args):
  print(args)
  assert len(args) == 5, (len(args), args)
  str_params = json.loads(args[1])
  params = {k: float(v) for k,v in str_params.items()}
  params_hash = args[2]
  eval_mode = args[3]
  train_mode = args[4]
  weighted_w2v = (train_mode == "W2V_WEIGHTED")
  os.makedirs(params_hash, exist_ok=True)
  result = Eval(params, params_hash, eval_mode, weighted_w2v)
  print(result)


if __name__ == "__main__":
  main(sys.argv)
