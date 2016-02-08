#!/usr/bin/env python

import gzip
import argparse
import random
import io
import os
import sys
from read_write import read_word_vectors
from read_write import gzopen
import subprocess

def get_test_treebank_filename():
  return 'test_treebank'

def get_train_treebank_filename():
  return 'train_treebank'

def get_test_arcstd_filename():
  return 'test_treebank.arcstd'

def get_train_arcstd_filename():
  return 'train_treebank.arcstd'

def get_relevant_word_types(test_treebank_filename, train_treebank_filename):
  relevant_word_types = set()
  for treebank_filename in [train_treebank_filename, test_treebank_filename]:
    with io.open(treebank_filename, encoding='utf8') as treebank_file:
      for line in treebank_file:
        line = line.strip()
        if len(line) == 0: continue
        splits = line.split('\t')
        assert len(splits) > 2
        word = splits[1]
        relevant_word_types.add(word)
  return relevant_word_types

def get_relevant_embeddings_filename(test_treebank_filename, train_treebank_filename, embeddings_filename):
  # We only need embeddings for a subset of word types. Copy the relevant embeddings in a new plain file.
  relevant_embeddings_filename = os.path.join(os.path.dirname(__file__), 'temp', str(random.randint(100000, 999999)))
  relevant_word_types = set(get_relevant_word_types(test_treebank_filename, train_treebank_filename))
  with gzopen(embeddings_filename) as all_embeddings_file:
    with open(relevant_embeddings_filename, mode='w') as relevant_embeddings_file:
      for line in all_embeddings_file:
        line = line.decode('utf8')
        if line.split(' ')[0] not in relevant_word_types: continue
        line = line.encode('utf8')
        relevant_embeddings_file.write(line)
  return relevant_embeddings_filename

def compute_coverage(test_treebank_filename, word_vecs):
  not_found, total_size = (0, 0)
  with io.open(test_treebank_filename, encoding='utf8') as test_treebank_file:
    for line in test_treebank_file:
      splits = line.strip().lower().split('\t')
      if len(splits) < 2: continue
      word = splits[1]
      total_size += 1
      if word not in word_vecs: not_found += 1

  assert total_size > 0
  return 1.0 - (not_found * 1.0 / total_size)

def parsing_wrapper(train_arcstd_filename, test_arcstd_filename, embeddings_filename, embeddings_dimensionality, parser_binary):
  score_filename = os.path.join(os.path.dirname(__file__), 'temp', str(random.randint(100000, 999999)))
  training_epochs = 3;

  print 'embeddings_filename used for the parser: ', embeddings_filename
  print 'embeddings_dimensionality: ', embeddings_dimensionality
  print 'train_arcstd_filename: ', train_arcstd_filename
  print 'test_arcstd_filename: ', test_arcstd_filename

  FNULL = open(os.devnull, 'w')
  command = "{} --train --training_data {} --dev_data {} --input_dim 0 ".format(parser_binary, 
                                                                                train_arcstd_filename, 
                                                                                test_arcstd_filename)
  command += "--pretrained_dim {} --pretrained {} --score_file {} --epochs {}".format(embeddings_dimensionality,
                                                                                      embeddings_filename, 
                                                                                      score_filename,
                                                                                      training_epochs)
  subprocess.call([command], shell=True, 
                  #stdout=FNULL, 
                  stderr=subprocess.STDOUT)
  score_string = ''
  with open(score_filename) as score_file:
    score_string = score_file.read()
  os.remove(score_filename)
  score = float(score_string)
  return score

def evaluate(eval_data_dir, embeddings_filename, parser_binary_filename):
  test_treebank_filename = '{}/{}'.format(eval_data_dir, get_test_treebank_filename())
  train_treebank_filename = '{}/{}'.format(eval_data_dir, get_train_treebank_filename())
  test_arcstd_filename = '{}/{}'.format(eval_data_dir, get_test_arcstd_filename())
  train_arcstd_filename = '{}/{}'.format(eval_data_dir, get_train_arcstd_filename())
  relevant_embeddings_filename = get_relevant_embeddings_filename(test_treebank_filename, train_treebank_filename, embeddings_filename)
  word_vecs = read_word_vectors(relevant_embeddings_filename)
  assert len(word_vecs) > 0; embeddings_dimensionality = len(word_vecs.itervalues().next())
  coverage = compute_coverage(test_treebank_filename, word_vecs)
  if coverage == 0: 
    print 'coverage = 0!!'
    return (0.0, 0.0)
  score = parsing_wrapper(train_arcstd_filename, test_arcstd_filename, relevant_embeddings_filename, embeddings_dimensionality, parser_binary_filename)
  os.remove(relevant_embeddings_filename)
  return (score, coverage,)

def main(argv):
  # parse/validate arguments
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-eval-data", help="Path to a directory which contains all data files needed to setup the evaluation script.")
  argparser.add_argument("-embeddings-file", help="Path to the embeddings file (lowercased, UTF8-encoded, space-delimited, optional: suffix .gz indicate the file is gzip compressed.)")
  argparser.add_argument("-parser-binary", help="Path to the parser binary")
  args = argparser.parse_args()
  # evaluate
  score, coverage = evaluate(args.eval_data, args.embeddings_file, args.parser_binary)
  # report
  print 'score={}, coverage={}'.format(score, coverage)

if __name__ == '__main__':
  main(sys.argv)
