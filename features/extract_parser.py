#!/usr/bin/env python3

import os
import argparse
import glob
import re

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--parser_train_coprus_file", default="../data/parser/train-zpar/train.txt")
parser.add_argument("--parser_output_file", default="../data/parser/train-parsed-zpar/train.parsed")
args = parser.parse_args()

TREE_DEPTH_FEATURE_NAME = "tree_depth"
VERB_TOKEN_RATIO_FEATURE_NAME = 'verb_token_ratio'
NOUN_TOKEN_RATIO_FEATURE_NAME = 'noun_token_ratio'
NUM_PP_FEATURE_NAME = 'num_pp'
NUM_NP_FEATURE_NAME = 'num_np'
NUM_VP_FEATURE_NAME = 'num_vp'

def CalcTreeDepth(pos_paragraph, line_num):
  max_depth = 0
  current_depth = 0
  for c in pos_paragraph:
    if c == ")":
      current_depth -= 1
    if c == "(":
      current_depth += 1
      max_depth = max(max_depth, current_depth)
  #assert current_depth == 0, (current_depth, max_depth, line_num+1, pos_paragraph)
  return max_depth

def main():
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  parser_corpus_file = open(args.parser_train_coprus_file)
  parser_output_file = open(args.parser_output_file)
  
  max_num_pp = -1
  max_num_np = -1
  max_num_vp = -1
  max_depth = -1
  max_verb_token_ratio = -1
  max_noun_token_ratio = -1

  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):    
    basename = os.path.basename(in_file_name)
    print("Processing file:", basename)
    out_file_name_prefix = os.path.join(args.out_feature_data_dir, basename + ".")

    num_pp_file = open(out_file_name_prefix + NUM_PP_FEATURE_NAME, "w")
    num_vp_file = open(out_file_name_prefix + NUM_VP_FEATURE_NAME, "w")
    num_np_file = open(out_file_name_prefix + NUM_NP_FEATURE_NAME, "w")
    depth_file = open(out_file_name_prefix + TREE_DEPTH_FEATURE_NAME, "w")
    verb_ratio_file = open(out_file_name_prefix + VERB_TOKEN_RATIO_FEATURE_NAME, "w")
    noun_ratio_file = open(out_file_name_prefix + NOUN_TOKEN_RATIO_FEATURE_NAME, "w")

    for line_num, orig_file_line in enumerate(open(in_file_name)):
      pos_paragraph = next(parser_output_file)

      # Validate glob result ordering hasn't changed.  
      parser_corpus_line = next(parser_corpus_file)
      line_too_long = parser_corpus_line == "-----------------------------\n"
      assert line_too_long or orig_file_line == parser_corpus_line, line_num
  
      if line_too_long:
        num_pp = 90
        num_np = 210
        num_vp = 115
        depth = 130
        verb_ratio = 1.0
        noun_ratio = 3.0
      else:
        num_pp = pos_paragraph.count("(PP ")
        num_np = pos_paragraph.count("(NP ")
        num_vp = pos_paragraph.count("(VP ")
        depth = CalcTreeDepth(pos_paragraph, line_num)
        verb_count = pos_paragraph.count("(VB")
        noun_count = pos_paragraph.count("(NN")
        token_count = len(orig_file_line.split())
        verb_ratio = verb_count * 1.0 / token_count if token_count else 0.0
        noun_ratio = noun_count * 1.0 / token_count if token_count else 0.0
      
      max_num_pp = max(max_num_pp, num_pp)
      max_num_np = max(max_num_np, num_np)
      max_num_vp = max(max_num_vp, num_vp)
      max_depth = max(max_depth, depth)
      max_verb_token_ratio = max(max_verb_token_ratio, verb_ratio)
      max_noun_token_ratio = max(max_noun_token_ratio, noun_ratio)

      num_pp_file.write(str(num_pp) + "\n")
      num_vp_file.write(str(num_vp) + "\n")
      num_np_file.write(str(num_np) + "\n")
      depth_file.write(str(depth) + "\n")
      verb_ratio_file.write(str(verb_ratio) + "\n")
      noun_ratio_file.write(str(verb_ratio) + "\n")

  print("max_num_pp:", max_num_pp)
  print("max_num_np:", max_num_np)
  print("max_num_vp:", max_num_vp)
  print("max_depth:", max_depth)
  print("max_verb_token_ratio:", max_verb_token_ratio)
  print("max_noun_token_ratio:", max_noun_token_ratio)

if __name__ == '__main__':
  main()
