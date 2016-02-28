#!/usr/bin/env python3

import argparse
import sys
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_dev")
parser.add_argument("--log_test")
args = parser.parse_args()


def GetScore(filename):
  line = open(filename).readlines()[-1] 
  # score=0.765138, coverage=0.885295385942
  score = float((line.split(",")[0]).split("=")[-1])
  return score
    
def main():
  dev = GetScore(args.log_dev)
  test = GetScore(args.log_test)
  print("Test_score", test, "Dev_score", dev)
  
if __name__ == '__main__':
  main()
