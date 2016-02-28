#!/usr/bin/env python3

import argparse
import sys
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log")
args = parser.parse_args()


def GetScore(filename):
  def ParseLine(line):
    assert "Score on dev: " in line or "Score on test: " in line, line
    tokens = line.strip().split(": ")
    return float(tokens[-1])

  def GetMax(fileiter):
    next(fileiter)
    results = []
    next_result = [None, None]
    for line in fileiter:
      if "Score on dev: " in line:
        next_result[0] = ParseLine(line)
      if "Score on test: " in line:
        next_result[1] = ParseLine(line)
        results.append(next_result)
    return max(results)
  
  fileiter = open(filename)
  [dev, test] = GetMax(fileiter)
  return dev, test
    
def main():
  dev, test = GetScore(args.log)
  print("Test_score", test, "Dev_score", dev)
  
if __name__ == '__main__':
  main()
