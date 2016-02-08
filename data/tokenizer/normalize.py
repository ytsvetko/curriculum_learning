#!/usr/bin/env python3

import gzip
import os
import sys

PUNCTUATION = set("“”'!$%(),-.:;?/")

def main(args):
  src, tgt = args[1], args[2] 
  f = open(tgt, "w")
  for line in open(src):
    tokens = line.split()
    new_line = []
    for token in tokens:
      if token.isalpha() or token.replace(".","").replace(",","").isalpha():
        new_line.append(token)
      elif token in PUNCTUATION or token == "--":
        new_line.append(token)
      elif token.isdigit() or token.replace(".","").replace(",","").isdigit():
        new_token = []
        for ch in token:
          if ch.isdigit():
            new_token.append("DG")
          else:
            new_token.append(ch)
        new_line.append("".join(new_token))
      else:
        new_line.append("UNK")
    if " ".join(new_line).strip():
      f.write(" ".join(new_line))
      f.write("\n")
    
if __name__ == '__main__':
    main(sys.argv)
