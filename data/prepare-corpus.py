#!/usr/bin/env python3

import shutil
import os
import random 

RAW_CORPUS = "/usr1/home/ytsvetko/projects/curric/data/en-wiki"
TGT_CORPUS = "/usr1/home/ytsvetko/projects/curric/data/train"

random.seed(12345)

def FlattenAndClean(src_corpus, tgt_corpus):
  articles = set(random.sample(range(20440), 1001))
  seen = 0
  for root, dirs, files in os.walk(src_corpus):
    path = root.split('/')
    for f in files:
      seen += 1
      if seen in articles:
        shutil.copy(os.path.join(root, f), os.path.join(tgt_corpus, os.path.basename(root)+"_"+f))
        
def main():
  FlattenAndClean(RAW_CORPUS, TGT_CORPUS)


if __name__ == "__main__":
  main()
