#!/bin/bash

for f in ../train-raw/* ; do
 ./tokenize-anything.sh < $f  | ./utf8-normalize.sh  \
    > ../tmp/`basename $f`
  ./normalize.py ../tmp/`basename $f` ../train/`basename $f`  
done
