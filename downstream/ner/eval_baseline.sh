#!/bin/bash

for f in tmp/*baseline-shuffled.wiki?.cbow.vectors/experiment.log
do
  ./parse-ner-results.py --log $f 
done
