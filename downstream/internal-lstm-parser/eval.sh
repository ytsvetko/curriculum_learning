#!/bin/bash

#python eval_parse.py -eval-data ../../data/ud1.1-en-dev -embeddings-file ${1} -parser-binary ./build/parser/lstm-parse 

python eval_parse.py -eval-data ../../data/ud1.1-en-test -embeddings-file ${1} -parser-binary ./build/parser/lstm-parse 

