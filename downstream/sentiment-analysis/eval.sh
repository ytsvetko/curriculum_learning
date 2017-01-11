#!/bin/bash

CURR_DIR=$PWD
ROOT_DIR=/usr1/home/ytsvetko/projects/curric/downstream/sentiment-analysis

mkdir -p tmp
pushd $ROOT_DIR
python ./senti-classify.py ${CURR_DIR}/$1 
popd > /dev/null

