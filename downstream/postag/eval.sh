#!/bin/bash

CURR_DIR=$PWD
ROOT_DIR=/usr1/home/ytsvetko/projects/curric/downstream/postag

pushd $ROOT_DIR
/usr0/home/glample/anaconda/bin/python  ${ROOT_DIR}/tagger.py --data_type pos --char_emb_dim 25 --char_lstm_dim 25 --char_bidirect 1 --token_emb_dim 100 --token_lstm_dim 100 --token_bidirect 1 --lr_method sgd-lr_.01 --crf 1 --dropout 1 --caps_dim 0 --lower_t 0 --pos_dim 0 --zeros 1 --ext_emb ${CURR_DIR}/$1
d=`dirname ${CURR_DIR}/$1`
${ROOT_DIR}/parse-pos-results.py --log tmp/pos-`basename $d`-`basename ${CURR_DIR}/$1`/experiment.log

popd > /dev/null

