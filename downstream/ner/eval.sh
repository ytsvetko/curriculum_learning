#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python  tagger.py --data_type ner --char_emb_dim 25 --char_lstm_dim 25 --char_bidirect 1 --token_emb_dim 100 --token_lstm_dim 100 --token_bidirect 1 --lr_method sgd-lr_.01 --crf 1 --dropout 1 --caps_dim 0 --lower_t 0 --pos_dim 0 --zeros 1 --ext_emb $1

