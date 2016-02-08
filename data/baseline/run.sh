#!/bin/bash


# Construct corpus
cat ../train/*  > wiki.txt

w2v=/usr0/home/ytsvetko/tools/word2vec
#[2016/01/27 15:55] ytsvetko@vivace: /usr1/home/ytsvetko/projects/curric/data/baseline$ 
${w2v} -train wiki.txt -output baseline.wiki.cbow.vectors -size 100 \
  -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -iter 1 -min-count 10 -binary 0
#[2016/01/27 15:59]

#Vocab size: 156663
#Words in train file: 100872713

w2v=/usr1/home/ytsvetko/tools/wang2vec/word2vec
#[2016/01/27 16:00]
${w2v} -train wiki.txt -output baseline.wiki.sskip.vectors -type 3 -size 100 \
  -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -iter 1 -min-count 10 -binary 0
#[2016/01/27 ]

function eval_model {
  vectors=$1
  echo "QVEC"
  /usr0/home/ytsvetko/usr1/projects/qvec/qvec.py --in_vectors ${vectors} --in_oracle /usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en
  
  echo "QVEC_CCA"
  /usr0/home/ytsvetko/usr1/projects/qvec/qvec_cca.py --in_vectors ${vectors} --in_oracle /usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en
  
  currdir=$PWD
  echo "sentiment-analysis"
  cd /usr1/home/ytsvetko/projects/curric/downstream/sentiment-analysis/
  ./eval.sh ${vectors}
  
  echo "newsgroups"
  cd /usr1/home/ytsvetko/projects/curric/downstream/newsgroups/
  ./eval.sh ${vectors}
  
  #cd /usr1/home/ytsvetko/projects/curric/downstream/ner/
  #./eval.sh ${vectors}
  
  #cd /usr1/home/ytsvetko/projects/curric/downstream/internal-lstm-parser
  #./eval.sh ${vectors} 

  cd /usr1/home/ytsvetko/projects/curric/wordsim/
  ./eval.sh ${vectors}
  
  cd ${currdir}
}

eval_model /usr1/home/ytsvetko/projects/curric/data/baseline/baseline.wiki.cbow.vectors
#QVEC score: 15.0147
#QVEC_CCA score: 0.335022
# sentiment-analysis 66.2273476112
# newsgroups 77.810360270225
# parsing 0.787244
# ner

eval_model /usr1/home/ytsvetko/projects/curric/data/baseline/baseline.wiki.sskip.vectors
#QVEC score: 16.6467
#QVEC_CCA score: 0.341761
# sentiment-analysis 66.7764964305
# newsgroups 77.61267225897501
# parsing 0.793166
# ner

eval_model /usr1/home/ytsvetko/projects/curric/data/baseline/baseline.wiki.sg.vectors 
#QVEC score: 15.4034
#QVEC_CCA score: 0.33737
# sentiment-analysis 67.6551345415
# newsgroups 79.58227878150001
# parsing 0.783144
# ner

eval_model /usr1/home/ytsvetko/projects/curric/data/baseline/baseline.wiki.cwindow.vectors 
#QVEC score: 15.3204
#QVEC_CCA score: 0.342581
# sentiment-analysis 67.3805601318
# newsgroups 76.958336083125
# parsing 0.7918
# ner

