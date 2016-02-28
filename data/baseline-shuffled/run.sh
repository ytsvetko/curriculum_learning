#!/bin/bash

#cat ../train/* | shuf > wiki${i}.txt 

comment='
for i in 1 2 3 4 5 6 7 8 9 0 ; do 
  w2v=/usr0/home/ytsvetko/tools/word2vec
  ${w2v} -train wiki${i}.txt -output baseline-shuffled.wiki${i}.cbow.vectors \
   -size 100 -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -iter 1 -min-count 10 -binary 0 -nce 0
   
  ${w2v} -train wiki${i}.txt -output baseline-shuffled.wiki${i}.sg.vectors \
   -size 100 -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -iter 1 -min-count 10 -cbow 0 -binary 0 -nce 0

  w2v=/usr1/home/ytsvetko/tools/wang2vec/word2vec
  ${w2v} -train wiki${i}.txt -output baseline-shuffled.wiki${i}.cwindow.vectors \
   -size 100 -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -iter 1 -min-count 10 -type 2 -binary 0 -nce 0
   
  ${w2v} -train wiki${i}.txt -output baseline-shuffled.wiki${i}.sskip.vectors \
   -size 100 -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -iter 1 -min-count 10 -type 3 -binary 0 -nce 0
done'


function eval_model {
  vectors=$1
  currdir=$PWD
  log_file=$2
  
  comment='
  /usr0/home/ytsvetko/usr1/projects/qvec/qvec.py --in_vectors ${vectors} --in_oracle /usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en >> ${log_file}.qvec
  
  /usr0/home/ytsvetko/usr1/projects/qvec/qvec_cca.py --in_vectors ${vectors} --in_oracle /usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en >> ${log_file}.qvec_cca
  
  /usr1/home/ytsvetko/projects/curric/downstream/sentiment-analysis/eval.sh ${vectors} >> ${log_file}.senti
  
  /usr1/home/ytsvetko/projects/curric/downstream/newsgroups/eval.sh ${vectors} >> ${log_file}.ng
  
  /usr1/home/ytsvetko/projects/curric/wordsim/eval.sh ${vectors} >> ${log_file}.wordsim'
 
  /usr1/home/ytsvetko/projects/curric/downstream/internal-lstm-parser/eval.sh ${vectors} >> ${log_file}.parse

  comment='
  /usr1/home/ytsvetko/projects/curric/downstream/ner/eval.sh ${vectors} >> ${log_file}.ner &
  
  /usr1/home/ytsvetko/projects/curric/downstream/postag//eval.sh ${vectors} >> ${log_file}.pos '

  #cd ${currdir}
  
}

for i in 1 2 3 4 5 6 7 8 9 0 ; do 
  echo baseline-shuffled.wiki${i}.cbow.vectors
  eval_model baseline-shuffled.wiki${i}.cbow.vectors  $PWD/log/cbow/cbow${i}.log 
 comment=' 
  echo baseline-shuffled.wiki${i}.sg.vectors 
  eval_model /usr1/home/ytsvetko/projects/curric/data/baseline-shuffled/baseline-shuffled.wiki${i}.sg.vectors  $PWD/log/sg${i}.log

  echo baseline-shuffled.wiki${i}.cwindow.vectors
  eval_model /usr1/home/ytsvetko/projects/curric/data/baseline-shuffled/baseline-shuffled.wiki${i}.cwindow.vectors  $PWD/log/cwindow${i}.log

  echo baseline-shuffled.wiki${i}.sskip.vectors 
  eval_model /usr1/home/ytsvetko/projects/curric/data/baseline-shuffled/baseline-shuffled.wiki${i}.sskip.vectors  $PWD/log/sskip${i}.log'
done


