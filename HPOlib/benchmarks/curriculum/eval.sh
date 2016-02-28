#!/bin/bash

function eval_model {
  vectors=$1
  currdir=$PWD
  log_file=$2
  
  /usr0/home/ytsvetko/usr1/projects/qvec/qvec.py --in_vectors ${vectors} --in_oracle /usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en >> ${log_file}.qvec
  
  /usr0/home/ytsvetko/usr1/projects/qvec/qvec_cca.py --in_vectors ${vectors} --in_oracle /usr0/home/ytsvetko/usr1/projects/qvec/oracles/semcor_noun_verb.supersenses.en >> ${log_file}.qvec_cca
  
  cd /usr1/home/ytsvetko/projects/curric/downstream/sentiment-analysis/
  ./eval.sh ${vectors} >> ${log_file}.senti
  
  cd /usr1/home/ytsvetko/projects/curric/downstream/newsgroups/
  ./eval.sh ${vectors} >> ${log_file}.ng
  
  cd /usr1/home/ytsvetko/projects/curric/wordsim/
  ./eval.sh ${vectors} >> ${log_file}.wordsim
 
  cd /usr1/home/ytsvetko/projects/curric/downstream/internal-lstm-parser
  ./eval.sh ${vectors} >> ${log_file}.parse 

  #cd /usr1/home/ytsvetko/projects/curric/downstream/ner/
  #./eval.sh ${vectors} >> ${log_file}.ner 
  
  cd ${currdir}
}



cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-50-diversity-standardized/57fabcc79cc45b2714a2adf541b96da2/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-50-diversity-standardized.eval

exit

cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-8--qvec_cca-cbow-50-prototypicality-standardized/bd33d926333d1d70ee054f3d008f31b7/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-8--qvec_cca-cbow-50-prototypicality-standardized.eval

cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-8--qvec_cca-cbow-50-prototypicality/ac338a2fd79e76c0a3c12b676f31f5df/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-8--qvec_cca-cbow-50-prototypicality.eval




cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-5--qvec_cca-cbow-50-diversity-new/17db81720d0a0420f6f7177f75510c1c/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-5--qvec_cca-cbow-50-diversity-new.eval

## 100 iterations
cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-100-diversity-standardized/00025bb43d2144b3fac051fa38cfe312/word_vectors   
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-100-diversity-standardized.eval

cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-100-diversity-standardized-3/4a873fb94f969a5398ea9850121710b8/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-100-diversity-standardized-3.eval


### hyperparams[-10, 10]
cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-50-diversity-standardized-10/e40e802a13d78e583a72309fee1c9167/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-50-diversity-standardized-10.eval



### standarsized ###
cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-50-diversity-standardized/57fabcc79cc45b2714a2adf541b96da2/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-2-4--qvec_cca-cbow-50-diversity-standardized.eval
### qvec ###
cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-29--qvec-cbow-50-diversity/facc9d27982d43e5d6784604a9fb27da/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-1-29--qvec-cbow-50-diversity.eval
#### 50 ####
cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-cbow-50-diversity/0d0200932619bb2711379bcbb34c40fd/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-cbow-50-diversity.eval

sg=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-sg-50-diversity/13d4a0db6513947acc3984e9192be497/word_vectors   
eval_model ${sg} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-sg-50-diversity.eval

sskip=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-sskip-50-diversity/15f8b48c612afdbfcafe845588509659/word_vectors
eval_model ${sskip} /usr1/home/ytsvetko/projects/curric/curriculum/log/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-sskip-50-diversity

#### 30 ####
cbow=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-cbow-50-diversity/e9a95e7791b0e653ff5cb8872a1c4c4e/word_vectors
eval_model ${cbow} /usr1/home/ytsvetko/projects/curric/curriculum/log/cbow-30

sg=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-sg-50-diversity/768db7fc2dc5c0ec47c1d8c690b5ab4d/word_vectors
eval_model ${sg} /usr1/home/ytsvetko/projects/curric/curriculum/log/sg-30

sskip=/usr1/home/ytsvetko/projects/curric/curriculum/hyperopt_august2013_mod_13_2016-1-30--qvec_cca-sskip-50-diversity/27d03d3a3b6dff44055b078bd2faa717/word_vectors
eval_model ${sskip} /usr1/home/ytsvetko/projects/curric/curriculum/log/sskip-30

