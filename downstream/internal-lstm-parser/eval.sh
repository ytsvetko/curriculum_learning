#!/bin/bash

CURR_DIR=$PWD
ROOT_DIR=/usr1/home/ytsvetko/projects/curric/downstream/internal-lstm-parser

pushd $ROOT_DIR
mkdir -p tmp
TMPFILE=tmp/`mktemp tmp.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
python eval_parse.py -eval-data /usr1/home/ytsvetko/projects/curric/data/ud1.1-en-dev -embeddings-file ${CURR_DIR}/${1} -parser-binary ./build/parser/lstm-parse > $TMPFILE.dev
python eval_parse.py -eval-data /usr1/home/ytsvetko/projects/curric/data/ud1.1-en-test -embeddings-file ${CURR_DIR}/${1} -parser-binary ./build/parser/lstm-parse > $TMPFILE.test
./parse-results.py --log_dev $TMPFILE.dev --log_test $TMPFILE.test
popd > /dev/null




#


