#!/bin/bash

CURR_DIR=$PWD
ROOT_DIR=/usr1/home/ytsvetko/projects/curric/downstream/newsgroups

mkdir -p tmp
pushd $ROOT_DIR
TMPFILE=tmp/`mktemp tmp.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
python ./newsgroup.py ${CURR_DIR}/$1 > $TMPFILE
./parse-ng-results.py --log $TMPFILE
popd > /dev/null

