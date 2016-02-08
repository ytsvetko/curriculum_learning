#!/bin/bash

source /usr1/home/ytsvetko/projects/curric/HPOlib/virtualHPOlib/bin/activate

HPOlib-run -o ../../optimizers/tpe/hyperopt_august2013_mod -s 13
