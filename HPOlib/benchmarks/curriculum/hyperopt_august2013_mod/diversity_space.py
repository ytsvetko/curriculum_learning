from hyperopt import hp

space = {'type_token_ratio': hp.uniform('type_token_ratio', -1, 1),
         'types': hp.uniform('types', -1, 1),
         'balance_shannon': hp.uniform('balance_shannon', -1, 1),
         'balance_simpson': hp.uniform('balance_simpson', -1, 1),
         'disparity': hp.uniform('disparity', -1, 1),
         }
