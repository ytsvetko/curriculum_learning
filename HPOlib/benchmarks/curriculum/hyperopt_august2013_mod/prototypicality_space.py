from hyperopt import hp

space = {'concreteness': hp.uniform('concreteness', -1, 1),
         'aoa': hp.uniform('aoa', -1, 1),
         'conventionalization': hp.uniform('conventionalization', -1, 1),
         'imageability': hp.uniform('imageability', -1, 1),
         'lm_score': hp.uniform('lm_score', -1, 1),
         'supersense_relative_freq': hp.uniform('supersense_relative_freq', -1, 1),
         'synset_relative_freq': hp.uniform('synset_relative_freq', -1, 1),
         'num_syllables': hp.uniform('num_syllables', -1, 1),
         'word_length': hp.uniform('word_length', -1, 1)
         }
