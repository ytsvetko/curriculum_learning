from hyperopt import hp

space = { 'char_lm_score': hp.uniform('char_lm_score', -1, 1),
          'length': hp.uniform('length', -1, 1),
          'concreteness': hp.uniform('concreteness', -1, 1),
          'verb_token_ratio': hp.uniform('verb_token_ratio', -1, 1),
          'noun_token_ratio': hp.uniform('noun_token_ratio', -1, 1),          
          'tree_depth': hp.uniform('tree_depth', -1, 1),          
          'num_pp': hp.uniform('num_pp', -1, 1),          
          'num_np': hp.uniform('num_np', -1, 1),                    
          'num_vp': hp.uniform('num_vp', -1, 1),                    
        }
