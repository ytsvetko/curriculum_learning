from hyperopt import hp

space = {'degree': hp.uniform('degree', -1, 1),
         'eigenvector_centrality': hp.uniform('eigenvector_centrality', -1, 1),
         'betweenness_centrality': hp.uniform('betweenness_centrality', -1, 1),
         'closeness_centrality': hp.uniform('closeness_centrality', -1, 1),
         }
