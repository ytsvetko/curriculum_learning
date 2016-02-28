#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np
import graph_tool
import graph_tool.centrality
import json
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--in_training_data_dir", default="../data/train")
parser.add_argument("--out_feature_data_dir", default="./data/train")
parser.add_argument("--distance_matrix_prefix", default="../data/coherence-graph/data/distances_unique.")
parser.add_argument("--corpus_index_file", default="../data/coherence-graph/data/index.json")
parser.add_argument("--max_distance_threshold", type=float, default=0.001)
args = parser.parse_args()

CLOSENESS_FEATURE_NAME = "closeness_centrality"
EIGENVECTOR_FEATURE_NAME = "eigenvector_centrality"
DEGREE_FEATURE_NAME = "degree"
BETWEENNESS_FEATURE_NAME = "betweenness_centrality"

def LoadCorpusIndex(filename):
  corpus_index_as_list = json.load(open(args.corpus_index_file))
  wiki_to_matrix = {}
  matrix_to_wiki = collections.defaultdict(set)
  for wiki_filename, line_num, matrix_file_num, matrix_row in corpus_index_as_list:
    assert (wiki_filename, line_num) not in wiki_to_matrix, (wiki_filename, line_num)
    wiki_to_matrix[(wiki_filename, line_num)] = (matrix_file_num, matrix_row)
    matrix_to_wiki[(matrix_file_num, matrix_row)].add((wiki_filename, line_num))
  return wiki_to_matrix, matrix_to_wiki

def SaveFeature(feature_values, feature_name, matrix_num,
                matrix_to_wiki, calculated_features):
  for col_num, value in enumerate(feature_values):
    for wiki_filename, line_num in matrix_to_wiki[(matrix_num, col_num)]:
      calculated_features[(wiki_filename, line_num)] = value

def CalcDegree(vertex, weights):
  result = 0.0
  for e in vertex.out_edges():
    result += weights[e]
  return result
  
def ProcessDistanceMatrix(matrix, matrix_num, matrix_to_wiki, max_distance_threshold, calculated_features):
  if matrix.shape[0] == 0:
    return
  print("  building graph for matrix size:", matrix.shape[0])
  g = graph_tool.Graph(directed=False)
  g.ep.weights = g.new_edge_property("float")
  vlist = list(g.add_vertex(matrix.shape[0]))
  num_edges = 0
  for row in range(matrix.shape[0]):
    if row % 1000 == 0:
      print("   ", row)
    for col in range(row+1, matrix.shape[1]):
      weight = max(0.0, matrix[row, col])  # No negative weights
      if weight > max_distance_threshold:
        continue
      edge = g.add_edge(vlist[row], vlist[col])
      g.ep.weights[edge] = weight
      num_edges += 1
  print("  num_edges:", num_edges)
  print("  calculating closeness feature")
  closeness = graph_tool.centrality.closeness(g, weight=g.ep.weights, norm=True, harmonic=False)
  SaveFeature(closeness.a, CLOSENESS_FEATURE_NAME,
              matrix_num, matrix_to_wiki, calculated_features)

  print("  calculating eigenvector feature")
  eigenvalue, eigenvector = graph_tool.centrality.eigenvector(g, weight=g.ep.weights, max_iter=1000)
  SaveFeature(eigenvector.a, EIGENVECTOR_FEATURE_NAME,
              matrix_num, matrix_to_wiki, calculated_features)

  print("  calculating betweenness feature")
  betweenness, e_b = graph_tool.centrality.betweenness(g, weight=g.ep.weights)
  SaveFeature(betweenness.a, BETWEENNESS_FEATURE_NAME,
              matrix_num, matrix_to_wiki, calculated_features)

  print("  calculating degree feature")
  degree = np.array([CalcDegree(v, g.ep.weights) for v in vlist])
  SaveFeature(degree, DEGREE_FEATURE_NAME,
              matrix_num, matrix_to_wiki, calculated_features)

def main():
  os.makedirs(args.out_feature_data_dir, exist_ok=True)
  print("Loading corpus index")
  _, matrix_to_wiki = LoadCorpusIndex(args.corpus_index_file)
  
  # key = (wiki_filename, line_num)
  # val = { feature_name : feature_val }
  calculated_features = collections.defaultdict(dict)

  for matrix_filename in glob.iglob(args.distance_matrix_prefix + "*.npy"):
    matrix_num = int(os.path.basename(matrix_filename).split(".")[1])
    print("Processing matrix num:", matrix_num)
    ProcessDistanceMatrix(np.load(matrix_filename), matrix_num, matrix_to_wiki,
                          args.max_distance_threshold, calculated_features)

  print ("Saving features")  
  for in_file_name in glob.glob(os.path.join(args.in_training_data_dir, "*")):
    wiki_filename = os.path.basename(in_file_name)
    print("File:", wiki_filename)
    out_filename_prefix = os.path.join(args.out_feature_data_dir, wiki_filename)
    closeness_file = open(out_filename_prefix + "." + CLOSENESS_FEATURE_NAME, "w")
    eigenvector_file = open(out_filename_prefix + "." + EIGENVECTOR_FEATURE_NAME, "w")
    degree_file = open(out_filename_prefix + "." + DEGREE_FEATURE_NAME, "w")
    betweenness_file = open(out_filename_prefix + "." + BETWEENNESS_FEATURE_NAME, "w")
    for line_num, _ in enumerate(open(in_file_name)):
      feature_dict = calculated_features[(wiki_filename, line_num)]
      closeness_file.write(str(feature_dict[CLOSENESS_FEATURE_NAME]) + "\n")
      eigenvector_file.write(str(feature_dict[EIGENVECTOR_FEATURE_NAME]) + "\n")
      degree_file.write(str(feature_dict[DEGREE_FEATURE_NAME]) + "\n")
      betweenness_file.write(str(feature_dict[BETWEENNESS_FEATURE_NAME]) + "\n")


if __name__ == '__main__':
  main()
