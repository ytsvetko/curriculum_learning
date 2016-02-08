#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np
import json
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser()
parser.add_argument("--training_dir", default="../train")
parser.add_argument("--dimentionality", default=300, type=int)
parser.add_argument("--out_svd_result_matrix", default="data/svd.npy")
parser.add_argument("--out_unique_distance_matrix_prefix", default="data/distances_unique.")
parser.add_argument("--out_unique_kmeans_labels", default="data/unique_kmeans_labels.npy")
parser.add_argument("--out_inv_idx", default="data/inv_idx.npy")
parser.add_argument("--out_idx", default="data/idx.npy")
parser.add_argument("--out_corpus_index", default="data/index.json")
args = parser.parse_args()


def LoadCorpus(corpus_dir):
  corpus = []
  file_index = []
  for in_file_name in sorted(glob.iglob(os.path.join(corpus_dir, "*"))):
    print("Loading:", in_file_name)
    for line_num, line in enumerate(open(in_file_name)):
      corpus.append(line.strip())
      file_index.append((in_file_name, line_num))
  return corpus, file_index


def CalcDistances(unique_labels, label, unique_X):
  idx = np.where(unique_labels == label)[0]
  if len(idx) == 0:
    return np.array([])
  M = unique_X[idx]
  return pairwise_distances(M, metric='cosine', n_jobs=1)


def GetCorpusIndex(file_index, labels, unique_labels, inv_idx):
  assert len(labels) == len(file_index), (len(labels), len(file_index))
  unique_distance_indexes = np.zeros(len(unique_labels)) - 1
  for l in range(unique_labels.max()+1):
    unique_l_idx = np.where(unique_labels == l)[0]
    unique_distance_indexes[unique_l_idx] = np.arange(len(unique_l_idx))
  assert unique_distance_indexes.min() >= 0.0
  distance_indexes = unique_distance_indexes[inv_idx]
  assert len(distance_indexes) == len(labels)
  result = []
  for i, (filename, line_num) in enumerate(file_index):
    result.append((filename, line_num, int(labels[i]), int(distance_indexes[i])))
  return result


def KMeans(M, max_cluster_size=10000, num_clusters=10, label_start=0, nest_level=0):
  print("Running KMeans on matrix with size:", M.shape[0], "labels start:", label_start,
        "nest_level", nest_level)
  labels = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                           init_size=1000, batch_size=1000,
                           random_state=12345).fit_predict(M)
  labels += label_start
  label_start += num_clusters
  counts = np.bincount(labels)
  print("counts.max:", counts.max())
  print("counts.mean:", counts.mean())
  print("counts.std:", counts.std())
  
  clusters_to_split = np.where(counts > max_cluster_size)[0] 
  print("counts > max_cluster_size:", len(clusters_to_split))
  for cluster in clusters_to_split:
    idx = np.where(labels == cluster)[0]
    sub_matrix = M[idx]
    sub_labels, label_start = KMeans(sub_matrix, max_cluster_size, num_clusters, label_start, nest_level+1)
    labels[idx] = sub_labels
  return labels, label_start

  
def main():
  if os.path.exists(args.out_svd_result_matrix):
    print("Loading SVD matrix from file")
    X = np.load(args.out_svd_result_matrix)
    print("Loading corpus")
    _, file_index = LoadCorpus(args.training_dir)
  else:
    print("Loading corpus")
    corpus, file_index = LoadCorpus(args.training_dir)
    print("Building TF-IDF")
    tf_idf = TfidfVectorizer(input="content", lowercase=False)
    X = tf_idf.fit_transform(corpus)
    del corpus
    print("Running LSA")
    svd = TruncatedSVD(args.dimentionality)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    print("Saving SVD results")
    np.save(args.out_svd_result_matrix, X)
  if os.path.exists(args.out_inv_idx) and os.path.exists(args.out_unique_kmeans_labels) and os.path.exists(args.out_idx):
    print("Loading labels")
    unique_labels = np.load(args.out_unique_kmeans_labels)
    inv_idx = np.load(args.out_inv_idx)
    idx = np.load(args.out_idx)
    unique_X = X[idx]
  else:
    print("Unique matrix")
    b = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, idx, inv_idx = np.unique(b, return_index=True, return_inverse=True)
    print("Saving inv_idx")
    np.save(args.out_inv_idx, inv_idx)
    print("Saving idx")
    np.save(args.out_idx, idx)
    unique_X = X[idx]
    print("Running K-Means")
    unique_labels, _ = KMeans(unique_X)
    print("Save unique K-Means labels")
    np.save(args.out_unique_kmeans_labels, unique_labels)
  print("Re-label non-unique")
  labels = unique_labels[inv_idx]

  for l in range(unique_labels.max()+1):
    out_filename = args.out_unique_distance_matrix_prefix + str(l) + ".npy"
    if os.path.exists(out_filename):
      continue
    print("Calculating distance matrix for label:", l)
    D = CalcDistances(unique_labels, l, unique_X)
    print("Saving to distance matrix to file")
    np.save(out_filename, D)

  if not os.path.exists(args.out_corpus_index):
    print("Calculating corpus index")
    corpus_index = GetCorpusIndex(file_index, labels, unique_labels, inv_idx)
    print("Saving corpus index")
    json.dump(corpus_index, open(args.out_corpus_index, "w"))

if __name__ == "__main__":
  main()
