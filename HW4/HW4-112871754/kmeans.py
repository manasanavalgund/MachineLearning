import argparse
import math
import random
from csv import reader

import numpy as np


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--distance', default="Euclidean")
  args = vars(parser.parse_args())
  return args['dataset'], args['distance']

def read_data(url):
  global input_data
  with open(url, 'r') as file:
    input_data = []
    file_reader = reader(file)
    for row in file_reader:
      input_data.append(row)
  input_data.pop(0)
  # Map input read as string to float
  input_data = [list(map(float, i)) for i in input_data]
  return input_data

def normalize_data(data):
  np_data = np.array(data)
  mean_list = np.mean(np_data, axis=0)
  std_list = np.std(np_data, axis=0)
  for idx in range(len(data)):
    for col in range(len(data[idx]) - 1):
      data[idx][col] = (data[idx][col] - mean_list[col]) / std_list[col]
  return data

def find_nearest_centroid(distance_type, data, centroids, idx):
  row= data[idx]
  min_distance= None
  cluster_id= None
  for idx in range(len(centroids)):
    centroid= centroids[idx]
    if distance_type == "Manhattan":
      distance = (sum([abs(a - b) for a, b in zip(row[:-1], centroid[:-1])]))
    else:
      distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(row[:-1], centroid[:-1])]))
    if min_distance is None or distance < min_distance:
      min_distance= distance
      cluster_id= idx
  return cluster_id

def mean(a):
  return sum(a) / len(a)

def update_centroids(clusters):
  centroids= []
  for cluster in clusters:
    cluster_rows = [data[i] for i in cluster]
    centroids.append([sum(col) / float(len(col)) for col in zip(*cluster_rows)])
  return centroids

def kmeans(distance_type):
  converged = False
  k=2
  centroid_indices= random.sample(range(0, len(data)), k)
  clusters= [[] for i in range(len(centroid_indices))]
  centroids= [data[idx] for idx in centroid_indices]
  while not converged:
    converged= True
    for idx in range(len(data)):
      old_cluster_id= None
      for index in range(len(clusters)):
        cluster= clusters[index]
        if idx in cluster:
          old_cluster_id= index
          break
      new_cluster_id= find_nearest_centroid(distance_type, data, centroids, idx)
      if old_cluster_id is None or old_cluster_id != new_cluster_id:
        converged= False
        if old_cluster_id is not None:
          clusters[old_cluster_id].remove(idx)
        clusters[new_cluster_id].append(idx)
    centroids= update_centroids(clusters)
  return clusters

def evaluate(clusters):
  for idx in range(len(clusters)):
    positive = 0
    negative = 0
    cluster = clusters[idx]
    for item in cluster:
      row = data[item]
      if row[len(row) - 1] == 1.0:
        positive += 1
      else:
        negative += 1
    print("Cluster " + str(idx) + " stats")
    print("Number of positive samples "+str(idx) + ": " + str(positive))
    print("Number of negative samples "+str(idx) + ": " + str(negative))
    print("Total number of samples "+str(idx) + ": " + str(negative + positive))
    print("Percentage of positive samples " + ": " + str(positive/len(cluster)*100))


def main():
  global data
  dataset_url, distance_type = parse_arguments()
  data= read_data(dataset_url)
  np.random.shuffle(data)
  data= normalize_data(data)
  clusters= kmeans(distance_type)
  evaluate(clusters)


if __name__=="__main__":
    main()