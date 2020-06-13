import argparse
import math
import random
from csv import reader
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True)
  args = vars(parser.parse_args())
  return args['dataset']

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


def knn(X_train, row, k):
  neighbours= []
  for training_sample_row in X_train:
    distance= math.sqrt(sum([(a - b) ** 2 for a, b in zip(row[:-1], training_sample_row[:-1])]))
    neighbours.append((training_sample_row, distance))
  neighbours.sort(key=lambda x: x[1])
  k_nearest_neighbours= [x[0] for x in neighbours[:3]]
  k_nearest_neighbours_labels= [x[-1] for x in k_nearest_neighbours]
  prediction = max(set(k_nearest_neighbours_labels), key=k_nearest_neighbours_labels.count)
  return prediction

def split_train_test_data():
  global X_train, Y_train, X_test, Y_test
  N= len(data)
  #First 80%: training data, remaining 20% : test data
  X_train= data[:int(N*0.8)]
  X_test= data[int(N*0.8):]
  return X_train, X_test

def plot(scorelist, k_range):
  plt.plot(k_range, scorelist)
  plt.show()

def evaluate(X_train, X_test, k):
  correct= 0
  incorrect= 0
  train_X= [row[:-1] for row in X_train]
  train_Y= [row[-1] for row in X_train]
  test_X= [row[:-1] for row in X_test]
  test_Y= [row[-1] for row in X_test]
  knn_classifier= KNeighborsClassifier(k)
  knn_classifier.fit(train_X, train_Y)
  label= knn_classifier.predict(test_X)
  for idx in range(len(test_X)):
    if label[idx] == test_Y[idx]:
      correct+= 1
    else:
      incorrect+= 1
  print(str(correct / len(X_test)))

def normalize_data(data):
  np_data = np.array(data)
  mean_list = np.mean(np_data, axis=0)
  std_list = np.std(np_data, axis=0)
  for idx in range(len(data)):
    for col in range(len(data[idx]) - 1):
      data[idx][col] = (data[idx][col] - mean_list[col]) / std_list[col]
  return data

def main():
  global data
  dataset_url = parse_arguments()
  data= read_data(dataset_url)
  data= normalize_data(data)
  X_train, X_test= split_train_test_data()
  evaluate(X_train, X_test, 5)

if __name__=="__main__":
    main()