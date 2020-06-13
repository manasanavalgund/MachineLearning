import argparse
import math
import random
from csv import reader


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--k', required=True)
  args = vars(parser.parse_args())
  return args['dataset'], int(args['k'])

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
  #Calculate distances of all points in dataset
  for training_sample_row in X_train:
    distance= math.sqrt(sum([(a - b) ** 2 for a, b in zip(row[:-1], training_sample_row[:-1])]))
    neighbours.append((training_sample_row, distance))
  #Sort neighbours by distance
  neighbours.sort(key=lambda x: x[1])
  #Pick first k neighbours
  k_nearest_neighbours= [x[0] for x in neighbours[:k]]
  k_nearest_neighbours_labels= [x[-1] for x in k_nearest_neighbours]
  #Get majority label in k nearest neighbours
  prediction = max(set(k_nearest_neighbours_labels), key=k_nearest_neighbours_labels.count, default= float(random.randint(0,10)%2))
  return prediction

def split_train_test_data():
  global X_train, Y_train, X_test, Y_test
  N= len(data)
  #First 80%: training data, remaining 20% : test data
  X_train= data[:int(N*0.8)]
  X_test= data[int(N*0.8):]
  return X_train, X_test

def evaluate(X_train, X_test, k):
  correct= 0
  incorrect= 0
  for row in X_test:
    label= knn(X_train, row, k)
    #Correct predicted output
    if label == row[len(row) - 1]:
      correct+= 1
      #Incorrect predicted output
    else:
      incorrect+= 1
  open('output.txt', 'w').close()
  with open('output.txt', 'a') as outputfile:
    print("Accuracy on test dataset : " + str((correct/len(X_test))*100) + "%", file=outputfile)
    print("Error rate on test dataset : " + str((incorrect / len(X_test))*100) + "%", file=outputfile)
  print("Accuracy on test dataset : " + str((correct/len(X_test))*100) + "%")
  print("Error rate on test dataset : " + str((incorrect / len(X_test))*100) + "%")


def main():
  global data
  #Two required parameters: dataset and k value for knn
  dataset_url, k = parse_arguments()
  data= read_data(dataset_url)
  #SPlit data in 80:20 proportion
  X_train, X_test= split_train_test_data()
  evaluate(X_train, X_test, k)

if __name__=="__main__":
    main()