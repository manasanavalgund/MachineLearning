import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

data_dict= {}
sample_data= None
labels= None
NUM_EPOCHS= 100000
X_train= []
X_test= []
Y_train= []
Y_test= []

def create_data():
  global data_dict, sample_data, labels, min_fval, max_fval
  X0, labels = make_blobs(n_samples=100, n_features=2, centers=2,
                     cluster_std=1.05, random_state=10)
  X1 = np.c_[np.ones((X0.shape[0])), X0]
  #Change labels of negative points to -1
  labels[labels == 0] = -1
  sample_data= X1
  positive_x =[]
  negative_x =[]
  for i,label in enumerate(labels):
    if label == -1:
      negative_x.append(X1[i])
    else:
      positive_x.append(X1[i])
  data_dict = {-1:np.array(negative_x), 1:np.array(positive_x)}

  #To extract maximum feature value in entire dataset
  max_fval = float('-inf')
  for y_i in data_dict:
    if np.amax(data_dict[y_i]) > max_fval:
      max_fval=np.amax(data_dict[y_i])

def split_train_test_data():
  global X_train, Y_train, X_test, Y_test
  N= len(sample_data)
  #First 80%: training data, remaining 20% : test data
  X_train= sample_data[:int(N*0.8)]
  X_test= sample_data[int(N*0.8):]
  Y_train= labels[:int(N*0.8)]
  Y_test= labels[int(N*0.8):]

def train(data_dict):
  global X_train, Y_train
  #Initialise weights with dimension as number of features
  w = np.zeros(X_train.shape[1])
  eta= 0.1*max_fval
  _lambda= 0.5
  weights= []
  #Setting seed so that np.random.uniform picks predictable z values
  np.random.seed(1)
  for epoch in range(1, NUM_EPOCHS):
    #Pick an instance from dataset at random with uniform prob.
    z = int(np.random.uniform(0, X_train.shape[0]))
    if (Y_train[z] * (np.dot(X_train[z][1:], w[1:]) + w[0])) >= 1:
      v= _lambda*w
    else:
      v= _lambda*w - Y_train[z]*X_train[z]
      w[0] = w[0] + Y_train[z]
    #Not updating bias with other weights so that bias is not regularized
    w[1:]= w[1:] - eta*v[1:]
    weights.append(w)
    if eta > 1e-4:
      eta= eta*0.1
  #Return avergae weights over epochs
  result_weight= 1/NUM_EPOCHS*(np.sum(weights, axis=0))
  return result_weight

def test(weights):
  misclassified= 0
  for i in range(X_test.shape[0]):
    y_predicted = np.sign(np.dot(X_test[i], weights))
    if y_predicted!= Y_test[i]:
      misclassified+= 1
  return X_test.shape[0], misclassified


def draw(weights):
  fig, ax = plt.subplots(figsize=(7, 5))
  fig.patch.set_facecolor('white')
  cdict = {-1: 'red', 1: 'blue'}

  for key, data in data_dict.items():
    ax.scatter(data_dict[key][:,1], data_dict[key][:,2], c=cdict[key])

  #Find 2 points with minimum and max value in feature dimension 1. Find corresponsing y values below to plot line
  min_x= np.min(sample_data[:,1])
  max_x= np.max(sample_data[:,1])

  #<w.x> + b = -1
  neg_y_xmin = (-weights[1] * min_x - weights[0] - 1) / weights[2]
  neg_y_xmax = (-weights[1] * max_x - weights[0] - 1) / weights[2]
  ax.plot([min_x, max_x], [neg_y_xmin, neg_y_xmax], 'k')

  #<w.x> + b = 1
  pos_y_xmin = (-weights[1] * min_x - weights[0] + 1) / weights[2]
  pos_y_xmax = (-weights[1] * max_x - weights[0] + 1) / weights[2]
  ax.plot([min_x, max_x], [pos_y_xmin, pos_y_xmax], 'k')

  #<w.x> + b = 0
  min_y = (-weights[1] * min_x - weights[0]) / weights[2]
  max_y = (-weights[1] * max_x - weights[0]) / weights[2]
  print((-weights[1] * 0 - weights[0]) / weights[2])
  ax.plot([min_x, max_x], [min_y, max_y], 'r--')
  plt.show()

def main():
  create_data()
  split_train_test_data()
  weights= train(data_dict)
  num_points, num_misclassified= test(weights)
  draw( weights)
  #Ouput data to Readme
  open('ReadMe.txt', 'w').close()
  with open('ReadMe.txt', 'a') as outputfile:
    print("the total number of data points on which your test method was run, : " + str(num_points), file=outputfile)
    print("the total number of data points on which your test method was misclassified, : " + str(num_misclassified), file=outputfile)

if __name__ == "__main__":
    main()
