import argparse
import copy
import math
from csv import reader

import matplotlib.pyplot as plt

global input_data, decision_stumps
global folds, fold_errors

def initialize_global_variables():
    global decision_stumps, fold_errors
    decision_stumps = []
    fold_errors = []

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', required=True)
    args = vars(parser.parse_args())
    return args['dataset'], args['mode']

def load_data(url):
    global input_data
    with open(url, 'r') as file:
        input_data = []
        file_reader = reader(file)
        for row in file_reader:
            input_data.append(row)
    input_data.pop(0)

#Prediction function. Evaluates accuracy/ error on test data
def evaluate(w_list, h_list, test_data):
    global fold_errors
    h  = [0]*len(test_data)
    #applying the output hypothesis on train data: w_list: weights list h_list: hypothesis list
    for itr in range(len(w_list)):
        best_split, best_feature  = h_list[itr]
        h_t = [(-1 if row[best_feature] < best_split else 1) for row in test_data]
        wthx = [(h_t[j]*w_list[itr]) for j in range(len(h_t))]
        h = [(h[i]  + wthx[i]) for i in range(len(test_data))]
    error = 0
    for i in range(len(test_data)):
        if sign(h[i]) != test_data[i][-1]:
            error = error+1
    #Append error to fold error which maintains overall errors in all folds. 1 fold in case of erm.
    fold_errors.append(error/len(test_data))


def train_algorithm(mode, file, rounds):
    global input_data, fold_errors
    input_data = [list( map(float,i) ) for i in input_data]
    #Convert all 0 labels to -1
    for row in input_data:
        if row[-1] == 0:
            row[-1] = -1
    #Test data = train data = input data in erm
    if mode == "erm":
        w,h = adaboost(input_data, input_data, rounds)
        print(str(w)+"\n", file=file)
    elif mode == "cv":
        #Create folds by slicing using index
        fold_range = int(len(input_data) / float(10))
        for index in range(10):
            test_start_offset = fold_range*index
            test_data = input_data[test_start_offset:test_start_offset+fold_range]
            train_data = input_data[:test_start_offset] + input_data[test_start_offset+fold_range:]
            w,h = adaboost(train_data, test_data, rounds)
            print(str(w)+"\n", file=file)

def weak_learner(train_data, D):
    data = copy.deepcopy(train_data)
    #Append distribution vector D as last column of train data, to maintain index of distribution when sorted
    for i in range(len(train_data)):
        data[i].append(D[i])
    F_star = float('inf')
    theta_star = None
    j_star = None
    additional_row = [0]*len(data[0])
    for j in range(len(data[0]) - 2):
        data.sort(key = lambda data: data[j])
        sorted_data_by_column  = copy.deepcopy(data)
        additional_row[j] = sorted_data_by_column[len(sorted_data_by_column) - 1][j] + 1
        sorted_data_by_column.append(additional_row)
        F = [(data[idx][-1] if sorted_data_by_column[idx][-2] == 1 else 0) for idx in range(len(sorted_data_by_column) - 1)]
        F = sum(F)
        if F < F_star:
            F_star = F
            theta_star = sorted_data_by_column[0][j] - 1
            j_star = j
        for i in range(len(sorted_data_by_column) - 1):
            F = F - sorted_data_by_column[i][-2]* data[i][-1]
            if F < F_star and sorted_data_by_column[i][j] != sorted_data_by_column[i+1][j]:
                F_star = F
                theta_star = 0.5*(sorted_data_by_column[i][j] + sorted_data_by_column[i+1][j])
                j_star = j
    #theta_star = best split value for dimension = j_star
    return theta_star, j_star

def adaboost(train_data, test_data, rounds):
    D = [1/len(train_data)] * len(train_data)
    hypothesis = []
    w_list = []
    for t in range(rounds):
        best_split, best_feature = weak_learner(train_data, D)
        hypothesis.append([best_split, best_feature])
        # Create hypothesis using received best split and best feature
        h_t = [(-1 if row[best_feature] < best_split else 1) for row in train_data]
        e_t = 0
        for i in range(len(h_t)):
            if h_t[i] != train_data[i][-1]:
                e_t  = e_t + D[i]
        w_t = 0.5*math.log(1/e_t - 1)
        w_list.append(w_t)
        denominator = 0
        for j in range(len(train_data)):
            denominator+= D[j]*math.exp(-1*w_t*train_data[j][-1]*h_t[j])
        for i in range(len(train_data)):
            numerator = (D[i] * math.exp(-1 * w_t * train_data[i][-1]*h_t[i]))
            D[i] = numerator/denominator
    evaluate(w_list,hypothesis,test_data)
    return w_list, hypothesis

#Mocked signum function
def sign(x):
    if x > 0:
        return 1.
    else:
        return -1.

def main():
    global input_data, decision_stumps, fold_errors
    initialize_global_variables()
    mean_errors = []
    dataset_url, mode = parse_arguments()
    load_data(dataset_url)
    # Since this is T which gives minimum error
    T = [60]
    open('output.txt', 'w').close()
    with open('output.txt', 'a') as output_file:
        for t in T:
            fold_errors = []
            train_algorithm(mode, output_file, t)
            mean_error = sum(fold_errors)
            if mode == "cv":
                print("Fold errors: " + str(fold_errors), file=output_file)
                mean_error = mean_error/10
            print("Average error : " + str(mean_error), file=output_file)
            mean_errors.append(mean_error)
    # Draw plot of error vs T
    plt.plot(T, mean_errors)
    plt.xlabel('Number of iterations(T)')
    plt.ylabel('Error')
    plt.show()
    print("Reults redirected to output.txt")

if __name__ == '__main__':
    main()