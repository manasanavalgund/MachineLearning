import argparse
from csv import reader

global input_data, fold_errors

def initialize_global_variables():
    global fold_errors, input_data
    fold_errors = []
    input_data = []

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
    #Map input read as string to float
    input_data = [list(map(float, i)) for i in input_data]

#return dot product of 2 array
def dot(x, y):
    return sum(float(x_i)*float(y_i) for x_i, y_i in zip(x, y))

#Mocked signum function
def sign(x):
    if x > 0:
        return 1.
    else:
        return -1.

def evaluate_model(test_data, weights):
    global fold_errors
    error = 0
    for current_data_point in test_data:
        x = current_data_point[:-1]
        x.insert(0, 1)
        y_actual = -1 if int(current_data_point[-1]) == 0 else 1
        wx = dot(weights, x)
        y_predicted = sign(wx)
        delta = y_actual - y_predicted
        #if not y_actual * wx > 0:
        if delta != 0:
            error += 1
    fold_errors.append(error/len(test_data))


def train_model(mode, output_file):
    global input_data, fold_errors
    weights = [0] * (len(input_data[0]))
    if mode == "erm":
        #In erm mode, test data = train data
        output_weights = perceptron(input_data, input_data, weights)
        print('weights:', str(output_weights), file=output_file)
    elif mode == "cv":
        block = int(len(input_data) / float(10))
        for index in range(10):
            weights = [0] * (len(input_data[0]))
            test_start_offset = block*index
            test_data = input_data[test_start_offset:test_start_offset+block]
            train_data = input_data[:test_start_offset] + input_data[test_start_offset+block:]
            output_weights = perceptron(train_data, test_data, weights)
            print('weights:', str(output_weights), file=output_file)
    else:
        raise Exception('Incorrect mode!')
    return output_weights

def perceptron(train_data, test_data, weights):
    weights_updated = True
    #Stopping criteria: Maximum number of iterations = 1000
    num_iterations = 1000
    iteration_index = 0
    #Algorithm continues until weights are uodated in previous iteration and max iterations not crossed
    while weights_updated and iteration_index <= num_iterations:
        weights_updated = False
        index = 0
        iteration_index+= 1
        while index < len(train_data):
            current_data_point = train_data[index]
            y = -1 if current_data_point[-1] == 0 else 1
            x = current_data_point[:-1]
            x.insert(0, 1)
            wx = dot(weights, x)
            if y * wx <= 0:
                yx = [i * y for i in x]
                weights = [sum(x) for x in zip(weights, yx)]
                weights_updated = True
            else:
                index = index + 1
    evaluate_model(test_data, weights)
    return weights

def main():
    global input_data, fold_errors
    initialize_global_variables()
    dataset_url, mode = parse_arguments()
    load_data(dataset_url)
    open('output.txt', 'w').close()
    with open('output.txt', 'a') as outputfile:
        train_model(mode, outputfile)
        mean_error = sum(fold_errors)
        if mode == "cv":
            mean_error = mean_error/10
            print("Fold errors: " + str(fold_errors), file=outputfile)
        print("Average error : "+str(mean_error), file=outputfile)
    print("Reults redirected to output.txt")

if __name__ == '__main__':
    main()