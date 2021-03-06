'''
*Authors: Edgar Oregel, Clement Garcia
*Main File: irisalgo.py
*Purpose: The main file with calculations and algo for computing iris dataset
*Last Updated: 4/30/18
    @Edgar/@Clement
    *Starting out
'''



'''
*Imported modules
'''
import csv
import random
import math
import time as T
import multiprocessing as MP
import argparse
import sys

'''
*Global seed for random number generator
'''
random.seed(123)
parser = argparse.ArgumentParser()


#Handle arguments
parser.add_argument("-t", help="number of threads", type=int)
args = parser.parse_args()
if not args.t:
    parser.error("Missing thread count, -t <count>")
    sys.exit()
THREADS = args.t

# Load dataset
with open('dataset.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None) # skip header
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in range(len(row))]

# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

# Matrix multiplication 
def matrix_mul_bias(A, B, bias): 
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C
 # Vector (A) x matrix (B) multiplication
def vec_mat_bias(A, B, bias):
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C

# Matrix (A) x vector (B) multipilicatoin (for backprop)
def mat_vec(A, B): 
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False): 
    if deriv: 
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

#divide the data equally between the threads
def divideData(items):
    divided_elements = {}
    i = 0
    amount = items/THREADS
    not_even = False
    not_even_total = 0
    total = 0
    thread_names = []

    #check for decimals to round 
    if not type(amount) != "integer":
        temp_amount = math.ceil(amount)
        not_even = True

    for x in range(THREADS):
        thread_name = "thread"
        thread_name += str(i)
        thread_names.append(thread_name)

        if not_even:
            not_even_total += temp_amount
            if (not_even_total + temp_amount) > len(items):
                temp_amount = len(items) - temp_amount
                divided_elements[thread_name] = temp_amount
            else:
                divided_elements[thread_name] = not_even_total
        else:
            total += amount
            divided_elements[thread_name] = total
        i += 1
    return divided_elements, thread_names


def computationalFunction(alfa, epoch, neuron, weight, bias, x, y, thread, start_time, thread_times):
    for e in range(x, y):
        cost_total = 0
        for idx, x in enumerate(train_X):
            h_1 = vec_mat_bias(x, weight, bias)
            X_1 = sigmoid(h_1)
            target = [0, 0, 0]
            target[int(train_y[idx])] = 1
            eror = 0
            for i in range(3):
                eror +=  0.5 * (target[i] - X_1[i]) ** 2 
            cost_total += eror
            delta = []
            for j in range(neuron[1]):
                delta.append(-1 * (target[j]-X_1[j]) * X_1[j] * (1-X_1[j]))

            for i in range(neuron[0]):
                for j in range(neuron[1]):
                    weight[i][j] -= alfa * (delta[j] * x[i])
                    bias[j] -= alfa * delta[j]

        cost_total /= len(train_X)
        # if(e % 100 == 0):
        #     print(cost_total)
    thread_end_time = T.time()
    thread_end_time = thread_end_time - start_time
    thread_times.append(thread_end_time)
    

def main():
    alfa = 0.005
    epoch = 400
    neuron = [4, 3]

    start = T.time()
    weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
    bias = [0 for i in range(neuron[1])]
    for i in range(neuron[0]):
        for j in range(neuron[1]):
            weight[i][j] = 2 * random.random() - 1

    
    elements, thread_names = divideData(epoch);

    thread_times = []

    thread_processes = []

    first = True
    for thread in thread_names:
        if first:
            x = 0
            y = elements[thread]
        else:
            x = elements[thread]
            y = elements[thread] + elements[thread]
        thread_start_time = T.time()
        thread = MP.Process(target=computationalFunction, args=(alfa, epoch, neuron, weight, bias, x, y, thread, thread_start_time, thread_times))
        thread_processes.append(thread)
        thread.start()
    # for e in range(epoch):
    #     cost_total = 0
    #     for idx, x in enumerate(train_X):
    #         h_1 = vec_mat_bias(x, weight, bias)
    #         X_1 = sigmoid(h_1)
    #         target = [0, 0, 0]
    #         target[int(train_y[idx])] = 1
    #         eror = 0
    #         for i in range(3):
    #             eror +=  0.5 * (target[i] - X_1[i]) ** 2 
    #         cost_total += eror
    #         delta = []
    #         for j in range(neuron[1]):
    #             delta.append(-1 * (target[j]-X_1[j]) * X_1[j] * (1-X_1[j]))

    #         for i in range(neuron[0]):
    #             for j in range(neuron[1]):
    #                 weight[i][j] -= alfa * (delta[j] * x[i])
    #                 bias[j] -= alfa * delta[j]

    #     cost_total /= len(train_X)
    #     if(e % 100 == 0):
    #         print(cost_total)
    # for thread in thread_processes:
    #     thread.join()
    stop = T.time()
    elapsed = stop - start
    print("\nTotal Time Elapsed:")
    print(str(elapsed) + " sec")
    # print("\nTotal Time Elapsed Per Thead: ")
    # for i in range(len(thread_times)):
    #     print(thread_names[i] + " time: " + thread_times[i])
    res = matrix_mul_bias(test_X, weight, bias)

    # Get prediction
    preds = []
    for r in res:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])

    # Print prediction
    print("\nPredictions:")
    for item in preds:
        print(item)

    # Calculate accuration
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(test_y[i]):
            acc += 1
    print("\nAccuracy:")
    print(str(acc / len(preds) * 100) + "%")

if __name__ == "__main__":
    p = MP.Pool(5)
    main()