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


'''
*Global seed for random number generator
'''
random.seed(123)

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

 # derivation of sigmoid (for backprop)
def sigmoid(A, deriv=False): 
    if deriv:
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A
alfa = 0.005
epoch = 400
neuron = [4, 3] 

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
bias = [0 for i in range(neuron[1])]

for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

print("Total Cost:")
for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X):
        
        # Forward propagation
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
    if(e % 100 == 0):
        print(cost_total)
        
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
print(acc / len(preds) * 100, "%")
