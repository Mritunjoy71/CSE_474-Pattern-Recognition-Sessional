from math import sqrt, exp
import numpy as np

file = open('input.txt')

f1 = file.readlines()

numFeatures, numClass, datasetLen = 0, 0, 0

dataset = []

count = 0
for line in f1:
    if count == 0:
        var = line.split()
        numFeatures = int(var[0])
        numClass = int(var[1])
        datasetLen = int(var[2])
    else:
        var = line.split()
        dataset.append([int(x) for x in var])
    count += 1


average = []

for i in range(numFeatures):
    sum = 0
    for vector in dataset:
        sum += vector[i]

    average.append(float(sum/datasetLen))

print(average)

# individual co-varience

co_varience = []

for i in range(numFeatures):
    sum = 0
    for vector in dataset:
        sum += (vector[i] - average[i])**2

    co_varience.append(float(sum/datasetLen))

standard_deviation = [sqrt(x) for x in co_varience]

print(co_varience)

# co-varience matrix

co_varience_matrix = []

for i in range(numFeatures):
    row = []
    for j in range(numFeatures):
        sum = 0
        for vector in dataset:
            sum += (vector[i]-average[i])*(vector[j]-average[j])
        row.append(float(sum/datasetLen))
    co_varience_matrix.append(row)

print(co_varience_matrix)

co_varience_matrix = np.matrix(co_varience_matrix)

print(co_varience_matrix)

inverse_co_varience_matrix = np.linalg.inv(co_varience_matrix)

print(inverse_co_varience_matrix)

feature_vectors = []

for data in dataset:
    data.pop()
    feature_vector = np.matrix(data)
    feature_vector = np.transpose(feature_vector)
    feature_vectors.append(feature_vector)

average = np.matrix(average).transpose()

det = np.linalg.det(co_varience_matrix)

normal_distribution_values = []

for vector in feature_vectors:
    x_u = np.matrix(vector-average)
    value = (1/(((2*3.1416)**(float(numFeatures/2)))*sqrt(det)))*(exp(-0.5 *(x_u.transpose()*(inverse_co_varience_matrix*x_u))))
    normal_distribution_values.append(value)

print(normal_distribution_values)

