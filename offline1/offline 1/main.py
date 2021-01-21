import numpy as np

input_file = open("trainLinearlySeparable.txt")
input_lines = input_file.readlines()
d_feature = 0
m_class = 0
n_samples = 0

train_LS = []
line_count = 0

for line in input_lines:
    if line_count == 0:
        L = line.split()
        d_feature = int(L[0])
        m_class = int(L[1])
        n_samples = int(L[2])
        #print(L)
    else:
        L = line.split()
        line_data = []
        for i in range(d_feature):
            line_data.append(float(L[i]))
        line_data.append(int(L[d_feature]))
        #print(line_data)
        train_LS.append(line_data)
    line_count = line_count + 1

input_file = open("testLinearlySeparable.txt")
input_lines = input_file.readlines()

test_LS = []
line_count = 0

for line in input_lines:
    L = line.split()
    line_data = []
    for i in range(d_feature):
        line_data.append(float(L[i]))
    line_data.append(int(L[d_feature]))
    #print(line_data)
    test_LS.append(line_data)

#__________Reward and Punishment Algorithm_________#
print("\nReward and punishment algorithm\n")
max_iteration = 1000
np.random.seed(41)
t = 0
row_not = 0.027
W = np.random.uniform(-10, 10, d_feature + 1)
print("Initial weight", W)

for itr in range(max_iteration):
    count = 0
    for train_data in train_LS:
        class_no = train_data[-1]
        X = train_data[:-1]
        X.append(1)
        t_data = np.array(X)
        #print(X)
        X = np.array(X).reshape(-1, 1)
        #print(X)
        mult = np.dot(W, X)[0]
        #print(mult)
        if mult <= 0 and class_no == 1:
            W = W + row_not * np.array(t_data)
            count = count + 1
        elif mult > 0 and class_no == 2:
            W = W - row_not * np.array(t_data)
            count = count + 1
        else:
            pass
    if count == 0:
        break

print("Final weight", W)


def Testdata(test_data, weight):
    print("sample no.\tfeature values\t\t\tactual class\tpredicted class")
    accuracy_count = 0
    sample_count = 0
    misclassified = []
    for d in test_data:
        sample_count += 1
        X = np.array(d)
        no_class = X[d_feature]
        X[d_feature] = 1
        X = X.reshape(d_feature + 1, 1)
        mult = np.dot(weight, X)[0]
        class_predicted = 0
        if mult >= 0:
            class_predicted = 1
        else:
            class_predicted = 2
        if class_predicted == no_class:
            accuracy_count = accuracy_count + 1
        else:
            misclassified.append(sample_count)
        print(sample_count, "\t", d[:d_feature], "\t", no_class, "\t",
              class_predicted)

    if len(misclassified) == 0:
        print("No misclassified sample")
    else:
        print("misclassified sample numbers :", misclassified)
    print("Accuracy is  ", float((accuracy_count / len(test_data)) * 100))
    return float(accuracy_count)


Testdata(test_LS, W)

#_____________BASIC PERCEPTRON ALGORUTHM_______________#

print("\n\nBasic perceptron Algorithm\n")
max_iteration = 1000
np.random.seed(41)
t = 0
row_not = 0.027
W = np.random.uniform(-10, 10, d_feature + 1)
print("Initial weight", W)

for itr in range(max_iteration):
    del_x = []
    Y = []
    for train_data in train_LS:
        X = np.array(train_data)
        #print(X)
        class_no = X[d_feature]
        X[d_feature] = 1
        X = X.reshape(d_feature + 1, 1)
        #print(X)
        mult = np.dot(W, X)[0]
        #print(mult)
        if mult < 0 and class_no == 1:
            Y.append(X)
            del_x.append(-1)
        elif mult > 0 and class_no == 2:
            Y.append(X)
            del_x.append(1)
        else:
            pass
    summation = np.zeros(d_feature + 1)
    for j in range(len(Y)):
        summation = summation + del_x[j] * Y[j].transpose()[0]

    W = W - row_not * summation

    if len(Y) == 0:
        break

print("Final weight", W)

Testdata(test_LS, W)

#__________Pocket Algorithm__________#

print("\n\nPocket algorithm\n")
input_file = open("trainLinearlyNonSeparable.txt")
input_lines = input_file.readlines()

train_NLS = []
line_count = 0

for line in input_lines:
    if line_count == 0:
        L = line.split()
        d_feature = int(L[0])
        m_class = int(L[1])
        n_samples = int(L[2])
        #print(L)
    else:
        L = line.split()
        line_data = []
        for i in range(d_feature):
            line_data.append(float(L[i]))
        line_data.append(int(L[d_feature]))
        #print(line_data)
        train_NLS.append(line_data)
    line_count = line_count + 1

input_file = open("testLinearlyNonSeparable.txt")
input_lines = input_file.readlines()

test_NLS = []
line_count = 0

for line in input_lines:
    L = line.split()
    line_data = []
    for i in range(d_feature):
        line_data.append(float(L[i]))
    line_data.append(int(L[d_feature]))
    #print(line_data)
    test_NLS.append(line_data)

np.random.seed(41)
t = 0
row_not = 0.027
W = np.random.uniform(-10, 10, d_feature + 1)
print("Initial weight", W)
Wp = W

M = len(test_NLS)

for itr in range(max_iteration):
    del_x = []
    Y = []
    count = 0
    for train_data in train_LS:
        X = np.array(train_data)
        #print(X)
        class_no = X[d_feature]
        X[d_feature] = 1
        X = X.reshape(d_feature + 1, 1)
        #print(X)
        mult = np.dot(W, X)[0]
        #print(mult)
        if mult < 0 and class_no == 1:
            Y.append(X)
            del_x.append(-1)
            count = count + 1
        elif mult > 0 and class_no == 2:
            Y.append(X)
            del_x.append(1)
            count = count + 1
        else:
            pass

    summation = np.zeros(d_feature + 1)
    for j in range(len(Y)):
        summation = summation + del_x[j] * Y[j].transpose()[0]

    W = W - row_not * summation
    if count < M:
        M = count
        Wp = W
    if count == 0:
        break

print("Final weight", Wp)

Testdata(test_NLS, Wp)
