import numpy as np
from sklearn.preprocessing import MinMaxScaler


def reading_dataset(filename): 
  file = open(filename) 

  lines = file.readlines() 

  X = [] 
  Y_class = [] 
  Y = [] 

  for line in lines:
    var = line.split() # spacing er through split hoise var -> string array 
    x = [float(each) for each in var[:-1]] # var[:-1] -> float convertion -> append to x
    y = int(var[-1]) # var[-1] -> last element 

    X.append(x)
    Y_class.append(y) # 1,2,3,4


  m = max(Y_class) # Y_class e max value koto m

  n = len(X) # number of input, n = 500

  for i in range(n):
    y = [0]*m # m size 0 zero create kora
    y[Y_class[i]-1] = 1
    Y.append(y)

  file.close()

  return np.array(X),np.array(Y)

X,Y = reading_dataset('trainNN.txt') 
X_test,Y_test = reading_dataset('testNN.txt')


num_input_neurons = X.shape[1]
num_output_neurons = Y.shape[1]
no_of_hidden_layers=int(input('Input no. of hidden layers :'))
no_of_nodes_per_layer=int(input('Input no. of nodes per layer :'))
K=[0]*(no_of_hidden_layers+2)
K[0]=num_input_neurons
K[no_of_hidden_layers+1]=num_output_neurons
for i in range(1,no_of_hidden_layers+1):
    K[i]=no_of_nodes_per_layer
L=len(K)-1
l_rate = 0.5
thres = 5
Num_samples = X.shape[0]
#print(N)

#print('Layer wise neuron count from hidden layer:',k[1:])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#print(X_scaled)

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)
#print(X_test_scaled)

# Generate weight vector for each layers
W = []
for r in range(0, L):
    w = np.random.uniform(-1, 1, (K[r + 1], K[r] + 1))
    #print(w.shape)
    W.append(w)

#print(len(W))


def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_der(x):
    return 1*sigmoid(x)*(1-sigmoid(x))


def forward_propagation(input_x):
    #hidden layer starts at position 0

    y = []  # per layer output
    v = []  # per layer without sigmoid
    input_vector = [1]
    input_vector.extend(input_x)

    for r in range(L):
        vr = np.dot(W[r], input_vector)
        #print(vr)
        yr = sigmoid(vr)
        #print(yr)
        input_vector = [1]
        input_vector.extend(yr)
        y.append(yr)
        v.append(vr)
    #print(y[L-2])
    return v, y


def back_propagation(v, y, instance_x, instance_y):
    deltas=[0]*L
    for r in reversed(range(L)):
        if r == L-1:
            deltas[r] = np.multiply((y[r] - instance_y),sigmoid_der(v[r]))
        else:
            error = np.dot(deltas[r+1],W[r+1][:,1:])
            deltas[r] = np.multiply(error,sigmoid_der(v[r]))
            
    #print(deltas)

    for r in range(L):
        for j in range(0, K[r + 1]):
            
            if r == 0:
                t = [1]
                t.extend(instance_x)
                Del_weight = np.multiply(-l_rate * deltas[r][j], np.array(t))
                W[r][j] = W[r][j] + Del_weight
            else:
                t = [1]
                t.extend(y[r-1])
                Del_weight = np.multiply(-l_rate * deltas[r][j],t)
                W[r][j] = W[r][j] + Del_weight
            
            
            #print(W[r][j],Del_weight)
    


def cost_calculation(y_got, y_instance):
    '''summation = 0
    for j in range(len(y_instance)):
        euclid_dis = distance.euclidean(y_m[j], y_instance[j])
        summation += euclid_dis
    #print(summation)

    return summation
'''
    return 0.5*np.sum((y_got-y_instance)**2,axis=0)


MAX_ITERATION=100
for epoch in range(MAX_ITERATION):
    print(epoch)
    y_m = []  # actual output of neural network
    total_cost=0

    for instance_x, instance_y in zip(X, Y):
        # forward propagation
        v, y = forward_propagation(instance_x)
        y_m.append(y[L - 1])
        back_propagation(v, y, instance_x, instance_y)
        total_cost += cost_calculation(y[L-1],instance_y)

    print(total_cost)
    if (total_cost < thres):
        break

def test_data(data_X,data_Y):
    count = 0
    for instance_x, instance_y in zip(data_X, data_Y):
        v, y = forward_propagation(instance_x)

        if np.argmax(y[L - 1]) == np.argmax(instance_y):
            count += 1
    return count        

print("Accuracy is on Train Data Set:   " ,test_data(X,Y)/len(X)*100)
print("Accuracy is on Test Data Set:   " ,test_data(X_test,Y_test)/len(X_test)*100)

print("hello")