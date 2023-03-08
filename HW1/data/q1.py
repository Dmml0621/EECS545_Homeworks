import numpy as np
# from tqdm import tqdm
from numpy.linalg import inv
import matplotlib.pyplot as plt

Q1x_train = np.load('q1xTrain.npy')
Q1y_train = np.load('q1yTrain.npy')
Q1x_test = np.load('q1xTest.npy')
Q1y_test = np.load('q1yTest.npy')


# a
N = len(Q1x_train)
x_train = np.append(Q1x_train.reshape(N, 1), np.ones((N, 1)), axis=1)
y_train = Q1y_train

def BGD(x_train, y_train, iterations=1000, degree=2, lr=0.01): 
    # input should be column vectors
    W = np.ones(degree) # initialize weight to one
    for i in range(iterations):
        loss = np.dot(x_train, W) - y_train
        # if i % 3 == 0 and i < 80:
        #     print(sum(loss ** 2) / len(x_train))
        W -= lr * np.dot(loss, x_train)
    return W

def SGD(x_train, y_train, iterations=1000, degree=2, lr=0.01):
    # input should be column vectors
    N = len(y_train)
    W = np.ones(degree) # initialize weight to one
    for i in range(iterations):
        loss_iter = 0
        for n in range(N):
            loss = np.dot(x_train[n], W) - y_train[n]
            W -= lr * np.dot(loss, x_train[n])
            loss_iter += loss ** 2
        # if i % 3 == 0 and i < 80:
        #     print(loss_iter / len(x_train))
    return W

print(BGD(x_train, y_train, iterations=1000, lr=0.01))
print(SGD(x_train, y_train, iterations=1000, lr=0.01))


# b
def generate_phi(x, degree):
    phi = np.ones(len(x)).reshape(len(x), 1)
    for i in range(0, degree):
        phi = np.append(phi, pow(x, i + 1).reshape(len(x), 1), axis=1)
    # highest power first in result
    return np.flip(phi, axis=1)

def closed_form(x_train, y_train):
    # highest power first in result
    return np.dot(inv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))

train_error = list()
test_error = list()

y_test = Q1y_test
degs = range(10)

for deg in degs:
    train_feature = generate_phi(Q1x_train, deg)
    test_feature = generate_phi(Q1x_test, deg)
    W = closed_form(train_feature, y_train)
    train_dy = train_feature @ W - y_train
    train_rms = ((train_dy.T @ train_dy) / len(y_train)) ** 0.5
    test_dy = test_feature @ W - y_test
    test_rms = ((test_dy.T @ test_dy) / len(y_test)) ** 0.5
    train_error.append(train_rms)
    test_error.append(test_rms)

plt.plot(list(degs), train_error, '-bo', label='Training')
plt.plot(list(degs), test_error, '-ro', label='Test')
plt.legend()
_ = plt.xlabel(r'M')
_ = plt.ylabel(r'$E_{RMS}$')

# c
def closed_form_l2(x_train, y_train, l, deg):
    return np.dot(inv(np.dot(x_train.T, x_train) + l * np.identity(deg + 1)), np.dot(x_train.T, y_train))

# ls = np.append(np.array([0]), np.logspace(-8, 0, 9))
ls = np.logspace(-17, 0, 18)
deg_l2 = 9

train_error_l2 = list()
test_error_l2 = list()

for l in ls:
    train_feature = generate_phi(Q1x_train, deg_l2)
    test_feature = generate_phi(Q1x_test, deg_l2)
    W = closed_form_l2(train_feature, y_train, l, deg_l2)
    train_dy = train_feature @ W - y_train
    train_rms = ((train_dy.T @ train_dy) / len(y_train)) ** 0.5
    test_dy = test_feature @ W - y_test
    test_rms = ((test_dy.T @ test_dy) / len(y_test)) ** 0.5
    train_error_l2.append(train_rms)
    test_error_l2.append(test_rms)

plt.plot(np.log(ls), train_error_l2, '-b', label='Training')
plt.plot(np.log(ls), test_error_l2, '-r', label='Test')
plt.legend()
_ = plt.xlabel(r'$\ln\lambda$')
_ = plt.ylabel(r'$E_{RMS}$')
