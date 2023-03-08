import numpy as np
import matplotlib.pyplot as plt

X = np.load('q1x.npy')
N = X.shape[0]
Y = np.load('q1y.npy')
# To consider intercept term, we append a column vector with all entries=1.
# Then the coefficient correpsonding to this column is an intercept term.
X = np.concatenate((np.ones((N, 1)), X), axis=1)

def sigmoid(x, w):
    return 1 / (1 + np.exp(-x @ w.T))

def hessian(x, w):
    s = np.identity(x.shape[0])
    for i in range(x.shape[0]):
        s[i, i] = sigmoid(x[i], w) * (1 - sigmoid(x[i], w))
    return x.T @ s @ x

def newton_LR(x, y, iter=2000):
    w = np.zeros((x.shape[1],))
    for i in range(iter):
        g = x.T @ (sigmoid(x, w) - y)
        h = hessian(x, w)
        w -= np.linalg.inv(h) @ g
    return w

q1_w = newton_LR(X, Y, iter=3000)
print(q1_w)

m = - q1_w[1] / q1_w[2]
c = - q1_w[0] / q1_w[2]
x_line = np.linspace(-4, 8)
plt.scatter(X[:, 1], X[:, 2], c=Y)
plt.plot(x_line, m * x_line + c, 'r--')
plt.title('Logistic Regression with Newton\'s Method')
