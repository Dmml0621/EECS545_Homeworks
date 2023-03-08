import numpy as np
# from tqdm import tqdm
from numpy.linalg import inv
import matplotlib.pyplot as plt

Q2x = np.load('q2x.npy')
Q2y = np.load('q2y.npy')

# a
N = len(Q2x)
x_train = np.append(Q2x.reshape(N, 1), np.ones((N, 1)), axis=1)
y_train = Q2y

def closed_form(x_train, y_train):
    # highest power first in result
    return np.dot(inv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))

W = closed_form(x_train, y_train)

x = x_train
plt.scatter(Q2x, Q2y, s=[5])
plt.plot(x, x * W[0] + W[1], '-r')
plt.xlabel(r'x')
plt.ylabel(r'y')

# b
def kernel(p, x, l):
    N = x.shape[0]
    W = np.mat(np.identity(N))
    for i in range(N):
        W[i, i] = np.exp(np.dot(p - x[i], p - x[i]) / (-2 * l * l))
    return W

def LWLR(x, y, l):
    N = x.shape[0]
    y_pred = np.zeros(N)
    for i in range(N):
        wt = kernel(x[i], x, l)
        W_local = inv(x.T @ wt @ x) * (x.T @ wt @ y.T).T
        y_pred[i] = np.dot(x[i], W_local)
    return y_pred

y_pred_lwlr = LWLR(x_train, y_train, 0.8)

plt.scatter(Q2x, Q2y, s=[5], label='Original')
plt.scatter(Q2x, y_pred_lwlr, s=[5], c='r', label='LWLR')
plt.legend()
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.title(r'$\lambda=0.8$')


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
ax1.scatter(Q2x, Q2y, s=[3], label='Original')
ax1.scatter(Q2x, LWLR(x_train, y_train, 0.1), s=[3], c='r', label='LWLR')
ax1.set_title(r'$\lambda=0.1$')
ax1.legend()
ax2.scatter(Q2x, Q2y, s=[3], label='Original')
ax2.scatter(Q2x, LWLR(x_train, y_train, 0.3), s=[3], c='r', label='LWLR')
ax2.set_title(r'$\lambda=0.3$')
ax2.legend()
ax3.scatter(Q2x, Q2y, s=[3], label='Original')
ax3.scatter(Q2x, LWLR(x_train, y_train, 2), s=[3], c='r', label='LWLR')
ax3.set_title(r'$\lambda=2$')
ax3.legend()
ax4.scatter(Q2x, Q2y, s=[3], label='Original')
ax4.scatter(Q2x, LWLR(x_train, y_train, 10), s=[3], c='r', label='LWLR')
ax4.set_title(r'$\lambda=10$')
ax4.legend()