import numpy as np
import matplotlib.pyplot as plt

q2_data = np.load('q2_data.npz')
q2x_train = q2_data['q2x_train']
q2y_train = q2_data['q2y_train']
q2x_test = q2_data['q2x_test']
q2y_test = q2_data['q2y_test']

def softmax_LR_train(x, y, lr=0.0005, iter=2000):
    np.random.seed(42)
    W = np.random.rand(max(y)[0].astype(int), x.shape[1])
    W[-1] = np.zeros((1, x.shape[1]))
    for j in range(iter):
        for w in range(len(W) - 1):
            g = np.zeros(x.shape[1])
            for i in range(x.shape[0]):
                temp = np.exp(W[w] @ x[i]) / (1 + sum(np.exp(W[:len(W) - 1, :] @ x[i])))
                if y[i] == w + 1:
                    g += (1 - temp) * x[i]
                else:
                    g += -temp * x[i]
            W[w] += lr * g
    return W

def softmax_LR_test(x, w):
    y_pred = np.zeros(x.shape[0])
    for i in range(len(x)):
        y_pred[i] = np.argmax(w @ x[i]) + 1
    return y_pred

def validate_LR(y_pred, y):
    correct = 0
    for i in range(len(y)):
        if y_pred.tolist()[i] == y[i]:
            correct += 1
    print('Error: {:2.4f}%'.format(100 * (1 - correct / len(y))))

q2_w = softmax_LR_train(q2x_train, q2y_train)
q2_ypred = softmax_LR_test(q2x_test, q2_w)
validate_LR(q2_ypred, q2y_test)
