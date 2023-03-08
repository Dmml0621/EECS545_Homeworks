import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))
    return 100 * error

def nb_train(matrix, category):
    # Implement your algorithm and return 
    state = {}
    N, m = matrix.shape
    state['Prior'] = [1 - sum(category) / len(category), sum(category) / len(category)]
    
    c_word_spam, c_word_nonspam = [0] * m, [0] * m
    word_spam, word_nonspam = 0, 0
    for r in range(N):
        if category[r] == 1:
            word_spam += sum(matrix[r])
            for c in range(m):
                c_word_spam[c] += matrix[r, c]
        else:
            word_nonspam += sum(matrix[r])
            for c in range(m):
                c_word_nonspam[c] += matrix[r, c]
    
    # Laplace smoothing
    state['P(W|spam)'] = [(p + 1) / (word_spam + m) for p in c_word_spam]
    state['P(W|nonspam)'] = [(p + 1) / (word_nonspam + m) for p in c_word_nonspam]
    
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    p_spam = state['P(W|spam)']
    p_nonspam = state['P(W|nonspam)']
    
    for r in range(matrix.shape[0]):
        sum_spam, sum_nonspam = np.log(state['Prior'][1]), np.log(state['Prior'][0])
        for c in range(matrix.shape[1]):
            sum_spam += matrix[r, c] * np.log(p_spam[c])
            sum_nonspam += matrix[r, c] * np.log(p_nonspam[c])
        if sum_spam > sum_nonspam:
            output[r] = 1
        else:
            output[r] = 0
            
    return output

dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

# Train
state = nb_train(dataMatrix_train, category_train)

# Test and evluate
prediction = nb_test(dataMatrix_test, state)
evaluate(prediction, category_test)

p_spam = state['P(W|spam)']
p_nonspam = state['P(W|nonspam)']
p_ratio = [np.log(p_spam[i] / p_nonspam[i]) for i in range(len(p_spam))]
p_dict = dict(zip(tokenlist, p_ratio))
top_5 = list(dict(sorted(p_dict.items(), key=lambda item: item[1], reverse=True)).keys())[:5]

sizes = [50, 100, 200, 400, 800, 1400]
errors = list()
file_prefix = 'q4_data/MATRIX.TRAIN.'

for s in sizes:
    dataMatrix_train, tokenlist, category_train = readMatrix(f'{file_prefix + str(s)}')
    state = nb_train(dataMatrix_train, category_train)
    prediction = nb_test(dataMatrix_test, state)
    print(f'Training size: {s}')
    errors.append(evaluate(prediction, category_test))

plt.plot(sizes, errors, '-ro')
plt.xlabel('Training size')
plt.ylabel('Error (%)')
