# EECS 545 HW3 Q5

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

np.random.seed(545)

def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
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


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error


def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    # Train
    model = LinearSVC(max_iter=20000).fit(dataMatrix_train, category_train)

    # Test and evluate
    prediction = np.ones(dataMatrix_test.shape[0])  # TODO: This is a dummy prediction.
    evaluate(prediction, category_test)

    sizes = [50, 100, 200, 400, 800, 1400]
    errors = list()
    file_prefix = 'q5_data/MATRIX.TRAIN.'

    for s in sizes:
        dataMatrix_train, tokenlist, category_train = readMatrix(f'{file_prefix + str(s)}')
        model = LinearSVC().fit(dataMatrix_train, category_train)
        prediction = model.predict(dataMatrix_test)
        decision_function = model.decision_function(dataMatrix_train)
        support_vector_indices = np.where(np.abs(decision_function) <= 1)[0]
        support_vector_count = len(dataMatrix_train[support_vector_indices])

        print(f'Training size: {s}')
        print(f'Number of support vectors: {support_vector_count}')
        errors.append(evaluate(prediction, category_test))

    plt.plot(sizes, errors, '-ro')
    plt.xlabel('Training size')
    plt.ylabel('Error')


if __name__ == '__main__':
    main()
