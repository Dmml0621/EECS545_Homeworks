import numpy as np
from scipy.io.wavfile import write
Fs = 11025
    
def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('q5data/q5.dat')
    return mix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    for alpha in anneal:
      # Hint: you might want to use this alpha as learning rate for fast convergence
      # But feel free to use different learning rate. Training should be done in 3 minutes
        print(f'working on alpha = {alpha}')
        for i in range(M):
            temp = 1 - 2 * sigmoid(W @ X[i].T)
            W += alpha * (np.linalg.inv(W.T) + np.outer(temp, X[i]))
    
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)
    ######### Your code here ##########
    S = X @ W.T
    ##################################
    return S

X = normalize(load_data())
print(X.shape)
print('Saving mixed track 1')
write('q5_mixed_track_1.wav', Fs, X[:, 0])

import time
t0 = time.time()
W = unmixer(X) # This will take around 2min
print('time=', time.time()-t0)
S = normalize(unmix(X, W))

for track in range(5):
    print(f'Saving unmixed track {track}')
    write(f'q5_unmixed_track_{track}.wav', Fs, S[:, track])

print('W solution:')
print(W)
