#from lightgbm import train
import numpy as np
import matplotlib.pyplot as plt
import time

def validate_PCA(states, train_data):
  from sklearn.decomposition import PCA
  pca = PCA()
  pca.fit(train_data)
  true_matrix = pca.components_.T
  true_ev = pca.explained_variance_

  
  output_matrix = states['transform_matrix']
  error = np.mean(np.abs(np.abs(true_matrix) - np.abs(output_matrix)) / np.abs(true_matrix))
  if error > 0.01:
    print('Matrix is wrong! Error=',error)
  else:
    print('Matrix is correct! Error=', error)

  output_ev = states['eigen_vals']
  error = np.mean(np.abs(true_ev - output_ev) / true_ev)
  if error > 0.01:
    print('Variance is wrong! Error=', error)
  else:
    print('Variance is correct! Error=', error)

def train_PCA(train_data):
  ##### TODO: Implement here!! #####
  # Note: do NOT use sklearn here!
  # Hint: np.linalg.eig() might be useful
  states = {
      'transform_matrix': np.identity(train_data.shape[-1]),
      'eigen_vals': np.ones(train_data.shape[-1])
  }
  ##### TODO: Implement here!! #####
  train_data_mean = train_data - np.mean(train_data, axis=0)
  cov = (1 / len(train_data_mean)) * train_data_mean.T @ train_data_mean
  evals, evcts = np.linalg.eig(cov)
  sorted_idx = np.argsort(evals)[::-1]
  sorted_evals = evals[sorted_idx]
  sorted_evcts = evcts[:, sorted_idx]
  states['eigen_vals'] = sorted_evals
  states['transform_matrix'] = sorted_evcts
  return states

# Load data
start = time.time()
images = np.load('q3data/q3.npy')
num_data = images.shape[0]
train_data = images.reshape(num_data, -1)

states = train_PCA(train_data)
print('training time = %.1f sec'%(time.time() - start))
validate_PCA(states, train_data)

evals = states['eigen_vals']
evcts = states['transform_matrix']
n_components = 10
print(evals[:n_components])
plt.plot(range(1, len(evals) + 1), evals)
plt.xlabel('i')
plt.ylabel(f'$\lambda_i$')
plt.show()

n_evcts = evcts[:, :n_components]
n_evals = evals[:n_components]
train_data_mean = np.mean(train_data, axis=0)

fig = plt.figure(figsize=[images.shape[1] / 2, images.shape[2] / 2])
for i in range(10):
  fig.add_subplot(2, 5, i + 1)
  if i == 0:
    plt.imshow(train_data_mean.reshape(images.shape[1], images.shape[2]))
    plt.title('Mean')
  else:
    plt.imshow(n_evcts[:, i-1].reshape(images.shape[1], images.shape[2]))
    plt.title(f'$\lambda_{i}$ = {n_evals[i-1]:.1e}')
  plt.axis('off')
plt.show()
cnt_pca = 0
while True:
  cnt_pca += 1
  if sum(evals[:cnt_pca]) / sum(evals) > 0.99:
    print(cnt_pca, 1 - cnt_pca / len(evals))
    break
