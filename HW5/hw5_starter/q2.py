#from lightgbm import train
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal  # Don't use other functions in scipy

def train_gmm(train_data, init_pi, init_mu, init_sigma):
  ##### TODO: Implement here!! #####
  # Hint: multivariate_normal() might be useful
  states = {
      'pi': init_pi,
      'mu': init_mu,
      'sigma': init_sigma,
  }
  ##### TODO: Implement here!! #####
  k = len(init_pi)
  n = len(train_data)
  w = np.zeros((n, k))
  p = lambda mu, sig: multivariate_normal.pdf(train_data, mean=mu, cov=sig)

  for i in range(50):
    # E
    for j in range(k):
      w[:, j] = states['pi'][j] * p(states['mu'][j], states['sigma'][j])
    ll = np.sum(np.log(np.sum(w, axis=1)))
    print('ll:', ll)
    w = (w.T / w.sum(axis=1)).T
    w_sum = np.sum(w, axis=0)
    # M
    for j in range(k):
      states['pi'][j] = w_sum[j] / n
      states['mu'][j] = (1 / w_sum[j]) * np.sum(w[:, j] * train_data.T, axis=1).T
      states['sigma'][j] = (1 / w_sum[j]) * (w[:, j] * (train_data - states['mu'][j]).T @ (train_data - states['mu'][j]))

  return states

def test_gmm(states, test_data):
  result = {}
  ##### TODO: Implement here!! #####
  compressed_data = np.zeros_like(test_data)
  k = len(states['pi'])
  n = len(test_data)
  w = np.zeros((n, k))
  p = lambda mu, sig: multivariate_normal.pdf(test_data, mean=mu, cov=sig)
  for i in range(k):
    w[:, i] = states['pi'][i] * p(states['mu'][i], states['sigma'][i])
  compressed_data[range(n)] = states['mu'][w.argmax(axis=1)]
  import matplotlib.pyplot as plt 
  plt.imshow(compressed_data.reshape(512, 512, 3) / 256)
  plt.show()
  ##### TODO: Implement here!! #####
  result['pixel-error'] = calculate_error(test_data, compressed_data)
  return result

### DO NOT CHANGE ###
def calculate_error(data, compressed_data):
  assert data.shape == compressed_data.shape
  error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
  return error
### DO NOT CHANGE ###

# Load data
img_small = np.array(imageio.imread('q12data/mandrill-small.tiff')) # 128 x 128 x 3
img_large = np.array(imageio.imread('q12data/mandrill-large.tiff')) # 512 x 512 x 3

ndim = img_small.shape[-1]
train_data = img_small.reshape(-1, ndim).astype(float)
test_data = img_large.reshape(-1, ndim).astype(float)

# GMM
num_centroid = 5
initial_mu_indices = [16041, 15086, 15419,  3018,  5894]
init_pi = np.ones((num_centroid, 1)) / num_centroid
init_mu = train_data[initial_mu_indices, :]
init_sigma = np.tile(np.identity(ndim), [num_centroid, 1, 1])*1000.

states = train_gmm(train_data, init_pi, init_mu, init_sigma)
result_gmm = test_gmm(states, test_data)
print(states)
print('GMM result=', result_gmm)
