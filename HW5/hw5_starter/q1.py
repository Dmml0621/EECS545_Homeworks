import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import pairwise_distances
# from torch import zeros_like  # Don't use other functions in sklearn

def train_kmeans(train_data, initial_centroids):
  ##### TODO: Implement here!! #####
  # Hint: pairwise_distances() might be useful
  states = {
      'centroids': initial_centroids
  }
  ##### TODO: Implement here!! #####
  for i in range(50):
    d = pairwise_distances(train_data, states['centroids'])
    r = np.zeros_like(d)
    r[range(len(d)), d.argmin(axis=1)] = 1
    states['r'] = r
    dot_sum = np.dot(train_data.T, r)
    cnt = np.sum(r[:, np.newaxis], axis=0)
    states['centroids'] = (dot_sum / cnt).T
  return states

def test_kmeans(states, test_data):
  result = {}
  ##### TODO: Implement here!! #####
  compressed_data = np.zeros_like(test_data)
  d = pairwise_distances(test_data, states['centroids'])
  compressed_data[range(len(compressed_data))] = states['centroids'][d.argmin(axis=1)]
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

# K-means
num_centroid = 16
initial_centroid_indices = [16041, 15086, 15419,  3018,  5894,  6755, 15296, 11460, 
                            10117, 11603, 11095,  6257, 16220, 10027, 11401, 13404]
initial_centroids = train_data[initial_centroid_indices, :]
states = train_kmeans(train_data, initial_centroids)
result_kmeans = test_kmeans(states, test_data)
print('Kmeans result=', result_kmeans)
