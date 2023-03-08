from builtins import range
from traceback import walk_tb
import numpy as np
import math
import scipy.signal


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    out = x @ w + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dw = x.T @ dout
    db = np.sum(dout, axis=0)
    dx = dout @ w.T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.array(dout, copy=True)
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # given in Q1(c). Note that, as specified in the question, this function  #
    # should actually implement the filtering operation with w. This is just  #
    # equivalent to implementing convolution with a flipped filter, and this  #
    # can be compensated for in the gradient computation as you saw in the    #
    # derivation for Q1 (c).                                                  #
    #                                                                         #
    # Note: You are free to use scipy.signal.convolve2d, but we encourage you #
    # to implement the convolution operation by yourself using just numpy     #
    # operations.                                                             #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, Hp, Wp = w.shape
    HH, WW = H - Hp + 1, W - Wp + 1
    out = np.zeros((N, F, HH, WW))

    w_T = w[:, :, :, :, np.newaxis].transpose(4, 1, 2, 3, 0)
    for i in range(HH):
      for j in range(WW):
        out[:, :, i, j] = np.sum(x[:, :, i:i+Hp, j:j+Wp, np.newaxis] * w_T, axis=(1, 2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    x, w = cache
    ###########################################################################
    # TODO: Implement the convolutional backward pass as defined in Q1(c)     #
    #                                                                         #
    # Note: You are free to use scipy.signal.convolve2d, but we encourage you #
    # to implement the convolution operation by yourself using just numpy     #
    # operations.                                                             #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, Hp, Wp = w.shape
    HH, WW = H - Hp + 1, W - Wp + 1
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    dout_pad = np.pad(dout, ((0,), (0,), (Hp - 1,), (Wp - 1,)), mode='constant')
    w_flip = np.zeros_like(w)
    for i in range(Hp):
      for j in range(Wp):
        w_flip[:, :, i, j] = w[:, :, Hp - i - 1, Wp - j - 1]

    w_T = w_flip[:, :, :, :, np.newaxis].transpose(4, 1, 2, 3, 0)
    dout_pad_T = dout_pad[:, :, :, :, np.newaxis].transpose(0, 4, 2, 3, 1)
    for i in range(H):
      for j in range(W):
        dx[:, :, i, j] = np.sum(dout_pad_T[:, :, i:i+Hp, j:j+Wp, :] * w_T, axis=(2, 3, 4))
    dout_T = dout[:, :, :, :, np.newaxis].transpose(1, 4, 0, 2, 3)
    x_T = x[:, :, :, :, np.newaxis].transpose(4, 1, 0, 2, 3)
    for i in range(Hp):
      for j in range(Wp):
        dw[:, :, i, j] = np.sum(x_T[:, :, :, i:i+HH, j:j+WW] * dout_T, axis=(2, 3, 4))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here and we can assume that the dimension of
    input and stride will not cause problem here. Output size is given by
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    Hp, Wp = int(1 + (H - pool_height) / stride), int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, Hp, Wp))

    for n in range(N):
      for d in range(C):
        for i in range(Hp):
          for j in range(Wp):
            out[n, d, i, j] = np.max(x[n, d, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    N, C, H, W = x.shape
    N, C, HH, WW = dout.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    dx = np.zeros_like(x)

    for n in range(N):
      for d in range(C):
        for i in range(HH):
          for j in range(WW):
            x_pool = x[n, d, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            mask = (x_pool == np.max(x_pool))
            dx[n, d, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width] += mask * dout[n, d, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the j-th
      class for the i-th input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the cross-entropy loss averaged over N samples.
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Implement the softmax loss
    ###########################################################################
    N, C = x.shape
    loss = np.zeros(N)
    p = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    dx = p
    for i in range(N):
      loss[i] = -np.log(p[i, y[i]])
      dx[i, y[i]] -= 1
    loss = np.sum(loss) / N
    dx /= N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
