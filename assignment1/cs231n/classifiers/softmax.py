import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_cases = X.shape[0]
  num_class = W.shape[1]
  y_label = np.zeros((num_cases,num_class))
  for i in range(num_cases):
    h1 = np.exp(X[i].dot(W))
    h = h1/np.sum(h1)
    y_label[i] = (np.arange(h.shape[0]) == y[i]) + 0
    loss -= (np.sum(y_label[i] * np.log(h) + (1 - y_label[i]) * np.log(1 - h)))
    delta = np.zeros(W.shape)
    for j in range(num_class):
      delta[:,j] += X[i]
      delta[:,j] *= h1[j]
      delta[:,j] *= (np.sum(h1) - h1[j])/(np.sum(h1) ** 2)
      delta[:,j] = y_label[i][j] / h[j] * delta[:,j] - (1 - y_label[i][j]) / (1 - h[j]) * delta[:,j]
    dW -= delta
  loss /= num_cases
  loss += reg * np.sum(W * W)
  dW /= num_cases
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  h1 = np.exp(X.dot(W))
  h = h1 / np.reshape(np.sum(h1,axis=1),(h1.shape[0],1))
  y_label = (np.arange(h.shape[1]) == y[:,None]) + 0
  loss = - np.mean(np.sum(y_label * np.log(h) + (1 - y_label) * np.log(1 - h),axis=1))
  H = h1 * ((np.reshape(np.sum(h1,axis=1),(h1.shape[0],1)) - h1) / (np.reshape(np.sum(h1,axis=1),(h1.shape[0],1)) ** 2))
  YH = y_label / h - (1 - y_label) / (1 - h)
  dW = X.T.dot(H * YH)
  dW = - dW / X.shape[0] + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

