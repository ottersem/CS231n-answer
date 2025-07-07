from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # 데이터 수 N
    num_classes = W.shape[1] # 클래스 수 C

    for i in range(num_train):
      scores = np.dot(X[i], W) # 각 클래스에 대한 score 벡터 (1,C)

      scores -= max(scores) # 정규화(수치 안정화)

      exp_scores = np.exp(scores)
      probability = exp_scores / sum(exp_scores) # 소프트맥스 확률 계산

      correct = y[i]
      loss += -np.log(probability[correct]) # 정답 클래스의 softmax확률에 로그를 취해 크로스 엔트로피 로스 계산

      for j in range(num_classes): # 클래스 j에 대한 gradient 벡터 (D,)
        if j == correct:
          dW[:,j] += (probability[j] - 1) * X[i]
        else:
          dW[:,j] += probability[j] * X[i]

    # Loss 갱신
    loss /= num_train
    loss += reg * np.sum(W*W)

    # gradient 갱신
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X,W) # (N,C)
    scores -= np.max(scores, axis=1, keepdims=True)

    num_train = scores.shape[0]

    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=1) # (N,C)

    correct_class_probs = probs[np.arange(num_train), y] # (N,)
    loss = -np.sum(np.log(correct_class_probs))/num_train
    loss += reg * np.sum(W * W)

    probs[np.arange(num_train), y] -= 1 # dL/dz : (N,C)
    dW = np.dot(np.transpose(X), probs) # (D,C)
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
