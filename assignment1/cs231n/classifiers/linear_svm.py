from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. 가중치 행렬(Weight matrix)
    - X: A numpy array of shape (N, D) containing a minibatch of data. 입력 데이터
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C. 레이블 벡터
    - reg: (float) regularization strength 정규화 계수

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # 클래스 수 C
    num_train = X.shape[0] # 데이터 수 N
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # 각 클래스에 대한 score 벡터 (1,C)
        correct_class_score = scores[y[i]] # 정답 클래스의 점수
        num_pos_margins = 0
        for j in range(num_classes):
            if j == y[i]: # 정답일때 패스
                continue
            margin = scores[j] - correct_class_score + 1  # delta = 1. 마진 계산, 수식은 svm.ipynb 참고
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                num_pos_margins += 1
        dW[:, y[i]] -= num_pos_margins * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X,W)
    N = X.shape[0]

    correct_scores = scores[np.arange(N),y] # 정답 클래스 추출
    correct_scores = correct_scores.reshape(-1,1)

    margins = np.maximum(0, scores-correct_scores + 1)
    margins[np.arange(N),y] = 0 # 정답 클래스 제외

    loss = np.sum(margins) / N
    loss += reg * np.sum(W*W)

    binary = margins > 0 # 마진이 양수인 곳만 분리해 마스크 만들기
    binary = binary.astype(int)

    row_sum = np.sum(binary, axis = 1) # 정답 클래스의 열에 대해 양수인 마진만큼 음수로 넣기
    binary[np.arange(N),y] = -row_sum


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = X.T.dot(binary) / N # 최종 gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
