import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  {conv - [batch_norm] - relu - conv - [batch_norm] - relu - 2x2 max pool}xN - {affine - [batch_norm] - relu}xM - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32, 64, 128],
               filter_sizes=[5, 3, 3, 3], hidden_dims=[100, 100], num_classes=10, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: A list of integers giving the number of filter for each conv layer.
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: A list of integers giving the size of each hidden layer.
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_conv = len(num_filters)
    self.num_layers = len(hidden_dims) + 1
    C, H, W = input_dim
    
    # initialize {conv - [batch_norm] - relu - conv - [batch_norm] - relu - 2x2 max pool} x N
    for i in range(self.num_conv):
        pad = (filter_sizes[i] - 1) / 2
        stride_conv = 1
        w_name, b_name = "W{}".format(i+1), "b{}".format(i+1)
        beta_name, gamma_name = "beta{}".format(i+1), "gamma{}".format(i+1)
        H_conv = 1 + (H + 2 * pad - filter_sizes[i]) / stride_conv
        W_conv = 1 + (W + 2 * pad - filter_sizes[i]) / stride_conv
    
        self.params[w_name] = weight_scale * np.random.randn(num_filters[i], C, filter_sizes[i], filter_sizes[i])
        self.params[b_name] = np.zeros((num_filters[i],))
        if (i + 1) % 2 == 0:
            pool_height, pool_width, stride_pool = 2, 2, 2 
            H = 1 + (H_conv - pool_height) / stride_pool
            W = 1 + (W_conv - pool_width) / stride_pool
        else:
            W = W_conv
            H = H_conv
            
        C = num_filters[i]
        self.params[beta_name] = np.zeros(C)
        self.params[gamma_name] = np.ones(C)
    
    # initialize {affine - [batch_norm] - relu} x M 
    D = num_filters[-1] * H * W
    for j in range(self.num_layers - 1):
        w_name, b_name = "W{}".format(i+j+2), "b{}".format(i+j+2)
        beta_name, gamma_name = "beta{}".format(i+j+2), "gamma{}".format(i+j+2)
        
        self.params[w_name] = weight_scale * np.random.randn(D, hidden_dims[j])
        self.params[b_name] = np.zeros((hidden_dims[j],))
        self.params[beta_name] = np.zeros(hidden_dims[j])
        self.params[gamma_name] = np.ones(hidden_dims[j])
        D = hidden_dims[j]
    
    w_name = "W{}".format(self.num_conv+self.num_layers)
    b_name = "b{}".format(self.num_conv+self.num_layers)
    self.params[w_name] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params[b_name] = np.zeros(num_classes)
    
    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_conv + self.num_layers - 1)]
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    mode = 'test' if y is None else 'train'   
    for bn_param in self.bn_params:
        bn_param[mode] = mode
    cache = [None for i in range(self.num_conv + self.num_layers)]

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    out = X
    for i in range(0, self.num_conv, 2):
        w1, b1 = self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)]
        w2, b2 = self.params['W{}'.format(i+2)], self.params['b{}'.format(i+2)]
        
        # pass beta & gamma param to the forward pass for the convolutional layer
        beta1, gamma1 = self.params['beta{}'.format(i+1)], self.params['gamma{}'.format(i+1)]
        beta2, gamma2 = self.params['beta{}'.format(i+2)], self.params['gamma{}'.format(i+2)]
        
        # pass conv_param to the forward pass for the convolutional layer
        conv_param1 = {'stride': 1, 'pad': (w1.shape[2] - 1) / 2}
        conv_param2 = {'stride': 1, 'pad': (w2.shape[2] - 1) / 2}
        
        out1, cache[i] = conv_batch_relu_forward(out, w1, b1, beta1, gamma1, conv_param1, self.bn_params[i])
        out, cache[i+1] = conv_batch_relu_pool_forward(out1, w2, b2, beta2, gamma2, conv_param2, self.bn_params[i+1], pool_param)
            
    for j in range(self.num_layers - 1):
        idx = i + j + 1
        w1, b1 = self.params['W{}'.format(idx+2)], self.params['b{}'.format(idx+2)]
        beta1, gamma1 = self.params['beta{}'.format(idx+2)], self.params['gamma{}'.format(idx+2)]
        out, cache[idx+1] = affine_batchnorm_relu_forward(out, w1, b1, beta1, gamma1, self.bn_params[idx+1])
        
    w1= self.params["W{}".format(self.num_conv+self.num_layers)]
    b1 = self.params["b{}".format(self.num_conv+self.num_layers)]
    scores, cache[-1] = affine_forward(out, w1, b1)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, grad = softmax_loss(scores, y)
    matrix_sum = 0
    total_layers = self.num_conv + self.num_layers
    
    w_name, b_name = 'W{}'.format(total_layers), 'b{}'.format(total_layers)
    dx, grads[w_name], grads[b_name] = affine_backward(grad, cache[-1])
    
    for i in range(total_layers-2, total_layers-1-self.num_layers, -1):
        w_name, b_name = 'W{}'.format(i+1), 'b{}'.format(i+1)
        gamma_name, beta_name = 'gamma{}'.format(i+1), 'beta{}'.format(i+1)
        dx, grads[w_name], grads[b_name], grads[gamma_name], grads[beta_name] = affine_batchnorm_relu_backward(dx, cache[i])
        grads[w_name] += self.reg * self.params[w_name]
        matrix_sum += np.sum(self.params[w_name] ** 2)

    for j in range(i-1, -1, -2):
        w1_name, b1_name = 'W{}'.format(j+1), 'b{}'.format(j+1)
        w2_name, b2_name = 'W{}'.format(j), 'b{}'.format(j)
        gamma_name1, beta_name1 = 'gamma{}'.format(j+1), 'beta{}'.format(j+1)
        gamma_name2, beta_name2 = 'gamma{}'.format(j), 'beta{}'.format(j)
        
        dx1, dw1, db1, dgamma1, dbeta1 = conv_batch_relu_pool_backward(dx, cache[j])
        dx, dw2, db2, dgamma2, dbeta2 = conv_batch_relu_backward(dx1, cache[j-1])
        
        grads[w1_name], grads[b1_name], grads[gamma_name1], grads[beta_name1] = dw1, db1, dgamma1, dbeta1
        grads[w2_name], grads[b2_name], grads[gamma_name2], grads[beta_name2] = dw2, db2, dgamma2, dbeta2
        
        grads[w1_name] += self.reg * self.params[w1_name]
        grads[w2_name] += self.reg * self.params[w2_name]
        matrix_sum += np.sum(self.params[w1_name] ** 2)
        matrix_sum += np.sum(self.params[w2_name] ** 2)
    
    loss += self.reg * 0.5 * matrix_sum
    
    return loss, grads
  



class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    pad = (filter_size - 1) / 2
    stride_conv = 1
    H_conv = 1 + (H + 2 * pad - filter_size) / stride_conv
    W_conv = 1 + (W + 2 * pad - filter_size) / stride_conv
    
    pool_height, pool_width, stride_pool = 2, 2, 2 
    H_pool = 1 + (H_conv - pool_height) / stride_pool
    W_pool = 1 + (W_conv - pool_width) / stride_pool
    
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros((num_filters,))
    
    self.params["W2"] = weight_scale * np.random.randn(num_filters * H_pool * W_pool, hidden_dim)
    self.params["b2"] = np.zeros((hidden_dim,))
    
    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b3"] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax            #
    ############################################################################
    out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out_aff, cache_aff = affine_relu_forward(out_conv, W2, b2)
    scores, cache_out = affine_forward(out_aff, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, grad = softmax_loss(scores, y)
    loss += self.reg * 0.5 * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    
    d_aff_x, grads["W3"], grads["b3"] = affine_backward(grad, cache_out)
    d_rel_aff_x, grads["W2"], grads["b2"] = affine_relu_backward(d_aff_x, cache_aff)
    dx, grads["W1"], grads["b1"] = conv_relu_pool_backward(d_rel_aff_x, cache_conv)
    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
