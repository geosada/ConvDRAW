#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

eps = 1e-8 # epsilon for numerical stability


class Layers(object):

    def __init__(self):
        self.do_share = False

    def set_do_share(self, flag):
        self.do_share = flag

    def W( self, W_shape,  W_name='W', W_init=None):
        if W_init is None:
            W_initializer = tf.contrib.layers.xavier_initializer()
        else:
            W_initializer = tf.constant_initializer(W_init)

        return tf.get_variable(W_name, W_shape, initializer=W_initializer)

    def Wb( self, W_shape, b_shape, W_name='W', b_name='b', W_init=None, b_init=0.1):

        W = self.W(W_shape, W_name=W_name, W_init=None)
        b = tf.get_variable(b_name, b_shape, initializer=tf.constant_initializer(b_init))

        def _summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
        _summaries(W)
        _summaries(b)

        return W, b


    def denseV2( self, scope, x, output_dim, activation=None):
        return tf.contrib.layers.fully_connected( x, output_dim, activation_fn=activation, reuse=self.do_share, scope=scope)

    def dense( self, scope, x, output_dim, activation=None):
        if len(x.get_shape()) == 2:   # 1d
            pass
        elif len(x.get_shape()) == 4: # cnn as NHWC
            #x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten
            x = tf.reshape(x, [x.get_shape().as_list()[0], -1]) # flatten
            #x = tf.reshape(x, [tf.cast(x.get_shape()[0], tf.int32), -1]) # flatten
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb([x.get_shape()[1], output_dim], [output_dim])
        #with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb([x.get_shape()[1], output_dim], [output_dim])
        o = tf.matmul(x, W) + b 
        return o if activation is None else activation(o)
    
    def lrelu(self, x, a=0.1):
        if a < 1e-16:
            return tf.nn.relu(x)
        else:
            return tf.maximum(x, a * x)

    def avg_pool(self, x, ksize=2, stride=2):
        return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    def max_pool(self, x, ksize=2, stride=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    def conv2d( self, scope, x, out_c, filter_size=(3,3), strides=(1,1,1,1), padding="SAME", activation=None):
        """
        x:       [BATCH_SIZE, in_height, in_width, in_channels]
        filter : [filter_height, filter_width, in_channels, out_channels]
        """
        filter = [filter_size[0], filter_size[1], int(x.get_shape()[3]), out_c]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [out_c])
        o = tf.nn.conv2d(x, W, strides, padding) + b
        return o if activation is None else activation(o)


    def deconv2d( self, scope, x, out_c, output_size=None, filter_size=(4,4), strides=(1,2,2,1), padding="SAME", activation=None):
        """
        x:  [BATCH_SIZE, in_height, in_width, in_channels]
            output shape is [BATCH_SIZE, in_height*2, in_width*2, out_c] as the way of DCGAN.
            default values are aslo set as DCGAN https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py """
        _b, in_h, in_w, in_c = map( lambda _x: int(_x), x.get_shape())
        filter = [filter_size[0], filter_size[1], out_c, in_c]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [out_c])

        if output_size is None:
            output_shape = ( _b, in_h*2, in_w*2, out_c )
        else:
            print('output_size:', output_size )
            output_shape = ( _b, output_size[0], output_size[1], out_c )
            
        o = tf.nn.conv2d_transpose( x, W, output_shape=output_shape, strides=strides) + b
        return o if activation is None else activation(o)
            

    def highway( self, scope, x, size=None, activation=tf.nn.elu, carry_bias=-2.0):

        """
        x: (B, dim_x)
        """
        if size is None: size = x.get_shape()[1]
        with tf.variable_scope(scope,reuse=self.do_share):
            W, b = self.Wb([size, size], [size], W_init=0.1, b_init=carry_bias)
        with tf.variable_scope(scope+'hw_trans_wb', reuse=self.do_share):
            W_T, b_T = self.Wb([size, size], [size], W_init=0.1, b_init=carry_bias)

        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
        H = activation(tf.matmul(x, W) + b, name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")

        o = tf.add(tf.multiply(H, T), tf.multiply(x, C), 'y') # y = (H * T) + (x * C)
        return o

    ###########################################
    """             Softmax                 """
    ###########################################
    def softmax( self, scope, input, size):
        if input.get_shape()[1] != size:
            print("softmax w/ fc:", input.get_shape()[1], '->', size)
            return self.dense(scope, input, size, tf.nn.softmax)
        else:
            print("softmax w/o fc")
            return tf.nn.softmax(input)
    
    ## SAMPLER (VARIATIONAL AUTOENCODER) ##
    
    ###########################################
    """             Split                   """
    ###########################################
    # https://github.com/openai/iaf/blob/master/tf_utils/
    def split(self, x, split_dim, split_sizes):
        #   split_dim:   output dimension, e.g. 1
        #   split_sizes: list of output's elements length, e.g. [30, 30] for mu and siguma to make 30 dim z
        n = len(list(x.get_shape()))
        assert int(x.get_shape()[split_dim]) == np.sum(split_sizes)
        ids = np.cumsum([0] + split_sizes)
        ids[-1] = -1
        begin_ids = ids[:-1]

        ret = []
        for i in range(len(split_sizes)):
            cur_begin = np.zeros([n], dtype=np.int32)
            cur_begin[split_dim] = begin_ids[i]
            cur_end = np.zeros([n], dtype=np.int32) - 1
            cur_end[split_dim] = split_sizes[i]
            ret += [tf.slice(x, cur_begin, cur_end)]
        return ret 

    ###########################################
    """      Rparameterization Tricks       """
    ###########################################
    def epsilon( self, _shape, _stddev=1.):
        return tf.truncated_normal(_shape, mean=0, stddev=_stddev)

    def sampler( self, mu, sigma):
        """
        mu,sigma : (BATCH_SIZE, z_size)
        """
        return mu + sigma*self.epsilon( tf.shape(mu) )
        #return mu + sigma*self.epsilon( tf.shape(mu)[0], tf.shape(mu)[1] )
        
    def vae_sampler( self, scope, x, size, activation=tf.nn.elu):
        # for LVAE
        with tf.variable_scope(scope,reuse=self.do_share): 
            mu       = self.dense(scope+'_vae_mu', x, size)
            logsigma = self.dense(scope+'_vae_logsigma', x, size, activation)
            logsigma = tf.clip_by_value(logsigma, eps, 50)
        sigma = tf.exp(logsigma)
        return self.sampler(mu, sigma), mu, logsigma 

    def vae_sampler_w_feature_slice( self, x, size):
        mu, logsigma = self.split( x, 1, [size]*2)
        logsigma = tf.clip_by_value(logsigma, eps, 50)
        sigma = tf.exp(logsigma)
        return self.sampler(mu, sigma), mu, logsigma 

    def precision_weighted( self, musigma1, musigma2):
        mu1, sigma1 = musigma1
        mu2, sigma2 = musigma2
        sigma1__2 = 1 / tf.square(sigma1)
        sigma2__2 = 1 / tf.square(sigma2)
        mu = ( mu1*sigma1__2 + mu2*sigma2__2 )/(sigma1__2 + sigma2__2)
        sigma = 1 / (sigma1__2 + sigma2__2)
        logsigma = tf.log(sigma + eps)
        return (mu, logsigma, sigma)
    
    def precision_weighted_sampler( self, scope, musigma1, musigma2):
        # assume input Tensors are (BATCH_SIZE, dime)
        mu1, sigma1 = musigma1
        mu2, sigma2 = musigma2
        size_1 = mu1.get_shape().as_list()[1]
        size_2 = mu2.get_shape().as_list()[1]

        if size_1 > size_2:
            print('convert 1d to 1d:', size_2, '->', size_1)
            with tf.variable_scope(scope,reuse=self.do_share): 
                mu2       = self.dense(scope+'_lvae_mu', mu2, size_1)
                sigma2 = self.dense(scope+'_lvae_logsigma', sigma2, size_1)
                musigma2  = (mu2, sigma2)
        elif size_1 < size_2:
            raise ValueError("musigma1 must be equal or bigger than musigma2.")
        else:
            # not need to convert
            pass

        mu, logsigma, sigma = self.precision_weighted( musigma1, musigma2)
        return (mu + sigma*self.epsilon(tf.shape(mu) ), mu, logsigma)
    

    def bn(self, scope, x, is_train=True, do_update_bn=True, collections=None, name="bn", decay=0.999):
    
        if len(x.get_shape()) == 2:   # fc
            size = x.get_shape().as_list()[1]
            axes = [0]
        elif len(x.get_shape()) == 4: # cnn as NHWC
            size = x.get_shape().as_list()[3]
            axes = [0,1,2]
        #params_shape = (dim,)
        n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
        axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
        mean = tf.reduce_mean(x, axis)
        var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
        avg_mean = tf.get_variable(
            name=name + "_mean",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(0.0),
            collections=collections,
            trainable=False
        )

        avg_var = tf.get_variable(
            name=name + "_var",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(1.0),
            collections=collections,
            trainable=False
        )

        gamma = tf.get_variable(
            name=name + "_gamma",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(1.0),
            collections=collections
        )

        beta = tf.get_variable(
            name=name + "_beta",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(0.0),
            collections=collections,
        )

        if is_train:
            avg_mean_assign_op = tf.no_op()
            avg_var_assign_op = tf.no_op()
            if do_update_bn:
                avg_mean_assign_op = tf.assign(
                    avg_mean,
                    decay * avg_mean + (1 - decay) * mean)
                avg_var_assign_op = tf.assign(
                    avg_var,
                    decay * avg_var + (n / (n - 1)) * (1 - decay) * var)

            with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
                z = (x - mean) / tf.sqrt(1e-6 + var)
        else:
            z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

        return gamma * z + beta


