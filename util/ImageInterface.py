#!/usr/bin/env python3

import tensorflow as tf
from layers import Layers
from subpixel import PS

eps = 1e-8

class ImageInterface(object):

    def __init__(self, is_3d, is_read_attention, is_write_attention, read_n, write_n, h, w, c):
    
        """ to manage do_share flag inside Layers object, ImageInterface has Layers as its own property """
        self.do_share = False
        self.ls       = Layers()
        self.is_3d    = is_3d
        self.read_n   = read_n
        self.write_n  = write_n
        self.h = h
        self.w = w
        self.c = c

        if is_read_attention:
            self.read = self._read_attention
        else:
            self.read = self._read_no_attention
    
        if is_write_attention:
            self.write = self._write_attention
        else:
            self.write = self._write_no_attention
     
    def set_do_share(self, flag):
        self.do_share    = flag
        self.ls.set_do_share(flag)
        
    ###########################
    """       READER        """
    ###########################
    def _read_no_attention(self, x,x_hat, h_dec):
        _h,_w,_c = self.h, self.w, self.c
        if self.is_3d:
            # x is a raw image and x_hat is an error one, and eash is handled as a different channel,
            # so the shape of r and return are [-1, _h,_w,_c*2]

            USE_CONV_READ = False # 170720
            if USE_CONV_READ:
                scope = 'read_1'
                x = self.ls.conv2d(scope+'_1', x, 64, activation=tf.nn.elu)
                x = self.ls.max_pool(x)
                x = self.ls.conv2d(scope+'_2', x, 64, activation=tf.nn.elu)
                x = self.ls.max_pool(x)
                x = self.ls.conv2d(scope+'_3', x, 64, activation=tf.nn.elu)

                scope = 'read_hat_1'
                x_hat = self.ls.conv2d(scope+'_1', x_hat, 16, activation=tf.nn.elu)
                x_hat = self.ls.max_pool(x_hat)
                x_hat = self.ls.conv2d(scope+'_2', x_hat, 16, activation=tf.nn.elu)
                x_hat = self.ls.max_pool(x_hat)
                x_hat = self.ls.conv2d(scope+'_3', x_hat, 16, activation=tf.nn.elu)

                r = tf.concat([x,x_hat], 3)
                h_dec = tf.reshape( self.ls.dense(scope, h_dec, _h*_w*_c), [-1, int(_h/4), int(_w/4),_c*4*4])
                return tf.concat([r,h_dec], 3)
            elif False:

                scope = 'read_1'
                x = self.ls.conv2d(scope+'_1', x, 128, activation=tf.nn.elu)
                x = self.ls.conv2d(scope+'_2', x, 128, activation=tf.nn.elu)
                x = self.ls.conv2d(scope+'_3', x, 128, activation=tf.nn.elu)
                x = self.ls.max_pool(x)
                scope = 'read_2'
                x = self.ls.conv2d(scope+'_1', x, 256, activation=tf.nn.elu)
                x = self.ls.conv2d(scope+'_2', x, 256, activation=tf.nn.elu)
                x = self.ls.conv2d(scope+'_3', x, 256, activation=tf.nn.elu)
                x = self.ls.max_pool(x)
                scope = 'read_3'
                x = self.ls.conv2d(scope+'_1', x, 512, activation=tf.nn.elu)
                x = self.ls.conv2d(scope+'_2', x, 256, activation=tf.nn.elu, filter_size=(1,1))
                x = self.ls.conv2d(scope+'_3', x, 128, activation=tf.nn.elu, filter_size=(1,1))
                x = self.ls.conv2d(scope+'_4', x, 64, activation=tf.nn.elu, filter_size=(1,1))

                scope = 'read_hat_1'
                x_hat = self.ls.conv2d(scope+'_1', x_hat, 128, activation=tf.nn.elu)
                x_hat = self.ls.max_pool(x_hat)
                scope = 'read_hat_2'
                x_hat = self.ls.conv2d(scope+'_1', x_hat, 256, activation=tf.nn.elu)
                x_hat = self.ls.max_pool(x_hat)
                scope = 'read_hat_3'
                x_hat = self.ls.conv2d(scope+'_4', x_hat, 16, activation=tf.nn.elu, filter_size=(1,1))
                r = tf.concat([x,x_hat], 3)
                h_dec = tf.reshape( self.ls.dense(scope, h_dec, _h*_w*_c), [-1, int(_h/4), int(_w/4),_c*4*4])
                return tf.concat([r,h_dec], 3)
            else:
                r = tf.concat([x,x_hat], 3)
            USE_DEC_LOWEST_PREV = True
            if USE_DEC_LOWEST_PREV:
                # use decoder feedback as element-wise adding   
                # Eq.(21) in [Gregor, 2016]
                scope = 'read'
                USE_CONV = True
                if USE_CONV:
                    h_dec = tf.reshape( self.ls.dense(scope, h_dec, _h*_w*_c), [-1, _h,_w,_c])
                    h_dec = self.ls.conv2d("conv", h_dec, _c*2, activation=tf.nn.elu)
                    return r + h_dec
                else:
                    h_dec = tf.reshape( self.ls.dense(scope, h_dec, _h*_w*_c*2), [-1, _h,_w,_c*2])
                    return r + h_dec
            else:
                return r
        else:
            return tf.concat([x,x_hat], 1)
    
    def _read_attention( self, x, x_hat, h_dec ):
        _h,_w,_c = self.h, self.w, self.c
        N = self.read_n
        if self.is_3d:
            Fx,Fy,gamma = self._set_window("read", h_dec,N)
            # Fx is (?, 5, 32, 3)
            # gamma is (?, 3)
            def filter_img(img,Fx,Fy,gamma, N):
                # Fx and Fy are (?, 5, 32, 3)
                Fxt = tf.transpose(Fx,perm=[0,3,2,1])
                Fy  = tf.transpose(Fy,perm=[0,3,2,1])
                
                # img.get_shape() has already been (?, 32, 32, 3)
                img  = tf.transpose(img, perm=[0,3,2,1])
                # tf.matmul(img,Fxt) is (?, 3, 32, 5)
                img_Fxt = tf.matmul(img,Fxt)
                img_Fxt = tf.transpose(img_Fxt, perm=[0,1,3,2])
                # Fy: (?, 3, 32, 5)
                Fy  = tf.transpose(Fy,perm=[0,1,3,2])
                glimpse = tf.matmul(Fy, img_Fxt, transpose_b=True)
                # glimpse.get_shape() is (?, 3, 32, 32)
                glimpse = tf.transpose(glimpse, perm=[0,2,3,1])
                glimpse = tf.reshape(glimpse,[-1,N*N, _c])
    
                glimpse = tf.transpose(glimpse, perm=[0,2,1])
                gamma   = tf.reshape(gamma,[-1,1, _c])
                gamma   = tf.transpose(gamma,   perm=[0,2,1])
                o = glimpse*gamma
                o = tf.transpose(o, perm=[0,2,1])
                return o
            x = filter_img( x, Fx, Fy, gamma, N) # batch x (read_n*read_n)
            x_hat = filter_img( x_hat, Fx, Fy, gamma, N)
            x = tf.reshape(x, [-1, N,N,_c])
            x_hat = tf.reshape(x_hat, [-1, N,N,_c])
            return tf.concat([x,x_hat], 3)
        else:
            Fx,Fy,gamma = self._set_window("read", h_dec,N)
            # Fx: (?, 5, 32), gamma: (?, 1)
            def filter_img(img,Fx,Fy,gamma,N):
                #print('filter_img in is_image == False')
                Fxt = tf.transpose(Fx,perm=[0,2,1])
                img = tf.reshape(img,[-1,_w,_h])
                # Fxt : (?, 32, 5)
                # img : (?, 32, 32)
                glimpse = tf.matmul(Fy,tf.matmul(img,Fxt))
                glimpse = tf.reshape(glimpse,[-1,N*N])
                return glimpse*tf.reshape(gamma,[-1,1])
            x = filter_img( x, Fx, Fy, gamma, N) # batch x (read_n*read_n)
            x_hat = filter_img( x_hat, Fx, Fy, gamma, N)
            return tf.concat([x,x_hat], 1) # concat along feature axis
    
    
    ###########################
    """       WRITER        """
    ###########################
    def _write_no_attention(self, h):
        scope = "write"
        _h,_w,_c = self.h, self.w, self.c
        if self.is_3d:
            IS_SIMPLE_WRITE = True
            if IS_SIMPLE_WRITE :
                print('IS_SIMPLE_WRITE:', IS_SIMPLE_WRITE)  
                return tf.reshape( self.ls.dense(scope, h, _h*_w*_c, tf.nn.elu), [-1, _h, _w, _c])
            else:
                IS_CONV_LSTM = True
                if IS_CONV_LSTM :
                    raise NotImplementedError
                else:
                    activation = tf.nn.elu
                    print('h in write:', h) # h.shape is (_b, RNN_SIZES[0])
                    L = 1
                    h = tf.reshape( h, (-1, 2,2,64*3)) # should match to RNN_SIZES[0]
                    h = self.ls.deconv2d(scope+'_1', h, 64*2) # 4
                    h = activation(h)
                    L = 2
                    h = self.ls.deconv2d(scope+'_2', h, 16*3) # 8
                    h = activation(h)
                    h = PS(h, 4, color=True)
                    print('h in write:', h)
                return tf.reshape( h, [-1, _h, _w, _c])
        else:
            return self.ls.dense( scope,h, _h*_w*_c )
    
    def _write_attention(self, h_dec):
        scope = "writeW"
        N          = self.write_n
        write_size = N*N
        _h,_w,_c = self.h, self.w, self.c
        Fx, Fy, gamma = self._set_window("write", h_dec, N)
        if self.is_3d:
            # Fx and Fy are (?, 5, 32, 3), gamma is (?, 3)
            w = self.ls.dense( scope, h_dec, write_size*_c) # batch x (write_n*write_n) [ToDo] replace self.ls.dense with deconv
            w = tf.reshape(w,[tf.shape(h_dec)[0],N,N,_c])
            w = tf.transpose(w, perm=[0,3,1,2])
            Fyt = tf.transpose(Fx,perm=[0,3,2,1])
            Fx  = tf.transpose(Fx, perm=[0,3,1,2])
    
            w_Fx = tf.matmul(w, Fx)
            # w_Fx.get_shape() is (?, 3, 5, 32)
            w_Fx = tf.transpose(w_Fx, perm=[0,1,3,2])
    
            wr = tf.matmul(Fyt, w_Fx, transpose_b=True)
            wr = tf.reshape(wr,[tf.shape(h_dec)[0],_w*_h, _c])
            wr = tf.transpose(wr, perm=[0,2,1])
            inv_gamma   = tf.reshape(1.0/gamma,[-1,1, _c])
            inv_gamma   = tf.transpose(inv_gamma, perm=[0,2,1])
            o = wr*inv_gamma
            o = tf.transpose(o, perm=[0,2,1])
            o = tf.reshape(o, [tf.shape(h_dec)[0], _w, _h, _c])
            return o
        else:
            w = self.ls.dense( scope, h_dec,write_size) # batch x (write_n*write_n)
            w = tf.reshape(w,[tf.shape(h_dec)[0],N,N])
            Fyt = tf.transpose(Fy,perm=[0,2,1])
            wr = tf.matmul(Fyt,tf.matmul(w,Fx))
            wr = tf.reshape(wr,[tf.shape(h_dec)[0],_w*_h])
            return wr*tf.reshape(1.0/gamma,[-1,1])

    ###########################
    """  Filter Functions   """
    ###########################
    def _filterbank(self, gx, gy, sigma2,delta, N):
        if self.is_3d:

            _h,_w,_c = self.h, self.w, self.c

            # gx and delta are (?,3)
            grid_i = tf.reshape(tf.cast(tf.range(N*_c), tf.float32), [1, -1, _c])
            mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
            mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
            # shape : [1, N, _c]
            w = tf.reshape( tf.cast( tf.range(_w*_c), tf.float32), [1, 1, -1, _c])
            h = tf.reshape( tf.cast( tf.range(_h*_c), tf.float32), [1, 1, -1, _c])
            mu_x = tf.reshape(mu_x, [-1, N, 1, _c])
            mu_y = tf.reshape(mu_y, [-1, N, 1, _c])
            sigma2 = tf.reshape(sigma2, [-1, 1, 1, _c])
            Fx = tf.exp(-tf.square((w - mu_x) / (2*sigma2))) # 2*sigma2?
            Fy = tf.exp(-tf.square((h - mu_y) / (2*sigma2))) # batch x N x B
            # normalize, sum over A and B dims
            Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
            Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
            return Fx,Fy
    
        else:
            grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
            # gx, delta and mu_x are (?, 1), and grid_i is (1, 5))
            mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
            mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
            h = tf.reshape(tf.cast(tf.range(_h), tf.float32), [1, 1, -1])
            w = tf.reshape(tf.cast(tf.range(_w), tf.float32), [1, 1, -1])
            mu_x = tf.reshape(mu_x, [-1, N, 1])
            mu_y = tf.reshape(mu_y, [-1, N, 1])
            sigma2 = tf.reshape(sigma2, [-1, 1, 1])
            Fx = tf.exp(-tf.square((w - mu_x) / (2*sigma2))) # 2*sigma2?
            Fy = tf.exp(-tf.square((h - mu_y) / (2*sigma2))) # batch x N x B
            # normalize, sum over A and B dims
            Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
            Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
            return Fx,Fy
    
    def _set_window(self, scope, h_dec,N):
        if self.is_3d:
            _h,_w,_c = self.h, self.w, self.c
            # get five (BATCH_SIZE, _c) matrixes
            gx_, gy_, log_sigma2, log_delta, log_gamma = self.ls.split( self.ls.dense(scope, h_dec, _c*5), 1, [_c]*5)
            gx_ = tf.reshape(gx_, [-1,1,_c])
            gy_ = tf.reshape(gy_, [-1,1,_c])
            log_sigma2 = tf.reshape(log_sigma2, [-1,1,_c])
            log_delta = tf.reshape(log_delta, [-1,1,_c])
            log_gamma = tf.reshape(log_gamma, [-1,1,_c])
            gx = (_w + 1)/2*(gx_+1)
            gy = (_h + 1)/2*(gy_+1)
            sigma2 = tf.exp(log_sigma2)
            delta = ( max(_h, _w) -1 ) / ( N -1 ) * tf.exp( log_delta ) # batch x N
            return self._filterbank( gx, gy, sigma2, delta, N) + ( tf.exp(log_gamma),)
        else:
            params = self.ls.dense(scope, h_dec,5)
            gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(value=params, num_or_size_splits=5, axis=1)
            gx=(_w + 1)/2*(gx_+1)
            gy=(_h + 1)/2*(gy_+1)
            sigma2=tf.exp(log_sigma2)
            delta=(max(_h, _w)-1)/(N-1)*tf.exp(log_delta) # batch x N
            return self._filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)
