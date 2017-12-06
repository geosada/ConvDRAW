import tensorflow as tf
import numpy as np
from layers import Layers
from losses import LossFunctions
from ImageInterface import ImageInterface

#np.set_printoptions(threshold=np.inf)

###########################################
""" Definition for Model Architecture """
###########################################

""" width/height size of attetioning grid """
T = 8
RNN_SIZES = [256*3, 128*3 ]
Z_SIZES   = [128*3,  128*3 ]
L = len(RNN_SIZES)

GLIMPSE_SIZE_READ  = 5
GLIMPSE_SIZE_WRITE = 5

""" dataset information """
_b,_h,_w,_c,_img_size,_is_3d = 0,0,0,0,0,False



class ConvDRAW(object):

    def __init__(self, d, lr, lambda_z_wu, read_attn, write_attn, do_classify, do_reconst):

        self.do_classify = do_classify 

        """ flags for each regularizor """
        self.do_reconst = do_reconst 
        self.read_attn  = read_attn
        self.write_attn = write_attn

        """ dataset information """
        self.set_datainfo(d)

        """ external toolkits """
        self.ls = Layers()
        self.lf = LossFunctions(self.ls, self.d, self.encoder)
        self.ii = ImageInterface( _is_3d, self.read_attn, self.write_attn, GLIMPSE_SIZE_READ, GLIMPSE_SIZE_WRITE, _h, _w, _c)
        # for refference from get_loss_kl_draw()
        self.T = T
        self.L = L
        self.Z_SIZES = Z_SIZES

        """ placeholders defined outside"""
        self.lr  = lr
        self.lambda_z_wu = lambda_z_wu

        """sequence of canvases """
        self.cs=[0]*T

        """ initialization """
        self.init_lstms()
        self.init_time_zero()

        """ workaround for variable_scope(reuse=True) """
        self.DO_SHARE=None 

    def set_datainfo(self, d):
        self.d  = d  # dataset manager
        global _b,_h,_w,_c,_img_size,_is_3d
        _b         = d.batch_size
        _h         = d.h
        _w         = d.w
        _c         = d.c
        _img_size  = d.img_size
        _is_3d     = d.is_3d

    def init_time_zero(self):

        self.cs[0] = tf.zeros((_b, _h,_w,_c)) if _is_3d else tf.zeros((_b, _img_size))
        self.h_dec[0][0] = tf.zeros((_b, RNN_SIZES[0]))

    def init_lstms(self):

        h_enc, e_mus, e_logsigmas = [[0]*L]*(T+1), [[0]*L]*(T+1), [[0]*L]*(T+1)     # q(z_i+1 | z_i), bottom-up inference
        h_dec, d_mus, d_logsigmas = [[0]*L]*(T+1), [[0]*L]*(T+1), [[0]*L]*(T+1)     # q(z_i | .), bidirectional inference
        p_mus, p_logsigmas = [[0]*L]*(T+1), [[0]*L]*(T+1)               # p(z_i | z_i+1), top-down prior

        """ set-up LSTM cells """
        e_cells, e_states = [None]*L, [None]*L
        d_cells, d_states = [None]*L, [None]*L
        
        for l in range(L):
            e_cells[l] = tf.contrib.rnn.core_rnn_cell.LSTMCell( RNN_SIZES[l] )
            d_cells[l] = tf.contrib.rnn.core_rnn_cell.LSTMCell( RNN_SIZES[l] )
            e_states[l]  = e_cells[l].zero_state(_b, tf.float32)
            d_states[l]  = d_cells[l].zero_state(_b, tf.float32)

            """ set as standard Gaussian, N(0,I). """
            d_mus[0][l], d_logsigmas[0][l] = tf.zeros((_b, Z_SIZES[l])), tf.zeros((_b, Z_SIZES[l]))
            p_mus[0][l], p_logsigmas[0][l] = tf.zeros((_b, Z_SIZES[l])), tf.zeros((_b, Z_SIZES[l]))

        self.h_enc, self.e_mus, self.e_logsigmas = h_enc, e_mus, e_logsigmas
        self.h_dec, self.d_mus, self.d_logsigmas = h_dec, d_mus, d_logsigmas
        self.p_mus, self.p_logsigmas = p_mus, p_logsigmas
        self.e_cells, self.e_states = e_cells, e_states
        self.d_cells, self.d_states = d_cells, d_states
        self.z = [[0]*L]*(T+1)


    ###########################################
    """            LSTM cells               """
    ###########################################
    def lstm_encode(self, state, x, l, is_train):

        scope = 'lstm_encode_' + str(l)
        x = tf.reshape(x, (_b, -1))
        if x.get_shape()[1] != RNN_SIZES[l]:
            print(scope, ':', x.get_shape()[1:], '=>', RNN_SIZES[l])
            x = self.ls.dense(scope, x, RNN_SIZES[l])
    
        return self.e_cells[l](x, state)

    def lstm_decode(self, state, x, l, is_train):

        scope = 'lstm_decode_' + str(l)
        x = tf.reshape(x, (_b, -1))
        if x.get_shape()[1] != RNN_SIZES[l]:
            print(scope, ':', x.get_shape()[1:], '=>', RNN_SIZES[l])
            x = self.ls.dense(scope, x, RNN_SIZES[l])
    
        return  self.d_cells[l](x, state)

    ###########################################
    """             Encoder                 """
    ###########################################
    def encoder(self, x, t, is_train=True, do_update_bn=True):

        for l in range(L):
            scope = 'Encode_L' + str(l)
            with tf.variable_scope( scope, reuse=self.DO_SHARE):

                if l == 0:
                    x_hat  = x - self.canvase_previous(t)
    
                    h_dec_lowest_prev = self.h_dec[t-1][0] if t == 0 else tf.zeros((_b, RNN_SIZES[0]))
    
                    input = self.ii.read(x, x_hat, h_dec_lowest_prev)
                else:
                    input = self.h_enc[t][l-1]
    
                self.h_enc[t][l], self.e_states[l] = self.lstm_encode(self.e_states[l], input, l, is_train)
    
                input = self.ls.dense(scope, self.h_enc[t][l], Z_SIZES[l]*2  )
                self.z[t][l], self.e_mus[t][l], self.e_logsigmas[t][l] = self.ls.vae_sampler_w_feature_slice( input, Z_SIZES[l])

        """ classifier """
        logit = self.ls.dense('top', self.h_enc[t][-1], self.d.l, activation=tf.nn.elu)
        return logit

    ###########################################
    """             Decoder                 """
    ###########################################
    def decoder(self, t, is_train=True, do_update_bn=True ):

        for l in range(L-1, -1, -1):
            scope = 'Decoder_L' + str(l)
            with tf.variable_scope( scope, reuse=self.DO_SHARE):
    
                if l == L-1:
                    input = self.z[t][l]
                else:
                    input = self.concat(self.z[t][l], self.h_dec[t][l+1], l)
    
                self.h_dec[t][l], self.d_states[l] = self.lstm_decode(self.d_states[l], input, l, is_train)
    
                """ go out to the input space """
                if l == 0:
                    # [ToDo] replace bellow reconstructor with conv-lstm 
                    if _is_3d:

                        o = self.canvase_previous(t) + self.ii.write(self.h_dec[t][l])
                        #if t == T-1: # for MNIST
                        o = tf.nn.sigmoid(o)
                        self.cs[t] = o
                    else:
                        self.cs[t] = tf.nn.sigmoid( self.canvase_previous(t) + self.ii.write(self.h_dec[t][l]) )
        return self.cs[t]


    """ set prior after building the decoder """
    def prior(self, t):

        for l in range(L-1, -1, -1):
            scope = 'Piror_L' + str(l)

            
            """ preparation for p_* for t+1 and d_* for t with the output from lstm-decoder"""
            if l != 0:
                input = self.ls.dense(scope, self.h_dec[t][l], Z_SIZES[l]*2+Z_SIZES[l-1]*2  )
                self.p_mus[t+1][l], self.p_logsigmas[t+1][l], self.d_mus[t][l], self.d_logsigmas[t][l] = self.ls.split( input, 1, [Z_SIZES[l]]*2 + [Z_SIZES[l-1]]*2)
            else:
                """ no one uses d_* """
                input = self.ls.dense(scope, self.h_dec[t][l], Z_SIZES[l]*2 )
                self.p_mus[t+1][l], self.p_logsigmas[t+1][l] = self.ls.split( input, 1, [Z_SIZES[l]]*2)


            """ setting p_mus[0][l] and p_logsigmas[0][l] """ 
            if t == 0:
                if l == L-1:
                    """ has already been performed at init() """ 
                    pass
                else:
                    """ by using only decoder's top-down path as prior since p(z) of t-1 does not exist """ 
                    self.p_mus[t][l], self.p_logsigmas[t][l] = self.d_mus[t][l+1], tf.exp(self.d_logsigmas[t][l+1]) # Eq.19 at t=0
            else:
                if l == L-1:
                    """ has already been performed at t-1 """ 
                    pass
                else:
                    """ update p(z) of current t """
                    _, self.p_mus[t][l], self.p_logsigmas[t][l]= self.ls.precision_weighted_sampler(
                            scope,
                            (self.p_mus[t][l], tf.exp(self.p_logsigmas[t][l])),
                            (self.d_mus[t][l+1], tf.exp(self.d_logsigmas[t][l+1]))
                    )  # Eq.19
    
    ###########################################
    """           Build Graph               """
    ###########################################
    def build_graph_train(self, x_l, y_l, x, is_supervised=True):

        o = dict()  # output
        loss = 0
        logit_ls = []

        """ Build DRAW """
        for t in range(T):
            logit_ls.append(self.encoder(x, t))
            x_reconst = self.decoder(t)
            self.prior(t)

            if t == 0:
                self.DO_SHARE = DO_SHARE = True
                self.ii.set_do_share( DO_SHARE )
                self.ls.set_do_share( DO_SHARE )

        """ p(x|z) Reconstruction Loss """
        o['x']  = x
        o['cs'] = self.cs
        o['Lr'] = self.lf.get_loss_pxz(x_reconst, x, 'DiscretizedLogistic')
        loss += o['Lr']

        """ VAE KL-Divergence Loss """
        o['KL1'], o['KL2'], o['Lz'] = self.lf.get_loss_kl_draw(self)
        loss += self.lambda_z_wu * o['Lz']

        """ set losses """
        o['loss'] = loss
        self.o_train = o

        """ set optimizer """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        grads = optimizer.compute_gradients(loss)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                #g = tf.Print(g, [g], "g %s = "%(v))
                grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
            else:
                print('g is None:', v)
                v = tf.Print(v, [v], "v = ", summarize=10000)
        self.op = optimizer.apply_gradients(grads) # return train_op


    def build_graph_test(self, x_l, y_l, is_supervised=False):

        o = dict()  # output
        loss = 0
        logit_ls = []

        """ Build DRAW """
        for t in range(T):
            logit_ls.append(self.encoder(x_l, t, is_train=False, do_update_bn=False ))
            x_reconst = self.decoder(t)
            self.prior(t)

            if t == 0:
                self.DO_SHARE = DO_SHARE = True
                self.ii.set_do_share( DO_SHARE )
                self.ls.set_do_share( DO_SHARE )


        """ classification loss """
        if is_supervised:
            o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_ls[-1], y_l)
            loss += o['Ly']

        """ for visualizationc """
        o['z'], o['y'] = logit_ls[-1], y_l

        """ set losses """
        o['loss'] = loss
        self.o_test = o

    ###########################################
    """             Utilities               """
    ###########################################
    def canvase_previous( self, t):
        if _is_3d:
            c_prev = tf.zeros((_b, _h,_w,_c)) if t == 0 else self.cs[t-1]
        else:
            c_prev = tf.zeros((_b, _img_size)) if t == 0 else self.cs[t-1]
        return c_prev

    def concat( self, x1, x2, l):
        if False: # [ToDo]
            x1 = tf.reshape( x1,  (_b, IMAGE_SIZES[l][0], IMAGE_SIZES[l][1], -1 ))
            x2 = tf.reshape( x2,  (_b, IMAGE_SIZES[l][0], IMAGE_SIZES[l][1], -1 ))
            return tf.concat([x1, x2], 3)
        else:
            x1 = tf.reshape( x1,  (_b, -1 ))
            x2 = tf.reshape( x2,  (_b, -1 ))
            return tf.concat([x1, x2], 1)
