import tensorflow as tf
import numpy as np
import sys, os, time


class Dataset(object):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size 

        if self.dataset == 'CIFAR10':
            n_train, n_test = 50000, 10000
            _h, _w, _c = 32,32,3
            _img_size = _h*_w*_c
            _l = 10
            _is_3d = True
        if self.dataset == 'SVHN':
            n_train, n_test = 73257, 26032
            _h, _w, _c = 32,32,3
            _img_size = _h*_w*_c
            _l = 10
            _is_3d = True
        elif self.dataset == 'MNIST':
            n_train, n_test = 55000, 10000
            _h, _w, _c = 28,28,1
            _img_size = _h*_w*_c
            _l = 10
            _is_3d = True

        self.h = _h
        self.w = _w
        self.c = _c
        self.l = _l
        self.is_3d     = _is_3d 
        self.img_size  = _img_size
        self.n_train   = n_train
        self.n_test    = n_test
        self.n_batches_train = int(n_train/batch_size)
        self.n_batches_test  = int(n_test/batch_size)

    def get_tfrecords(self):

        # xtrain: all records
        # *_l   : partial records
        if self.dataset == 'CIFAR10':
            from cifar10 import inputs, unlabeled_inputs
            xtrain_l, ytrain_l = inputs(batch_size=self.batch_size, train=True,  validation=False, shuffle=True)
            xtrain             = unlabeled_inputs(batch_size=self.batch_size,    validation=False, shuffle=True)
            xtest , ytest      = inputs(batch_size=self.batch_size, train=False, validation=False, shuffle=True)
        elif self.dataset =='SVHN':
            from svhn import inputs, unlabeled_inputs
            xtrain_l, ytrain_l = inputs(batch_size=self.batch_size, train=True,  validation=False, shuffle=True)
            xtrain             = unlabeled_inputs(batch_size=self.batch_size,    validation=False, shuffle=True)
            xtest , ytest      = inputs(batch_size=self.batch_size, train=False, validation=False, shuffle=True)
        elif self.dataset == 'MNIST':
            from mnist import inputs
            xtrain,_           = inputs(self.batch_size, 'train')
            xtrain_l, ytrain_l = inputs(self.batch_size, 'train_labeled')
            xtest , ytest      = inputs(self.batch_size, 'test')

        return (xtrain_l, ytrain_l), xtrain, (xtest , ytest)
