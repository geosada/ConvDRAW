import tensorflow as tf
import numpy as np
import sys, os, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util')
from dataset import Dataset
from ConvDRAW import ConvDRAW

DO_CLASSIFY = False
DO_RECONST  = True
DO_DRAW = True
IS_MODEL_DRAW = True


tf.flags.DEFINE_string("dataset", "SVHN", "MNIST / CIFAR10 / SVHN ")
tf.flags.DEFINE_boolean("read_attn",  True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", False, "enable attention for writer")
tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
tf.flags.DEFINE_string("dir_logs", "./out/", "")
FLAGS = tf.flags.FLAGS

N_EPOCHS = 100
BATCH_SIZE = 50
N_PLOTS  = 5000

FILE_OF_CKPT  = os.path.join(FLAGS.dir_logs,"drawmodel.ckpt")

# learning rate decay
STARTER_LEARNING_RATE = 1e-3
DECAY_AFTER = 2
DECAY_INTERVAL = 2
DECAY_FACTOR = 0.97

# warming-up coefficient for KL-divergence term
Nt = 10 # warmig-up during the first Nt epochs
_lambda_z_wu = np.linspace(0, 1, Nt)

d = Dataset(FLAGS.dataset, BATCH_SIZE)

def test():
    accur = []
    for i in range(d.n_batches_test):
        r = sess.run(m.o_test)
        accur.append( r['accur'])
    return np.mean(accur, axis=0)

def draw(input, canvases):

    """ draw reconstructed image """
    if IS_MODEL_DRAW:
        cs = canvases
    else:
        cs = [canvases] # not have time dimension

    _draw_times = len(cs)

    """ make dynamic image """
    col, row = 7,7
    N_IMAGES = col*row

    for t in range(_draw_times):
        rows = []
        images = cs[t][:N_IMAGES]
        images = [tf.squeeze(image, [0]) for image in tf.split(images, N_IMAGES, 0)]
        for i in range(row):
            rows.append(tf.concat( images[col * i + 0:col * i + col], 1))
        o = tf.concat(rows, 0)
            
        o = tf.image.convert_image_dtype(o, tf.uint8, saturate=True)
        with open("image_dynamic%02d.png"%(t), 'wb') as f:
            f.write(sess.run(tf.image.encode_png(o)))
        
         
    """ make static image """
    cs_static = np.transpose(np.array(cs), (1,0,2,3,4))
    # -> (batch_size, draw_time, h,w,c)
    
    N_IMAGES = 8
    cs_static = cs_static[:N_IMAGES]
    list_of_sequential_image = [tf.squeeze(image_sequence, [0]) for image_sequence in tf.split(cs_static, N_IMAGES, 0)]
    # -> [(draw_time, h,w,c) x N_IMAGES]

    a_row = []
    for i, sequential_image in enumerate(list_of_sequential_image): # <- N_IMAGES = number of rows
        images = [tf.squeeze(image, [0]) for image in tf.split(sequential_image , _draw_times, 0)]
        # -> [(h,w,c) x _draw_times]

        """ append the original image on the end of each row"""
        images.append(input[i])
        a_row.append(tf.concat(images, 1))
    o = tf.concat(a_row,0)

    o = tf.image.convert_image_dtype(o, tf.uint8, saturate=True)
    with open("test.png", 'wb') as f:
        f.write(sess.run(tf.image.encode_png(o)))

    print('draw done.')
    return

def test_draw():

    input_image = np.load('input.npy')
    cs = np.load('cs.npy')
    print(cs.shape)
    #row, column = 8,8
    n_images = 9
    o = np.zeros((0, 32,32,3))
    o = []
    idx = 11
    for i in range(n_images):
        a_image = []
        for t in range(len(cs)):
            a_image.append(cs[t][idx])
        a_image = np.array(a_image)
        o.append(a_image)

    o = np.array(o)
    sess  = tf.InteractiveSession()
    with open("test.png", 'wb') as f:

        o = tf.concat( tf.concat(o,1), 0)
        o = [tf.squeeze(image, [0]) for image in tf.split(o, n_images, 0)]
        a = []
        for row in o:
            b = [tf.squeeze(image, [0]) for image in tf.split(row, 8, 0)]
            a.append(tf.concat(b, 1))
        o = tf.concat(a,0)
        o = tf.image.convert_image_dtype(o, tf.uint8, saturate=True)
        f.write(sess.run(tf.image.encode_png(o)))
    sess.close()

with tf.Graph().as_default() as g:

    ###########################################
    """             Load Data               """
    ###########################################
    (xtrain_l, ytrain_l), xtrain, (xtest , ytest) = d.get_tfrecords()

    ###########################################
    """        Build Model Graphs           """
    ###########################################
    with tf.variable_scope("watashinomodel") as scope:

        lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        lambda_z_wu = tf.placeholder(tf.float32, shape=(), name="lambda_z_wu")

        m = ConvDRAW(d, lr, lambda_z_wu, FLAGS.read_attn, FLAGS.write_attn, do_classify=DO_CLASSIFY, do_reconst=DO_RECONST )

        print('... now building the graph for training.')
        m.build_graph_train(xtrain_l, ytrain_l, xtrain)
        scope.reuse_variables()
        if DO_CLASSIFY:
            print('... now building the graph for test.')
            m.build_graph_test(xtest, ytest)


    ###########################################
    """              Init                   """
    ###########################################
    init_op = tf.global_variables_initializer()
    #for v in tf.all_variables(): print("%s : %s" % (v.name,v.get_shape()))

    sess  = tf.InteractiveSession()
    sess.run(init_op)
    saver = tf.train.Saver()
    _lr, ratio = STARTER_LEARNING_RATE, 1.0

    if FLAGS.restore:
        print("... restore from the last check point.")
        saver.restore(sess, FILE_OF_CKPT)
    ###########################################
    """         Training Loop               """
    ###########################################
    print('... start training')
    tf.train.start_queue_runners(sess=sess)
    for epoch in range(1, N_EPOCHS+1):

        # set coefficient of warm-up
        idx = -1 if Nt <= epoch else epoch
        _lambda_z_wu[idx]

        for i in range(d.n_batches_train):
        
            feed_dict = {lr:_lr, lambda_z_wu:_lambda_z_wu[idx]}

            """ do update """
            time_start = time.time()
            r, op, current_lr = sess.run([m.o_train, m.op, m.lr], feed_dict=feed_dict)
            elapsed_time = time.time() - time_start
    
            if i % 300 == 0:
                print(" iter:%2d, loss: %s, Lr: %s, Lz: %s, KL: %s, time:%.3f" % \
                         (i, r['loss'], r['Lr'], r['Lz'], r['KL2'], elapsed_time ))

            if DO_DRAW and DO_RECONST and i%300==0:
                draw(r['x'], r['cs'])

        """ test """
        if DO_CLASSIFY and epoch % 2 == 0:
            time_start = time.time()
            accur = test()
            elapsed_time = time.time() - time_start
            print("epoch:%d, accur: %s, time:%.3f" % (epoch, accur, elapsed_time ))

        """ save """
        if epoch % 5 == 0:
            print("Model saved in file: %s" % saver.save(sess,FILE_OF_CKPT))

        """ drawing
        """
        if DO_DRAW and DO_RECONST and epoch % 1 == 0:
            draw(r['x'], r['cs'])

        """ learning rate decay"""
        if (epoch % DECAY_INTERVAL == 0) and (epoch > DECAY_AFTER):
            ratio *= DECAY_FACTOR
            _lr = STARTER_LEARNING_RATE * ratio
            print('lr decaying is scheduled. epoch:%d, lr:%f <= %f' % ( epoch, _lr, current_lr))

sess.close()
print('Done drawing! Have a nice day! :) by Eric Jang')
