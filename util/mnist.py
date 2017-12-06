# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#https://raw.githubusercontent.com/tensorflow/tensorflow/r0.11/tensorflow/examples/how_tos/reading_data/convert_to_records.py

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  # MNIST filenames
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

IMAGE_SIZE   = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

tf.app.flags.DEFINE_integer('validation_size', 5000,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
tf.app.flags.DEFINE_integer('n_labeled', 100, "The number of labeled examples")
tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

DATA_DIR = 'SET_YOUR_OWN_MNIST_DIR_PATH'
VALIDATION_SIZE = 5000

N_LABELED = 100

DATASET_SEED = 1
#FLAGS = tf.app.flags.FLAGS

N_CLASSES = 10
N_EXAMPLES_TRAIN = 55000
N_EXAMPLES_TEST = 10000

TFRECORDS_TRAIN              = "train.tfrecords"
TFRECORDS_TRAIN_LABELED      = "train_labeled.tfrecords"
TFRECORDS_TEST               = "test.tfrecords"
TFRECORDS_VALIDATION         = "validation.tfrecords"
TFRECORDS_VALIDATION_LABELED = "validation_labeled.tfrecords"

def prepare_dataset(is_process_only_labeled_data=True):
    # Get the data as 1D if reshape=False, otherwise as 3D.
    data_sets = mnist.read_data_sets(DATA_DIR, dtype=tf.uint8, reshape=True, validation_size=VALIDATION_SIZE)
    train_images = data_sets.train.images
    train_labels = data_sets.train.labels

    test_images = data_sets.test.images
    test_labels = data_sets.test.labels

    rng = np.random.RandomState(DATASET_SEED)
    rand_ix = rng.permutation(N_EXAMPLES_TRAIN)
    #_train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]
    train_images = train_images[rand_ix]
    train_labels = train_labels[rand_ix]

    n_per_class = int(N_LABELED / 10)
    #train_images_labeled = np.zeros((N_LABELED, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    train_images_labeled = np.zeros((N_LABELED, IMAGE_PIXELS), dtype=np.float32)
    train_labels_labeled = np.zeros((N_LABELED), dtype=np.int64)

    for i in range(10):
        indxes_of_this_class = np.where(train_labels == i)[0]
        train_images_labeled[i * n_per_class:(i + 1) * n_per_class] = train_images[indxes_of_this_class[0:n_per_class]]
        train_labels_labeled[i * n_per_class:(i + 1) * n_per_class] = train_labels[indxes_of_this_class[0:n_per_class]]
        train_images = np.delete(train_images, indxes_of_this_class[0:n_per_class], 0)
        train_labels = np.delete(train_labels, indxes_of_this_class[0:n_per_class])

    rand_ix_l = rng.permutation(N_LABELED)
    train_images_labeled = train_images_labeled[rand_ix_l]
    train_labels_labeled = train_labels_labeled[rand_ix_l]

    print('num of labeled examples:', N_LABELED)
    # Convert to Examples and write the result to TFRecords.
    convert_to(train_images_labeled, train_labels_labeled, TFRECORDS_TRAIN_LABELED)
    if not is_process_only_labeled_data:
        convert_to(train_images,         train_labels,         TFRECORDS_TRAIN)
        convert_to(test_images,          test_labels,          TFRECORDS_TEST)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to(images, labels, filename):

    print(images.shape, labels.shape)
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))

    #rows = images.shape[1]
    #cols = images.shape[2]
    #depth = images.shape[3]
    rows = 28
    cols = 28
    depth = 1

    filename = os.path.join(DATA_DIR, filename)
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image = images[index].tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _float_feature(image)}))
        writer.write(example.SerializeToString())
    writer.close()


""" begin https://github.com/tensorflow/ecosystem/blob/master/docker/mnist.py
"""
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                    'image_raw': tf.FixedLenFeature([IMAGE_PIXELS], tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64),
            })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    
    image = features['image_raw']

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.    Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # -> [0, 1]
    image = tf.cast(image, tf.float32) * (1. / 255)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 10)

    return image, label


def inputs(batch_size, type='train'):
    """Reads input data.
    Args:
        batch_size: Number of examples per returned batch.
    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
            in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
            a number in the range [0, mnist.NUM_CLASSES).
    """

    if type == 'train':
        filename = TFRECORDS_TRAIN
    elif type == 'train_labeled':
        filename = TFRECORDS_TRAIN_LABELED
    elif type == 'test':
        filename = TFRECORDS_TEST
    elif type == 'validation':
        filename = TFRECORDS_VALIDATION

    filename = os.path.join(DATA_DIR, filename)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)

        return images, sparse_labels


""" end https://github.com/tensorflow/ecosystem/blob/master/docker/mnist.py
"""

def main(argv):
    prepare_dataset()

if __name__ == '__main__':
    tf.app.run()
