import tensorflow as tf
import numpy as np 
import cv2 
import sys 
import os 

import sys
sys.path.append('..')
import config


def parser(record):
    keys_to_features = {
        'image': tf.FixedLenFeature([], tf.string),
        'heatmap': tf.FixedLenFeature([], tf.string),
        'offset': tf.FixedLenFeature([], tf.string),
        'paf': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string)
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed['image'], tf.float32)
    # image = tf.cast(image, tf.float32) *1./255
    image = tf.reshape(image, [512, 512, 3])
    heatmap = tf.decode_raw(parsed['heatmap'], tf.float32)
    heatmap = tf.reshape(heatmap, [128, 128, 4])
    offset = tf.decode_raw(parsed['offset'], tf.float32)
    offset = tf.reshape(offset, [128, 128, 2])
    paf = tf.decode_raw(parsed['paf'], tf.float32)
    paf = tf.reshape(paf, [128, 128, 8])
    mask = tf.decode_raw(parsed['mask'], tf.float32)
    mask = tf.reshape(mask, [128, 128])
    # pts = tf.cast(pts, tf.float32)

    return {'image': image,
         'heatmap': heatmap, 
         'offset': offset,
         'paf': paf,
         'mask': mask}

def input_fn(filenames='helper/data_2.tfrecords', is_training=True):
    if is_training: # train
        dataset = (
            tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=3)
                .shuffle(buffer_size=5*config.BATCH_SIZE)
                .apply(tf.contrib.data.map_and_batch(parser, config.BATCH_SIZE))
                .prefetch(buffer_size=1)
        )
    else: #eval
        dataset = (
            tf.data.TFRecordDataset(filenames=filenames)
                .apply(tf.contrib.data.map_and_batch(parser, config.BATCH_SIZE))
                .prefetch(buffer_size=1)
        )
    # print(dataset)

    iterator = dataset.make_initializable_iterator()
    data = iterator.get_next()
    iterator_init_op = iterator.initializer

    return data, iterator_init_op