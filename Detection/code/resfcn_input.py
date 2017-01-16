"""FCN Input Interface."""
import os
import random

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import resfcn_config as config
from resfcn_config import logger


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _,serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example,
    features={
    'image': tf.VarLenFeature(dtype=tf.float32),
    'image_shape':tf.FixedLenFeature(shape=[3],dtype=tf.int64),
    'label': tf.VarLenFeature(dtype=tf.float32),
    'label_shape':tf.FixedLenFeature(shape=[3],dtype=tf.int64),
    'name': tf.FixedLenFeature(shape=[],dtype=tf.string),
    })

  image = tf.cast(features['image'],tf.float32)
  image_shape = tf.cast(features['image_shape'],tf.int32)
  image = tf.sparse_tensor_to_dense(image)
  image = tf.reshape(image,image_shape)

  label = tf.cast(features['label'],tf.float32)
  label_shape = tf.cast(features['label_shape'],tf.int32)
  label = tf.sparse_tensor_to_dense(label)
  label = tf.reshape(label,label_shape)
  label = tf.cast(label>1e-8,tf.float32)
  # label = tf.expand_dims(label,-1)

  # precrop = tf.concat(2,[image,tf.image.per_image_whitening(image),label])
  precrop = tf.concat(2,[image,label])
  crop = tf.cond(tf.reduce_any(tf.less(tf.shape(precrop),[1024,1024,4])),
  	lambda:tf.image.resize_image_with_crop_or_pad(precrop, 1024,1024),
  	lambda:tf.random_crop(precrop,[1024,1024,4]))
  
  # crop = tf.image.resize_image_with_crop_or_pad(precrop,1024,1024)
  crop = tf.image.resize_images(crop,config.IMAGE_SIZE[0:2])
  
  # raw_image = crop[:,:,0:3]
  image = crop[:,:,0:3]
  image = (image-128)/128
  label = crop[:,:,-1:]

  # raw_image.set_shape(config.IMAGE_SIZE)
  image.set_shape(config.IMAGE_SIZE)
  label.set_shape(config.LABEL_SIZE)

  name = tf.cast(features['name'],tf.string)

  neg_w = tf.reduce_mean(label)
  pos_w = tf.sub(1.0,neg_w,name='pos_w')

  neg_add_w = tf.constant(config.neg_add_w,name='neg_add_w')
  # tf.scalar_summary('neg_add_w',neg_add_w,['PARAM_OP'])
  # neg_w = tf.cond(tf.less(neg_w,tf.constant(1e-8)),lambda:tf.add(neg_w,neg_add_w),lambda:neg_w,name='neg_w')
  neg_w = tf.add(neg_w,neg_add_w,name='neg_w')
  if config.is_training:
    tf.scalar_summary('neg_add_w', neg_add_w,['PARAM_OP'])

  # debug
  logger.info('image:%s',str(image))
  logger.info('label:%s',str(label))
  logger.info('pos_w:%s',str(pos_w))
  logger.info('neg_w:%s',str(neg_w))
  logger.info('name:%s',str(name))

  return image,label,pos_w,neg_w,name #,image_shape,label_shape

def inputs():
  filename_queue = tf.train.string_input_producer(config.tfrecords)
  example = read_and_decode(filename_queue)
  batch = tf.train.shuffle_batch(
    example,
    batch_size=config.batch_size,num_threads=config.input_threads,
    capacity=128+config.batch_size,min_after_dequeue=100+config.batch_size)
  return batch