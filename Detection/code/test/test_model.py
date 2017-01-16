import os
import sys

import numpy as np
import tensorflow as tf

import test_input
import test_config as config
from test_config import logger


# Global constants describing the Residue FCN model.
## Standard deviation for initialization
CONV_STDDEV = config.CONV_STDDEV
BIAS_STDDEV = config.BIAS_STDDEV
DECONV_STDDEV = config.DECONV_STDDEV 
## Weight decay
CONV_WD = config.CONV_WD
BIAS_WD = config.BIAS_WD
DECONV_WD = config.DECONV_WD


def _get_variable(name, shape, initializer, 
  weight_decay=None,dtype=tf.float32,trainable=True):

  # Optionally add weigth decay according to weight_decay
  if weight_decay is not None:
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    regularizer = None

  var = tf.get_variable(name=name,shape=shape,
    initializer=initializer,dtype=dtype,
    regularizer=regularizer,trainable=trainable)

  return var


def _get_filter_stddev(kernel_size,channel_size):
  filter_stddev = np.sqrt(2.0/(kernel_size[0]*kernel_size[1]*channel_size))
  return filter_stddev

def _conv2d(input_op, kernel_shape, op_name='conv2d',
  stride = None, padding='SAME',
  conv_stddev = CONV_STDDEV, conv_wd=CONV_WD):
  
  if conv_stddev is None:
    conv_stddev = _get_filter_stddev(kernel_shape[0:2],kernel_shape[2])

  if stride is None:
    stride = [1, 1, 1, 1]

  kernel = _get_variable(name='conv_kernel',shape=kernel_shape,
    initializer=tf.truncated_normal_initializer(stddev=conv_stddev),
    weight_decay=conv_wd)

  conv = tf.nn.conv2d(input_op, kernel, strides=stride, padding=padding,name=op_name)
  return conv

def _conv2d_bias(input_op, kernel_shape, op_name='conv2d_bias',
  stride = None, padding='SAME',
  conv_stddev = CONV_STDDEV, conv_wd=CONV_WD,
  bias_stddev = BIAS_STDDEV, bias_wd=BIAS_WD):
  
  with tf.variable_scope(op_name) as scope:
    conv = _conv2d(input_op, kernel_shape,
      stride=stride, padding=padding,
      conv_stddev = conv_stddev, conv_wd=conv_wd)

    bias = _get_variable('bias',shape=kernel_shape[3],
      initializer=tf.truncated_normal_initializer(stddev=bias_stddev),weight_decay=bias_wd)

    conv_bias = tf.nn.bias_add(conv, bias, name='conv2d_bias')

  return conv_bias


def _batch_norm(input_op,channel_size,op_name='batch_norm'):
  
  with tf.variable_scope(op_name) as scope:
    offset = _get_variable(name='offset',shape=channel_size,initializer=tf.zeros_initializer)
    scale = _get_variable(name='scale',shape=channel_size,initializer=tf.ones_initialzer)
    moving_mean = tf.get_variable(name='moving_mean',shape=channel_size,
      initializer=tf.zeros_initializer,trainable=False)
    moving_variance = tf.get_variable(name='moving_variance',shape=channel_size,
      initializer=tf.ones_initialzer,trainable=False)

    mean,variance  = tf.nn.moments(input_op,axes=[0,1,2])
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean,MOVING_AVERAGE_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance,variance,MOVING_AVERAGE_DECAY)
    tf.add_to_collection('UPDATE_OP',update_moving_mean)
    tf.add_to_collection('UPDATE_OP',update_moving_variance)

  if FLAGS.is_training:
    norm = tf.nn.batch_normalization(deconv,moving_mean,moving_variance,offset,scale,BN_EPSILON)
  else:
    norm = tf.nn.batch_normalization(deconv,mean,var,offset,scale,BN_EPSILON)

  return norm


def _conv_layer(input_op, kernel_shape, op_name='conv_layer',
  stride=None, padding='SAME',
  batch_norm=False, activation=tf.nn.elu,
  conv_stddev = CONV_STDDEV, conv_wd=CONV_WD, 
  bias_stddev = BIAS_STDDEV, bias_wd=BIAS_WD):

  with tf.variable_scope(op_name) as scope:
    pre_activation = _conv2d_bias(input_op, kernel_shape, stride=stride, padding=padding,
      conv_stddev = conv_stddev, conv_wd=conv_wd, bias_stddev = bias_stddev, bias_wd=bias_wd)

    if batch_norm:
      pre_activation = _batch_norm(pre_activation)
    
    if activation is not None:
      conv = activation(pre_activation)
    else:
      conv = pre_activation

  if config.is_training:
      print('conv layer:%s is established'%op_name)
      logger.debug(str(conv))

  return conv



def _deconv_layer(input_op, kernel_shape, shape_op, op_name='deconv_layer',
  stride=None, padding='SAME',
  deconv_stddev=DECONV_STDDEV, deconv_wd=DECONV_WD):


  if deconv_stddev is None:
    deconv_stddev = _get_filter_stddev(kernel_shape[0:2],kernel_shape[3])

  if stride is None:
    stride = [1,2,2,1]

  with tf.variable_scope(op_name) as scope:
      kernel = _get_variable(name='deconv_kernel', shape = kernel_shape, 
        initializer=tf.truncated_normal_initializer(stddev=deconv_stddev),
        weight_decay=deconv_wd)
      deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                      output_shape= shape_op,
                                      strides=stride, padding=padding,name='deconv')
  
  if config.is_training:
    print('deconv layer:%s is established...'%op_name)
    logger.debug(str(deconv))
  
  return deconv

def _pool_layer(input_op, op_name='pool_layer', pooling=tf.nn.max_pool):
  with tf.variable_scope(op_name) as scope:
    pool = pooling(input_op, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  if config.is_training:
    print('pool layer:%s is established...'%op_name)
    logger.debug(str(pool))
  return pool

def _unpool_layer(input_op, size=None, op_name='unpool_layer'):
  if size is None:
    size = 2*(tf.shape(input_op)[1:3])

  with tf.variable_scope(op_name) as scope:
    unpool = tf.image.resize_images(input_op, size=size)
  
  if config.is_training:
    print('unpool layer:%s is established...'%op_name)
    logger.debug(str(unpool))
  return unpool


def _residue_block(input_op, residue_op, kernel_shape, 
  op_name='residue_block', activation=tf.nn.elu, batch_norm=False, 
  conv_stddev = CONV_STDDEV, conv_wd=CONV_WD,
  bias_stddev = BIAS_STDDEV, bias_wd=BIAS_WD):
	

    with tf.variable_scope(op_name) as scope:
        conv1 = _conv_layer(input_op,kernel_shape,op_name='residue_conv1',
          batch_norm=batch_norm, activation=activation,
          conv_stddev = conv_stddev, bias_stddev = bias_stddev, 
          conv_wd=conv_wd, bias_wd=bias_wd)

        conv2 = _conv_layer(input_op,kernel_shape,op_name='residue_conv2',
          batch_norm=batch_norm, activation=None,
          conv_stddev = conv_stddev, bias_stddev = bias_stddev, 
          conv_wd=conv_wd, bias_wd=bias_wd)

        addition = tf.add(residue_op,conv2,name='addition')
        
        if activation is not None:
          residue = activation(addition)
        else:
          residue = addition

    if config.is_training:
      print('residue block:%s is established...'%op_name)
      logger.debug(str(residue))

    return residue


def inference(images):
  # Convolution
  # stage1
  conv1_1 = _conv_layer(input_op=images, op_name='conv1_1', kernel_shape=[1,1,images.get_shape().as_list()[-1],32])
  block1_1 = _residue_block(input_op=conv1_1,residue_op=conv1_1, op_name='block1_1', kernel_shape=[3,3,32,32])
  block1_2 = _residue_block(input_op=block1_1,residue_op=block1_1, op_name='block1_2', kernel_shape=[3,3,32,32])
  pool1 = _pool_layer(block1_2,'pool1')

  # stage2
  conv2_1 = _conv_layer(input_op=pool1, op_name='conv2_1', kernel_shape=[1,1,32,64])
  block2_1 = _residue_block(input_op=conv2_1,residue_op=conv2_1, op_name='block2_1', kernel_shape=[3,3,64,64])
  block2_2 = _residue_block(input_op=block2_1,residue_op=block2_1, op_name='block2_2', kernel_shape=[3,3,64,64])
  pool2 = _pool_layer(block2_2,'pool2')

  # stage3
  conv3_1 = _conv_layer(input_op=pool2, op_name='conv3_1', kernel_shape=[1,1,64,128])
  block3_1 = _residue_block(input_op=conv3_1,residue_op=conv3_1, op_name='block3_1', kernel_shape=[3,3,128,128])
  block3_2 = _residue_block(input_op=block3_1,residue_op=block3_1, op_name='block3_2', kernel_shape=[3,3,128,128])
  pool3 = _pool_layer(block3_2,'pool3')

  # stage4
  conv4_1 = _conv_layer(input_op=pool3, op_name='conv4_1', kernel_shape=[1,1,128,256])
  block4_1 = _residue_block(input_op=conv4_1,residue_op=conv4_1, op_name='block4_1', kernel_shape=[3,3,256,256])
  block4_2 = _residue_block(input_op=block4_1,residue_op=block4_1, op_name='block4_2', kernel_shape=[3,3,256,256])
  pool4 = _pool_layer(block4_2,'pool4')

  # stage5
  conv5_1 = _conv_layer(input_op=pool4, op_name='conv5_1', kernel_shape=[1,1,256,512])
  block5_1 = _residue_block(input_op=conv5_1,residue_op=conv5_1, op_name='block5_1', kernel_shape=[3,3,512,512])
  block5_2 = _residue_block(input_op=block5_1,residue_op=block5_1, op_name='block5_2', kernel_shape=[3,3,512,512])

  # upsample and fuse
  deconv4_1 = _deconv_layer(input_op=block5_2, op_name='deconv4_1', 
    shape_op= tf.shape(block4_2), kernel_shape=[2,2,256,512])
  block4_3 = _residue_block(input_op=deconv4_1,residue_op=block4_2, op_name='block4_3', kernel_shape=[3,3,256,256])
  block4_4 = _residue_block(input_op=block4_3,residue_op=block4_3, op_name='block4_4', kernel_shape=[3,3,256,256])

  deconv3_1 = _deconv_layer(input_op=block4_4, op_name='deconv3_1', 
    shape_op= tf.shape(block3_2), kernel_shape=[2,2,128,256])
  block3_3 = _residue_block(input_op=deconv3_1,residue_op=block3_2, op_name='block3_3', kernel_shape=[3,3,128,128])
  block3_4 = _residue_block(input_op=block3_3,residue_op=block3_3, op_name='block3_4', kernel_shape=[3,3,128,128])

  deconv2_1 = _deconv_layer(input_op=block3_4, op_name='deconv2_1', 
    shape_op= tf.shape(block2_2), kernel_shape=[2,2,64,128])
  block2_3 = _residue_block(input_op=deconv2_1,residue_op=block2_2, op_name='block2_3', kernel_shape=[3,3,64,64])
  block2_4 = _residue_block(input_op=block2_3,residue_op=block2_3, op_name='block2_4', kernel_shape=[3,3,64,64])

  deconv1_1 = _deconv_layer(input_op=block2_4, op_name='deconv1_1', 
    shape_op= tf.shape(block1_2), kernel_shape=[2,2,32,64])
  block1_3 = _residue_block(input_op=deconv1_1,residue_op=block1_2, op_name='block1_3', kernel_shape=[3,3,32,32])
  block1_4 = _residue_block(input_op=block1_3,residue_op=block1_3, op_name='block1_4', kernel_shape=[3,3,32,32])

  conv1_3 = _conv2d_bias(input_op=block1_4, kernel_shape = [1,1,32,1], op_name='conv1_3')

  # sigmoid
  preds = tf.sigmoid(conv1_3,name='pred')  

  #conv1_3 = _conv_layer(input_op=block1_4, op_name='conv1_3', kernel_shape=[3,3,32,1])
  #conv1_4 = _conv_layer(input_op=block1_4, op_name='conv1_4', kernel_shape=[3,3,32,1])

  ## softmax
  #with tf.variable_scope('pred') as scope:
    #pred1 = tf.exp(conv1_3,name='pred1')
    #pred2 = tf.exp(conv1_4,name='pred2')
    #preds = tf.truediv(pred1,tf.add(pred1,pred2),name='softmax')

  return preds