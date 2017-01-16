import sys
import os.path
from datetime import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import test_config as config
from test_config import logger
import test_model as resfcn
import test_input

def test():
  inputs = test_input.TEST_INPUT();

  # Get images and labels for FCN.
  image = tf.placeholder(tf.float32,shape=[None,None,config.IMAGE_SIZE[2]],name='image_placeholder');
  precrop = tf.image.resize_image_with_crop_or_pad(image,900,1600)
  crop = tf.image.resize_images(precrop,[450,800])
  images = tf.expand_dims(crop,0)
  images = (images-128)/128
  print('inputs initialize.')

  # Build a Graph that computes the score map predictions from the
  # inference model.
  preds = resfcn.inference(images)
  # tf.image_summary('preds',preds[:,:,:,int(config.LABEL_SIZE[-1]/2)],max_images=int(config.batch_size/2))
  print('inference initialize.')

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables(),name='saver')
  print('saver initialize.')

  # Start running operations on the Graph.
  sess = tf.Session(config=config.session_config)
  saver.restore(sess,config.restore_path)
  logger.info('load from %s',config.restore_path)

  print('test begins.')
  
  while inputs.isValid:
    # try:
    scan_time = datetime.now().strftime('%H:%M')
    image_batch = inputs.next_fig()
    # except Exception:
    #   logger.info('encounter with error. Abort fig %s.'%inputs.figName)
    #   inputs.next_fig()
    #   continue

    start_time = time.time()
    crop_image,heatmap = sess.run([crop,preds],feed_dict = {image:image_batch})
    duration = time.time() - start_time

    # num_examples_per_step = inputs.batch_size
    # examples_per_sec = num_examples_per_step / duration
    # sec_per_batch = float(duration)

    format_str = ('%s: %s (%.3f sec/fig)')
    print (format_str % (scan_time, inputs.figName, duration))

    inputs.save(crop_image,heatmap)

  sess.close()
  config.close_all()


def main(argv=None):  # pylint: disable=unused-argument
  os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
  test()


if __name__ == '__main__':
  tf.app.run()
