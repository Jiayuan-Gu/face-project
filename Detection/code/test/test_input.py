"""FCN Input Interface."""

import os
import random

import numpy as np
from scipy import ndimage
from scipy import misc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from matplotlib.patches import Rectangle
import tensorflow as tf
import test_config as config
from test_config import logger


class TEST_INPUT:
  def __init__(self):
    self.batch_size = config.batch_size
    self.fig_list = os.listdir(config.test_dir)
    self.curFig = -1
    self.figName = None
    self.numFig = len(self.fig_list)
    self.isValid = self.numFig>0

  def next_fig(self):
    if self.isValid == False:
      return None

    self.curFig += 1
    if self.curFig+1>=self.numFig:
      self.isValid = False
    filename = self.fig_list[self.curFig]
    self.image = mpimg.imread(os.path.join(config.test_dir,filename))
    self.image = misc.imresize(self.image,0.8)
    # plt.imshow(self.image)
    # plt.show()
    # image = np.expand_dims(image,0)
    self.figName,_ = os.path.splitext(filename)
    return self.image



  def save(self,image,heatmap):
    heatmap = np.squeeze(heatmap)
    conn,numCandidate = ndimage.label(heatmap>=config.thresh)
    candis = ndimage.find_objects(conn)
    # plt.imshow(256-image)
    # plt.show()
    for candi in candis:
      image[candi[0].start:candi[0].stop,candi[1].start,0]= 0
      image[candi[0].start:candi[0].stop,candi[1].start,1]= 255
      image[candi[0].start:candi[0].stop,candi[1].start,2]= 0

      image[candi[0].start:candi[0].stop,candi[1].stop-1,0]= 0
      image[candi[0].start:candi[0].stop,candi[1].stop-1,1]= 255
      image[candi[0].start:candi[0].stop,candi[1].stop-1,2]= 0

      image[candi[0].start,candi[1].start:candi[1].stop,0] = 0
      image[candi[0].start,candi[1].start:candi[1].stop,1] = 255
      image[candi[0].start,candi[1].start:candi[1].stop,2] = 0

      image[candi[0].stop-1,candi[1].start:candi[1].stop,0] = 0
      image[candi[0].stop-1,candi[1].start:candi[1].stop,1] = 255
      image[candi[0].stop-1,candi[1].start:candi[1].stop,2] = 0      # ly = candi[0].start
      # height = candi[0].stop-candi[0].start
      # lx = candi[1].start
      # width = candi[1].stop-candi[1].start
      # plt.gca.add_patch(Rectangle((lx,ly),width,height,fill=None,alpha=1))
    image = 256-image

    mpimg.imsave(os.path.join(config.output_dir,'pred','%s_bbxs.jpg'%self.figName),image)
    mpimg.imsave(os.path.join(config.output_dir,'pred','%s_heatmap.jpg'%self.figName),heatmap)