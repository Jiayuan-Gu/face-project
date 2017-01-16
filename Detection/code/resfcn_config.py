import os
import shutil
import logging
from datetime import datetime
import tensorflow as tf

version = 'v0.1.0.20170110'
run = '1_0'
time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
is_training = True

# tf config
gpu_device = '5'
session_config = tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True))

# resfcn_train
data_dir = '/data-disk/gujy/Face/WF/tf_records'
train_dir = os.path.join('/data-disk/gujy/Face/train',version,run)

resume = True
restore_path = '/data-disk/gujy/Face/train/v0.0.0.20161217/adam/3_0/checkpoint/model.ckpt-43000'
# restore_path = None
max_steps = 1e6
num_steps_debug = 20
num_steps_summary = 100
num_steps_checkpoint = 250

# resfcn_model
CONV_STDDEV = None
BIAS_STDDEV = 0.0 
DECONV_STDDEV = None 
## Weight decay
CONV_WD = 1e-8
BIAS_WD = None
DECONV_WD = 1e-8

# training setting
learning_rate = 1e-4
# optimizer = tf.train.GradientDescentOptimizer
optimizer = tf.train.AdamOptimizer

# resfcn_input
IMAGE_SIZE = [512,512,3]
LABEL_SIZE = [512,512,1]
CROP_SIZE = [512,512,4]
neg_add_w = 0.1
input_threads = 4
batch_size = 4
tfrecords = [os.path.join(data_dir,tfrecord) for tfrecord in os.listdir(data_dir)]

num_examples_per_epoch = 12880-362
num_batches_per_epoch =  int(num_examples_per_epoch/batch_size)


# whether to new a folder
if not os.path.isdir(train_dir):
	os.makedirs(os.path.join(train_dir,'fig'))
	os.makedirs(os.path.join(train_dir,'checkpoint'))
	os.makedirs(os.path.join(train_dir,'summary'))
	os.makedirs(os.path.join(train_dir,'log'))
shutil.copy('resfcn_config.py',os.path.join(train_dir,'log','%s.config'%time_stamp))
shutil.copy('resfcn_model.py',os.path.join(train_dir,'log','%s.model'%time_stamp))
log_file = os.path.join(train_dir,'log','%s.log'%time_stamp)

# logger config
logger = logging.getLogger('mylog')
logger.propagate = False
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fileHandler = logging.FileHandler(log_file)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(fmt)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(fmt)
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

def close_all():
	fileHandler.close()
	consoleHandler.close()