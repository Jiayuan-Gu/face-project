import os
import shutil
import logging
from datetime import datetime
import tensorflow as tf

version = 'v0.1.0.20170110'
run = 'train'
time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
is_training = False

# tf config
gpu_device = '4'
session_config = tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True))

# resfcn_test
test_dir = '/data-disk/gujy/Face/train_set';
output_dir = os.path.join('/data-disk/gujy/Face/test',version,run)
resume = True
restore_path = '/data-disk/gujy/Face/train/v0.0.0.20161217/adam/4_0/checkpoint/model.ckpt-91500'
# restore_path = None;

# resfcn_model
CONV_STDDEV = None
BIAS_STDDEV = 0.0 
DECONV_STDDEV = None
## Weight decay
CONV_WD = 1e-8
BIAS_WD = None
DECONV_WD = 1e-8


# resfcn_input
IMAGE_SIZE = [None,None,3]
LABEL_SIZE = [None,None,1]
batch_size = 1
thresh = 0.95

# whether to new a folder
if not os.path.isdir(output_dir):
	os.makedirs(os.path.join(output_dir,'config'))
	os.makedirs(os.path.join(output_dir,'pred'))
shutil.copy('test_config.py',os.path.join(output_dir,'config','%s.config'%time_stamp))
log_file = os.path.join(output_dir,'config','%s.log'%time_stamp)

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
