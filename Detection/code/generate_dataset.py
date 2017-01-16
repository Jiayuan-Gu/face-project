import os
import sys
from datetime import datetime
import logging

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf

rawDir = '/data-disk/gujy/Face/WF/preprocess'
dataDir = '/data-disk/gujy/Face/WF/tf_records'

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to_tfrecord(file_name,writer):
	data = sio.loadmat(file_name)
	image = data['img']
	label = data['label']
	name = str(data['imgName'])

	image_shape = image.shape
	label_shape = list(label.shape)+[1]
	image = np.reshape(image,image.size).tolist()
	label = np.reshape(label,label.size).tolist()
	# print(image_shape,label_shape)

	example = tf.train.Example(features=tf.train.Features(feature={
			'image': _float_feature(image),
			'image_shape':_int64_feature(image_shape),
			'label': _float_feature(label),
			'label_shape':_int64_feature(label_shape),
			'name':_bytes_feature([name.encode()])
			}))
	writer.write(example.SerializeToString())

def check_tfrecords(record_file,eventName):
	for serialized_example in tf.python_io.tf_record_iterator(record_file):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)
		image = example.features.feature['image'].float_list.value
		image = np.array(image)
		# print(image.shape)
		image_shape = example.features.feature['image_shape'].int64_list.value
		image = np.reshape(image,image_shape)
		# print(image_shape)
		label = example.features.feature['label'].float_list.value
		label = np.array(label)
		# print(label.shape)
		label_shape = example.features.feature['label_shape'].int64_list.value
		# print(label_shape)
		label=np.reshape(label,label_shape)
		# pos_w = example.features.feature['pos_w'].float_list.value
		# neg_w = example.features.feature['neg_w'].float_list.value
		name = example.features.feature['name'].bytes_list.value
		name = name[0].decode()
		print(name)
		# plt.subplot(121)
		# plt.imshow(255-image)
		# plt.title(name)
		# plt.subplot(122)
		# plt.imshow(label[:,:,0])
		# plt.savefig('%s.png'%eventName)
		# break;

def main(argv):
	index = int(argv[0]);
	eventName = os.listdir(rawDir)[index]

	if not os.path.isdir(dataDir):
		os.makedirs(dataDir)
		print('establish a new dir at %s.'%dataDir)

	logFile = os.path.join(dataDir,'%s.log'%eventName)

	logging.basicConfig(level=logging.DEBUG,
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
		datefmt='%m-%d-%H:%M:%S',
		filename=logFile,
		filemode='w')
	
	# file_list = [];
	# for file in os.listdir(os.path.join(rawDir,eventName)):
	# 	if file.endswith('.mat'):
	# 		file_list.append(file)

	# writer = tf.python_io.TFRecordWriter(os.path.join(dataDir,'%s.tfrecords'%eventName))

	# for file_name in file_list:
	# 	print(datetime.now().strftime('%H:%M:%S'),"%s:%s"%(eventName,file_name))
	# 	# logging.info("%s:%s"%(eventName,file_name))
	# 	try:
	# 		convert_to_tfrecord(os.path.join(rawDir,eventName,file_name),writer)
	# 	except:
	# 		logging.error('encounter error with %s.'%file_name)

	# writer.close()
	check_tfrecords(os.path.join(dataDir,'%s.tfrecords'%eventName),eventName)
	logging.info('%s is finished.'%eventName)
	

if __name__ == '__main__':
    main(sys.argv[1:])