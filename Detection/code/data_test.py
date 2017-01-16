import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import logging

# import resfcn_input
# import resfcn_config as config

def main(argv=None):  # pylint: disable=unused-argument
	# images,labels,pos_ws,neg_ws,scan_names,z_indexs = resfcn_input.inputs()
	# print('inputs initialize.')
	data_dir = '/data-disk/gujy/Face/WF/tf_records'
	recordName = os.listdir(data_dir)[int(argv[0])]

	logFile = os.path.join('%s.log'%recordName)

	logging.basicConfig(level=logging.DEBUG,
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
		datefmt='%m-%d-%H:%M:%S',
		filename=logFile,
		filemode='w')

	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(tf.train.string_input_producer([os.path.join(data_dir,recordName)]))
	features = tf.parse_single_example(serialized_example,
		features={
		'image': tf.VarLenFeature(dtype=tf.float32),
		'image_shape':tf.FixedLenFeature(shape=[3],dtype=tf.int64),
		'label': tf.VarLenFeature(dtype=tf.float32),
		'label_shape':tf.FixedLenFeature(shape=[3],dtype=tf.int64),
		'name': tf.FixedLenFeature(shape=[],dtype=tf.string)
		})

	image = tf.cast(features['image'],tf.float32)
	image_shape = tf.cast(features['image_shape'],tf.int32)
	image = tf.sparse_tensor_to_dense(image)
	image = tf.reshape(image,image_shape)

	label = tf.cast(features['label'],tf.float32)
	label_shape = tf.cast(features['label_shape'],tf.int32)
	label = tf.sparse_tensor_to_dense(label)
	label = tf.reshape(label,label_shape)

	name = tf.cast(features['name'],tf.string)
	
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True)))
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	for i in range(1000):
		try:
	# 	for i in range(100):
	# 		sess_output = sess.run([images,labels,pos_ws,neg_ws,scan_names,z_indexs])
	# 		#sess_output = sess.run([image,label,pos_w,neg_w,scan_name,z_index,image_shape,label_shape])
	# 		#sess_output[0]=np.reshape(sess_output[0],sess_output[-2])
	# 		#sess_output[1]=np.reshape(sess_output[1],sess_output[-1])
	# 		plt.subplot(121)
	# 		plt.imshow(sess_output[0][0,:,:,int(config.IMAGE_SIZE[-1]/2)])
	# 		plt.subplot(122)
	# 		plt.imshow(sess_output[1][0,:,:,int(config.LABEL_SIZE[-1]/2)])
	# 		plt.show()
	# 		print(sess_output[2])
	# 		print(sess_output[3])
			IMAGE,LABEL,NAME = sess.run([image,label,name])
			print(NAME.decode())
		except:
			logging.error('%d'%i)
			break
		finally:
			coord.request_stop()
	coord.join(threads)
	sess.close()

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '4'
  main(sys.argv[1:])

# img = 5*np.random.randn(5,5)+5;
# print(img)
# image = tf.constant(img)
# a = tf.clip_by_value(image,4,6)
# b = tf.image.adjust_brightness(a,-4)


# print('Begin.')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# init = tf.initialize_all_variables()
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
# sess.run(init)
# print(sess.run([a,b]))

