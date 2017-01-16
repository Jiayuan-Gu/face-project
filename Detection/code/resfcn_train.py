from datetime import datetime
import time
import os.path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import resfcn_config as config
from resfcn_config import logger
import resfcn_model as resfcn
import resfcn_input

def train():
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.constant(config.learning_rate)

  # Get images and labels for FCN.
  images,labels,pos_ws,neg_ws,names = resfcn_input.inputs()
  print('inputs initialize.')

  # Build a Graph that computes the score map predictions from the
  # inference model.
  preds = resfcn.inference(images)
  # tf.image_summary('preds',preds[:,:,:,int(config.LABEL_SIZE[-1]/2)],max_images=int(config.batch_size/2))
  print('inference initialize.')

  # Calculate loss.
  loss, pos_loss, neg_loss = resfcn.loss(preds, labels, pos_ws, neg_ws)
  print('loss initialize.')

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  train_op = resfcn.train(loss, global_step,learning_rate,optimizer=config.optimizer)
  print('train initialize.')

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables(),max_to_keep=100,name='saver')
  print('saver initialize.')

  # Build the summary op
  summary_loss_op = tf.merge_all_summaries(key='LOSS_OP')
  summary_param_op = tf.merge_all_summaries(key='PARAM_OP')

  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  # Start running operations on the Graph.
  sess = tf.Session(config=config.session_config)
  sess.run(init)
  print('session initialize.')

  if config.resume:
    if not config.restore_path:
      restore_path = tf.train.latest_checkpoint(os.path.join(config.train_dir,'checkpoint'))
      if not restore_path:
        print('no checkpoint to continue.')
        sys.exit(1)
      saver.restore(sess,restore_path)
      logger.info('continue from %s',restore_path)
    else:
      saver.restore(sess,config.restore_path)
      logger.info('global step is set to %d',sess.run(tf.assign(global_step,0)))
      logger.info('learning rate is set to %.3f',sess.run(learning_rate))
      logger.info('restart from %s',config.restore_path)
      optimizer_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"scope/prefix/for/optimizer")
      sess.run(tf.initialize_variables(optimizer_scope))
    
  summary_writer = tf.train.SummaryWriter(os.path.join(config.train_dir,'summary'),sess.graph)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  print('training begins.')
  
  step = sess.run(global_step)

  try:
    while step<config.max_steps:
      sess_input = [train_op,loss]
      if step%config.num_steps_debug==0:
        sess_input.append(preds)
        sess_input.append(images)
        sess_input.append(labels)
        sess_input.append(names)
        sess_input.append(pos_loss)
        sess_input.append(neg_loss)
        sess_input.append(summary_loss_op)

      if step%config.num_steps_summary==0:
        sess_input.append(summary_param_op)
    
      start_time = time.time()
      sess_output = sess.run(sess_input)
      duration = time.time() - start_time
      assert not np.isnan(sess_output[1]),'Model with loss = nan.'

      num_examples_per_step = config.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f sec/batch)')
      print (format_str % (datetime.now().strftime('%H:%M:%S'), 
        step, sess_output[1], examples_per_sec, sec_per_batch))

      if step % config.num_steps_debug==0:
        format_str = ('epoch %d, pos_loss = %.8f, neg_loss = %.8f')
        print(format_str % (int(step/config.num_batches_per_epoch),
          sess_output[6],sess_output[7]))

        summary_writer.add_summary(sess_output[8],step)

        ind = int(step%(2*config.num_steps_debug)==0)*int(config.batch_size/2)
        plt.subplot(131)
        plt.imshow(128-128*sess_output[3][ind,:,:,:])
        plt.subplot(132)
        plt.imshow(sess_output[4][ind,:,:,int(config.LABEL_SIZE[-1]/2)])
        plt.title(sess_output[5][ind].decode())
        plt.subplot(133)
        plt.imshow(sess_output[2][ind,:,:,int(config.LABEL_SIZE[-1]/2)])
        # plt.show()
        plt.savefig(os.path.join(config.train_dir,'fig/pred%d.png'%step))

      # Summary the training process.
      if step % config.num_steps_summary == 0:
        summary_str = sess_output[-1]
        summary_writer.add_summary(summary_str,step)

      if step % config.num_steps_checkpoint == 0:
        checkpoint_path = os.path.join(config.train_dir,'checkpoint','model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      step=sess.run(global_step)

  except tf.errors.OutOfRangeError:
    print('Running %d steps.'%step)
  finally:
    coord.request_stop()
    
  coord.join(threads)
  sess.close()
  config.close_all()


def main(argv=None):  # pylint: disable=unused-argument
  os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
  train()


if __name__ == '__main__':
  tf.app.run()
