from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import math
from scipy import ndimage

import argparse
import os
import sys
import tempfile

from init_test import *
from mnist import *

import tensorflow as tf

from tensorflow.python.client import device_lib
# from tensorflow.python.client import timeline

import time

tf.logging.set_verbosity(tf.logging.ERROR)

tf.app.flags.DEFINE_integer('training_iteration', 20000,'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')

SAVED_MODEL = './SAVED_MODEL/'
DR_SAVED_MODEL = os.path.dirname(SAVED_MODEL)

#######################################################################
#                          Get available GPUs                         #
#######################################################################
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return[x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

#######################################################################
#                          Weigths & Biases                           #
#######################################################################
# Initilization with a slightly positive bias to avoid "dead neurons" (ReLu)
def weight_variable(shape, number):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=('weight_'+str(number)))


def bias_variable(shape, number):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=('bias_'+str(number)))


#######################################################################
#                         Convolution Modules                         #
#######################################################################
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#######################################################################
#                               Main                                  #
#######################################################################
def main(_):

  """
                             Check Args                              
  """
  if len(sys.argv) < 2:
    print('Usage: mnist_export.py --training_iteration=x '
      '--model_version=y')
    sys.exit(-1)
  if tf.app.flags.FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if tf.app.flags.FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)

  """
                    Create folder to store metadata                              
  """
  if not os.path.exists(DR_SAVED_MODEL):
    os.makedirs(DR_SAVED_MODEL)

  export_path_base = sys.argv[-1]
  export_path = os.path.join(
      tf.compat.as_bytes(SAVED_MODEL),
      tf.compat.as_bytes(str(tf.app.flags.FLAGS.model_version)))

  if os.path.exists(export_path):
    print('There is no saved_model.pb for this version number (%s).' % str(tf.app.flags.FLAGS.model_version))
    sys.exit(-1)  

  """
                              Load Data                               
  """
  print('\nLoading data...\n')
  mnist = load_dataset("mnist")


  """
                          Configure Profiling                               
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  sess = tf.InteractiveSession()

  """
                      Input Images & Output Classes                    
  """
  """
  a placeholder for our image data:
  None stands for an unspecified number of images
  784 = 28*28 pixel
  """
  x = tf.placeholder(tf.float32, [None, 784], "Placeholder_x")
  """
  y_ will be filled with the real values
  which we want to train (digits 0-9)
  for an undefined number of images
  """
  y_ = tf.placeholder('float', [None, 10], "Placeholder_y_")

  #######################################################################
  #                 Multilayer Convolutional Network                    #
  #######################################################################
  """
                              Build Graph                           
  """
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)

  """
                              Convolution #1                             
  """
  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32], 1)
    b_conv1 = bias_variable([32], 1)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  """
                               Pooling #1                              
  """
  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  """
                             Convolution #2                             
  """
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64], 2)
    b_conv2 = bias_variable([64], 2)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  """
                               Pooling #2                              
  """
  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  """
                                Dense #1                               
  """
  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features (neurons) to 
  # allow processing on the entire image.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024], 3)
    b_fc1 = bias_variable([1024], 3)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  """
                                 Dropout                                
  """
  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features. To reduce overfitting. On during training and off during testing.
  with tf.name_scope('dropout_'):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  """
                                Dense #2                               
  """
  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10], 4)
    b_fc2 = bias_variable([10], 4)
    logits = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

  with tf.name_scope('get_pred_'):
    tf.argmax(input=logits, axis=1, name='classe')
    y = tf.nn.softmax(logits, name='perc')


  """  
                             Loss Function                             
  """
  with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('loss', cross_entropy)

  """
    Compute gradient for a loss and apply gradients to variables     
  """
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  """
                         Store predictions     
  """
  with tf.name_scope('classes_'):
    values, indices = tf.nn.top_k(y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(10)]))
    prediction_classes = table.lookup(tf.to_int64(indices), name="predict")

  """
                           Evaluate model                         
  """
  with tf.name_scope('accuracy_'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name='predict')
    tf.summary.scalar('accuracy', accuracy)

  """
                            Save graph     
  """
  merged = tf.summary.merge_all()
  graph_location = tempfile.mkdtemp(
    prefix='graph_for_model_' + str(tf.app.flags.FLAGS.model_version) + '_',
    dir=SAVED_MODEL)
  print('\nSaving graph to %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  
  print('\nTraining model on GPU %s...\n' % get_available_gpus())


  sess.run(tf.global_variables_initializer())

  """
                     Train model by repeating train_step               
  """
  start_time_train = time.time()
  for i in range(tf.app.flags.FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    if i % 1000 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('Step %d/%d, Training accuracy %g' % (i, tf.app.flags.FLAGS.training_iteration, train_accuracy))
      summary, _ = sess.run([merged, train_step],
        feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5},
        options=run_options,
        run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      """
                          Tracing timeline on chrome://tracing/              
      """
      # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      # chrome_trace = fetched_timeline.generate_chrome_trace_format()
      # with open('timeline_%d_step_%d.json' % (tf.app.flags.FLAGS.training_iteration, i), 'w') as f:
      #   f.write(chrome_trace)
    else:
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      train_writer.add_summary(summary, i)
  duration_train = time.time() - start_time_train
  train_writer.close()      

  print('\nDone training! (%.3f sec)' % (duration_train))    

  """
                                  Evaluate model               
  """
  print('\nStarting evaluation...\n')
  start_time_eval= time.time()
  acc = sess.run(accuracy, feed_dict={
      x: mnist.test.images, 
      y_: mnist.test.labels, 
      keep_prob: 1.0})
  duration_eval = time.time() - start_time_eval
  print('Evaluation accuracy: %g' % (acc))
  print('\nDone evaluating! (%.2f msec)\n' % (duration_eval*1000.0))  


  """
                                   Export model                         
  """
  print('Exporting trained model to %s/...' % export_path.decode())
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  
  builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], None, None)

  builder.save()

  print('\nDone exporting!\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)