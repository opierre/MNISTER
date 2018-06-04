import cv2
import numpy as np
import math
from scipy import ndimage

import os
import sys
import argparse

import numpy

from init_test import *
from mnist import *

import tensorflow as tf

from math import *

import time

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_integer('nb_images', 4, 'number of images to deal with.')
tf.app.flags.DEFINE_integer('nb_batchs', 0, 'batchs of images to deal with.')
tf.app.flags.DEFINE_boolean('digits_per_img', True, 'many digits per image (True) or just one digit (False).')
tf.app.flags.DEFINE_boolean('print_prob', True, 'print probabilities.')

SAVED_MODEL = './SAVED_MODEL/'

#######################################################################
#                       Operation on pictures                         #
#######################################################################  
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


#######################################################################
#                          Tools Functions                            #
####################################################################### 
def maxs(x):
    first_max = np.amax(x) 
    y = np.argmax(x)
    x = np.delete(x, y)
    new_max = np.amax(x) 
    new_index = np.argmax(x)
    if (new_index >= y):
      new_index += 1
    return first_max, new_max, new_index

def get_labels(file):
    labels = []
    with open(file, 'r') as f:
      for line in f:
        labels.append(int(line))
    return labels

def mean_average(file, predictions, nb_img):
    average = 0
    with open(file, 'r') as f:
      i = 0
      for line in f:
        if (i < nb_img):
          if (int(line) == predictions[i]):
            average += 1
          i += 1
        elif (nb_img == 0):
          if (int(line) == predictions[i]):
            average += 1
          i += 1
        else:
          break
    return (float(average/len(predictions))*100.0)

#######################################################################
#                               Main                                  #
#######################################################################
def main(_):

  """
                             Check Args                              
  """
  if len(sys.argv) < 4:
    print('Usage: mnist_load.py --model_version=x --digits_per_img=True/False '
      '--nb_images=y [--print_prob=True/False] [--nb_batchs=z]')
    sys.exit(-1)
  if tf.app.flags.FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)

  print("------------------------------------------------------------------------------------------")
  print('Program:\t\t\tmnist_load.py\nModel version:\t\t\t%d\nMore than one digit per image:\t%r\n'
    'Number of images:\t\t%d\nPrint results:\t\t\t%r\nNumer of batchs:\t\t%d'
    % (tf.app.flags.FLAGS.model_version, tf.app.flags.FLAGS.digits_per_img, 
    tf.app.flags.FLAGS.nb_images, tf.app.flags.FLAGS.print_prob,tf.app.flags.FLAGS.nb_batchs))
  print("------------------------------------------------------------------------------------------")
  
  """
                    Check folder to load metadata                              
  """
  export_path_base = sys.argv[-1]
  export_path = os.path.join(
      tf.compat.as_bytes(SAVED_MODEL),
      tf.compat.as_bytes(str(tf.app.flags.FLAGS.model_version)))

  if not os.path.exists(export_path):
    print('There is no saved_model.pb for this version number (%s).' % str(tf.app.flags.FLAGS.model_version))
    sys.exit(-1)

  print('\nLoading data...')
  
  if (tf.app.flags.FLAGS.digits_per_img == True):
  
    img = cv2.imread("./images/own_"+str(0)+".png")

    gray = cv2.imread("./images/own_"+str(0)+".png", 0)

    im_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, im_th = cv2.threshold(im_gray, 170, 255, cv2.THRESH_BINARY_INV)

    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images = np.zeros((len(ctrs),784))

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    i = 0 
    for rect in rects:

      cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 3) 

      leng = int(rect[3] * 1.6)
      pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
      pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
      roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

      roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
      roi = cv2.dilate(roi, (3, 3))

      cv2.imwrite("./images/image_"+str(i)+".png", roi)

      flatten = roi.flatten() / 255.0

      images[i] = flatten
      i+=1

  else:

    images = np.zeros((tf.app.flags.FLAGS.nb_images,784))

    i = 0
    for no in range(0, tf.app.flags.FLAGS.nb_images):
      gray = cv2.imread("./images/own_"+str(no)+".png", 0)

      gray = cv2.resize(255-gray, (28, 28))

      (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

      while np.sum(gray[0]) == 0:
        gray = gray[1:]

      while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

      while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

      while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

      rows,cols = gray.shape

      if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
      else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols,rows))

      colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
      rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
      gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

      shiftx,shifty = getBestShift(gray)
      shifted = shift(gray,shiftx,shifty)
      gray = shifted

      cv2.imwrite("./images/image_"+str(no)+".png", gray)

      flatten = gray.flatten() / 255.0

      images[i] = flatten
      i+=1

  # images = np.reshape(images[0], [-1, 784])

  # config=tf.ConfigProto(log_device_placement=True)
  
  with tf.Session() as sess:
    print('\nLoading trained model from %s/...\n' % export_path.decode())
    saved_model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_path)
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("Placeholder_x:0")
    prob = graph.get_tensor_by_name("dropout_/keep_prob:0")
    classes = graph.get_operation_by_name('get_pred_/classe').outputs[0]
    perc = graph.get_operation_by_name('get_pred_/perc').outputs[0]
    sess.run(tf.tables_initializer())

    print('Testing...\n')

    if (tf.app.flags.FLAGS.print_prob == True):
      print("------------------------------------------------------------------------------------------")
      print(" Digit | Prediction | Error | Probability | Processing Time | 2nd Prediction | Probability")
      print("------------------------------------------------------------------------------------------")

    images_save = images
    digits = get_labels("./images/labels.txt")

    if (tf.app.flags.FLAGS.nb_batchs != 0):

      y_save = np.zeros((2, len(images_save)))
      new_index = np.zeros(len(images_save))
      new_max = np.zeros(len(images_save))
      duration_test = []

      start_step = 0
      for k in range(1, ceil(len(images_save)/tf.app.flags.FLAGS.nb_batchs)+1): 
        
        if ((k*tf.app.flags.FLAGS.nb_batchs) > len(images_save)):
          stop_step = len(images_save)
        else:
          stop_step = k*tf.app.flags.FLAGS.nb_batchs

        images = np.reshape(images_save[start_step:stop_step], [-1, 784])
        
        start_time_test = time.time()
        y = sess.run((classes, perc),feed_dict={x: images, prob:1.0})
        duration_test.append(time.time() - start_time_test)

        y_save[0][start_step:stop_step] = y[0]

        for j in range(0, tf.app.flags.FLAGS.nb_batchs):          
          if (j >= len(y[1])):
            break
          else: 
            first_max, new_max[j+start_step], new_index[j+start_step] = maxs(y[1][j])
            y_save[1][j+start_step] = first_max

        start_step += tf.app.flags.FLAGS.nb_batchs

      if (tf.app.flags.FLAGS.print_prob == True):
        for t in range (1, len(images_save)+1):
          if (digits[t-1] == y_save[0][t-1]): 
            if ((t % tf.app.flags.FLAGS.nb_batchs == 0) & (t != len(images_save))):
              print('   %d   |     %d\t    |       |  %f   |   %.3f msec    |      %d\t     |  %f' % 
                (digits[t-1], y_save[0][t-1], y_save[1][t-1], (duration_test[int(t/tf.app.flags.FLAGS.nb_batchs)-1]*1000.0), new_index[t-1], new_max[t-1]))
              print('------------------------------------------------------------------------------------------')
            elif (t == len(images_save)):
              print('   %d   |     %d\t    |       |  %f   |   %.3f msec    |      %d\t     |  %f' % 
                (digits[t-1], y_save[0][t-1], y_save[1][t-1], (duration_test[-1]*1000.0), new_index[t-1], new_max[t-1]))
              print('------------------------------------------------------------------------------------------')            
            else:
              print('   %d   |     %d\t    |       |  %f   |                 |      %d\t     |  %f' % 
                (digits[t-1], y_save[0][t-1], y_save[1][t-1], new_index[t-1], new_max[t-1]))
              print('------------------------------------------------------------------------------------------')
          else:
            if ((t % tf.app.flags.FLAGS.nb_batchs == 0) & (t != len(images_save))):
              print('   %d   |     %d\t    |*******|  %f   |   %.3f msec    |      %d\t     |  %f' % 
                (digits[t-1], y_save[0][t-1], y_save[1][t-1], (duration_test[int(t/tf.app.flags.FLAGS.nb_batchs)-1]*1000.0), new_index[t-1], new_max[t-1]))
              print('------------------------------------------------------------------------------------------')
            elif (t == len(images_save)):
              print('   %d   |     %d\t    |*******|  %f   |   %.3f msec    |      %d\t     |  %f' % 
                (digits[t-1], y_save[0][t-1], y_save[1][t-1], (duration_test[-1]*1000.0), new_index[t-1], new_max[t-1]))
              print('------------------------------------------------------------------------------------------')            
            else:
              print('   %d   |     %d\t    |*******|  %f   |                 |      %d\t     |  %f' % 
                (digits[t-1], y_save[0][t-1], y_save[1][t-1], new_index[t-1], new_max[t-1]))
              print('------------------------------------------------------------------------------------------')            
      
      if (tf.app.flags.FLAGS.digits_per_img == True):
        j = 0
        for rect in rects: 
          if (digits[j] == y_save[0][j]): 
            cv2.putText(img, str(int(y_save[0][j])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)    
          else:
            cv2.putText(img, str(int(y_save[0][j])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)      
          j += 1

        cv2.imwrite("./images/image_finish.png", img)
        img = cv2.imread("./images/image_finish.png")
        cv2.namedWindow("Resulting Image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Resulting Image", img)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
      
      print('\nTesting done!\n') 


      print("------------------------------------------------------------------------------------------")
      if (tf.app.flags.FLAGS.digits_per_img == False):
        print("Mean Accuracy:        \t %.1f %%" % (mean_average("./images/labels.txt", y_save[0], tf.app.flags.FLAGS.nb_images)))
      else:
        print("Mean Accuracy:        \t %.1f %%" % (mean_average("./images/labels.txt", y_save[0], 0)))  
      print("Total Processing Time:\t %.3f msec" % (np.sum(duration_test)*1000.0))
      print("------------------------------------------------------------------------------------------")

    else:  

      y_save = np.zeros((2,len(images_save)))
      duration_test = []

      for k in range(0, len(images_save)):

        images = np.reshape(images_save[k], [-1, 784])

        start_time_test = time.time()
        y = sess.run((classes, perc),feed_dict={x: images, prob:1.0})
        duration_test.append(time.time() - start_time_test)

        y_save[0][k] = y[0]

        first_max, new_max, new_index = maxs(y[1])
        y_save[1][k] = first_max
        if (tf.app.flags.FLAGS.print_prob == True):
          if (digits[k] == y_save[0][k]): 
            print('   %d   |     %d\t    |       |  %f   |   %.3f msec    |      %d\t     |  %f' % 
              (digits[k], y[0], first_max, duration_test[k]*1000.0, new_index, new_max))
            print('------------------------------------------------------------------------------------------')
          else:
            print('   %d   |     %d\t    |*******|  %f   |   %.3f msec    |      %d\t     |  %f' % 
              (digits[k], y[0], first_max, duration_test[k]*1000.0, new_index, new_max))
            print('------------------------------------------------------------------------------------------')            

      if (tf.app.flags.FLAGS.digits_per_img == True):
        j = 0
        for rect in rects: 
          if (digits[j] == y_save[0][j]): 
            cv2.putText(img, str(int(y_save[0][j])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)    
          else:
            cv2.putText(img, str(int(y_save[0][j])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)      
          j += 1

        cv2.imwrite("./images/image_finish.png", img)
        img = cv2.imread("./images/image_finish.png")
        cv2.namedWindow("Resulting Image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Resulting Image", img)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
      
      print('\nTesting done!\n') 

      print("------------------------------------------------------------------------------------------")
      if (tf.app.flags.FLAGS.digits_per_img == False):
        print("Mean Accuracy:        \t %.1f %%" % (mean_average("./images/labels.txt", y_save[0], tf.app.flags.FLAGS.nb_images)))
      else:
        print("Mean Accuracy:        \t %.1f %%" % (mean_average("./images/labels.txt", y_save[0], 0))) 
      print("Total Processing Time:\t %.3f msec" % (np.sum(duration_test)*1000.0))
      if (tf.app.flags.FLAGS.nb_images > 1 & (tf.app.flags.FLAGS.digits_per_img == False)):
        print("\tBake:   \t %.3f msec" % (duration_test[0]*1000.0))
        compute_time = []
        compute_time = duration_test[1:]
        print("\tCompute:\t %.3f msec (%.3f msec/digit)" % ((np.sum(compute_time)*1000.0), (np.sum(compute_time)*1000.0)/len(compute_time)))
      print("------------------------------------------------------------------------------------------")

    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)