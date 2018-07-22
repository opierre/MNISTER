# MNISTER

Handwritten digits recognition using TensorFlow framework from Google

## Getting Started

Follow these instructions to get a copy of the project and make it run on your local machine (only tested with Ubuntu 16.04/18.04). 

### Prerequisites

You need to have Python 3.4+ and Pip3 installed on your Ubuntu machine. To check if it is correctly installed:

```bash
$ python3 -V
$ pip3 -V
```

### Installing

Install [TensorFlow](http://www.tensorflow.org/install/install_linux) for your GPU with the pip3 method and check your configuration by running the Hello World example:

```bash
$ python3
```
```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello from TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

If the system outputs your Hello World sentence as the following one, then you are ready to run the MNIST example:

```bash
'Hello from TensorFlow!'
```

## Running the tests

Before running the tests, you have to make sure you have the following directory tree:

```
./WORKSPACE
	|-- images
	|	|-- labels.txt
	|	|-- own_0.png
	|	|-- own_1.png
	|	|-- own_2.png
	|	|-- own_3.png
	|	|-- own_4.png
	|	|-- own_5.png
	|	|-- own_6.png
	|	|-- own_7.png
	|	|-- own_8.png
	|	|-- own_9.png
	|	|-- t10k-images-idx3-ubyte.gz
	|	|-- t10k-labels-idx1-ubyte.gz
	|	|-- train-images-idx3-ubyte.gz
	|	|-- train-labels-idx1-ubyte.gz
	|-- scripts
	|	|-- cpufreq.sh
	|	|-- launch.sh
	|-- base.py
	|-- init_test.py
	|-- mnist.py
	|-- mnist_export.py
	|-- mnist_load.py
	|-- README.md	
```

Then, you can run [cpufreq.sh](https://github.com/polivier2/MNISTER/blob/master/scripts/cpufreq.sh) according to the number of CPU cores you have. This script sets the performance mode over all CPU cores and stop the load balancing.

### Exporting your neural network

In order to train the MNIST neural network on your GPU, you should run:

```bash
$ python3 mnist_export.py --training_iteration=20000 --model_version=1
```

[mnist_export.py](https://github.com/polivier2/MNISTER/blob/master/mnist_export.py) contains all MNIST layers, with training and testing modes. This file is written according to the [Deep MNIST tutorial](http://www.tensorflow.org/versions/r1.4/get_started/mnist/pros). It also exports your trained model and save it as *saved_model.pb* in the SAVED_MODEL/ folder just created. At this point, you have also exported graph variables in the folder named as your model_version and the graph definition:

```
./SAVED_MODEL
	|-- 1
	|   |-- variables
	|	   |-- variables.data
	|	   |-- variables.index
	|   |-- saved_model.pb
	|-- graph_for_model_1_xxxxxxxx
	|   |-- events.out.tfevents.xxxxxxxxxx.user
```

If you want to visualize the graph and other metadata in Mozilla Firefox browser, you can run:

```bash
$ tensorboard --logdir=/WORKSPACE/SAVED_MODEL/
```

### Loading your neural network and running inference with your own handwritten digit dataset

Once your model has been trained and exported, you may want to load it without using the TensorFlow Serving API. [mnist_load.py](https://github.com/polivier2/MNISTER/blob/master/mnist_load.py) allows you to load your *saved_model.pb* locally and feed the input with your own handwritten digits images. In order to run inference, you have to store your image(s) in the images/ folder and name it *own_X.png*, X following the ascending order and starting from 0. Then, you have two different use cases:

* **One picture with many handwritten digits:** for both use cases, you have to write your own *labels.txt* in the images/ folder. According to the pre-processing of this picture, you should start writting it from the bottom to the top of the image and label each digit. This step allows you to detect a possible error when running inference and thus to measure inference accuracy. Once labels are ready, you can run:

```bash
$ python3 mnist_load.py --model_version=1 --digits_per_img=True --nb_images=1
``` 

If the system outputs *Testing done!*, you are now able to visualize *image_finish.png* which has been written in the images/ folder thanks to OpenCV for Python. Otherwise, you must adjust the threshold value (T) in:

```python
ret, im_th = cv2.threshold(im_gray, T, 255, cv2.THRESH_BINARY_INV) #line 130
```

This pre-processing step on your own dataset is neccessary to detect handwritten digits contours and resize each detected digit as a 28x28 image. Then, you dataset will look like the MNIST training dataset and can be sent to the trained neural network. This part has been written thanks to [Bikramjot Hanzra digit-recognition project](http://github.com/bikz05/digit-recognition/blob/master/performRecognition.py). 

* **Many pictures with one handwritten digit:** for both use cases, you have to write your own *labels.txt* in the images/ folder. According to the pre-processing of these picture, you should start writting it from *own_0.png* to the last one of your dataset and label each digit. This step allows you to detect a possible error when running inference and thus to measure inference accuracy. Once labels are ready, you can run:

```bash
$ python3 mnist_load.py --model_version=1 --digits_per_img=False --nb_images=9
``` 

If the system outputs *Testing done!*, you are now able to visualize all *image_X.png* which have been written in the images/ folder thanks to OpenCV for Python. Otherwise, you must adjust the threshold value (T) in:

```python
(thresh, gray) = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

This pre-processing step on your own dataset is neccessary to detect handwritten digits and resize each detected digit as a 28x28 image. Then, you dataset will look like the MNIST training dataset and can be sent to the trained neural network. This part has been written thanks to [opensourcesblog project](https://github.com/opensourcesblog/tensorflow-mnist/blob/master/step2.py).

For each case, you can add these following options:

* Dealing images with batchs: number of batchs should always be lesser than number of images when *--digits_per_img=False*

```bash
$ python3 mnist_load.py --model_version=1 --digits_per_img=False --nb_images=9 --nb_batchs=4
``` 

* Printing probabilities: you can choose if you want to print all results per image/batchs or just print the final result

```bash
$ python3 mnist_load.py --model_version=1 --digits_per_img=False --nb_images=9 --print_prob=False 
``` 

## Author

* **Pierre OLIVIER** - [polivier2](https://github.com/polivier2)
