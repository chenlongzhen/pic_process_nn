# -*- coding: utf-8 -*-
############################################################################################
#Author  : clz
#Date    : 2017
#Function: image convert to tfrecords 
#############################################################################################

import tensorflow as tf
import numpy as np
import cv2,time
import os
import os.path
from PIL import Image

#参数设置
###############################################################################################
#train_file = 'train.txt' #训练图片
#name='train'      #生成train.tfrecords
#output_directory='./tfrecords'
#resize_height=28 #存储图片高度
#resize_width=28 #存储图片宽度
###############################################################################################
def load_file(examples_list_file):
    '''load train list file:
       type: whole\file\path label
    '''
    lines = np.genfromtxt(examples_list_file, delimiter=" ", names=['col1','col2'],dtype=[('col1', 'S120'), ('col2', 'i4')])
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example.decode('utf-8')) #byte2utf8
        labels.append(int(label)) #convert to int type for tf.train.Example type.
    print("[INFO] load {} lines data.".format(len(lines)))
    return examples, labels, len(lines)

def extract_image(filename,  resize_height, resize_width):
    '''use opencv to read img data.
    '''
    image = cv2.imread(filename)
    image = cv2.resize(image, (resize_height, resize_width))
    b,g,r = cv2.split(image)       
    rgb_image = cv2.merge([r,g,b])     
    return rgb_image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def transform2tfrecord(train_file, name, output_directory, resize_height, resize_width):
    ''' convert img to tf record and save to local.
    '''
    #check path
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    #load file path and label
    _examples, _labels, examples_num = load_file(train_file)
    #convert img to tfrecords and save.
    filename = output_directory + "/" + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        image = extract_image(example, resize_height, resize_width)
        if i % 1000 == 0:
            print('processing No.%d pic.' % (i))
            print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

#def disp_tfrecords(tfrecord_list_file):
#    filename_queue = tf.train.string_input_producer([tfrecord_list_file])
#    reader = tf.TFRecordReader()
#    _, serialized_example = reader.read(filename_queue)
#    features = tf.parse_single_example(
#        serialized_example,
# features={
#          'image_raw': tf.FixedLenFeature([], tf.string),
#          'height': tf.FixedLenFeature([], tf.int64),
#          'width': tf.FixedLenFeature([], tf.int64),
#          'depth': tf.FixedLenFeature([], tf.int64),
#          'label': tf.FixedLenFeature([], tf.int64)
#      }
#    )
#    image = tf.decode_raw(features['image_raw'], tf.uint8)
#    #print(repr(image))
#    height = features['height']
#    width = features['width']
#    depth = features['depth']
#    label = tf.cast(features['label'], tf.int32)
#    print(label)
#    init_op = tf.initialize_all_variables()
#    resultImg=[]
#    resultLabel=[]
#    with tf.Session() as sess:
#        sess.run(init_op)
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#        for i in range(2):
#            print(label.eval())
#            print(width.eval())
#            print(height.eval())
#            print(depth.eval())
#            image_eval = image.eval()
#            resultLabel.append(label.eval())
#            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
#            resultImg.append(image_eval_reshape)
#            pilimg = Image.fromarray(np.asarray(image_eval_reshape))
#            # pilimg.show()
#        coord.request_stop()
#        coord.join(threads)
#    return resultImg,resultLabel
#
#def read_tfrecord(filename_queuetemp):
#    filename_queue = tf.train.string_input_producer([filename_queuetemp])
#    reader = tf.TFRecordReader()
#    _, serialized_example = reader.read(filename_queue)
#    features = tf.parse_single_example(
#        serialized_example,
#        features={
#          'image_raw': tf.FixedLenFeature([], tf.string),
#          'height': tf.FixedLenFeature([], tf.int64),
#          'width': tf.FixedLenFeature([], tf.int64),
#          'depth': tf.FixedLenFeature([], tf.int64),
#          'label': tf.FixedLenFeature([], tf.int64)
#      }
#    )
#    image = tf.decode_raw(features['image_raw'], tf.uint8)
#    # image
#    #    image.set_shape([28,28,3])
#    #tf.reshape(image, [28, 28, 3])
#    # normalize
#    image = tf.cast(image, tf.float32) * (1. /255) - 0.5
#    # label
#    label = tf.cast(features['label'], tf.int32)
#    init_op = tf.initialize_all_variables()

def read_and_decode(filename_queuetemp,image_height, image_width):
    '''read tfrecords 
       return image and label tensors
    '''
    #filename_queue = tf.train.string_input_producer([filename_queuetemp])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queuetemp)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
  
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8) # raw img data
    image.set_shape([image_height*image_width*3]) # set img pixels
  
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
  
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    
    print("=======decode=======")
    print(image)
    print(label)
    return image, label


def inputs(train, batch_size=1000, num_epochs=0, image_height=28,image_width=28):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data. nouse. input filepath!!
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  #filename = os.path.join(FLAGS.train_dir,
  #                        TRAIN_FILE if train else VALIDATION_FILE)
  filename=train

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue,image_height,image_width)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
    print("[INFO] got tfrecords data, info:\nimage:{}\nsparse_labels:{}"
           .format(images,sparse_labels))

    return images, sparse_labels

######################################################################################
def add_layer(inputs, in_size, out_size, n_layer,activation_function=None):
    # add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.histogram_summary(layer_name+'/weights',Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.histogram_summary(layer_name+'/biases',biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )

        tf.histogram_summary(layer_name+'/outputs',outputs)

        return outputs


def run_training(trainfile):
    """Train MNIST for a number of steps."""
    # Input images and labels.
    images, labels = inputs(train=trainfile, batch_size=123,
                            num_epochs=0)
    print("===input in run_training====")
    print(images)
    print(labels)
    #xs,ys = tf.train.batch([images, labels], batch_size=1)
    xs = tf.placeholder(tf.float32, [None, 28*28*3], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # add hidden layer 
    l1 = add_layer(xs, 28*28*3, 10, n_layer=1, activation_function=tf.nn.relu) 
    # add output layer 
    prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None) 

    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # The op for initializing the variables.
    init_op = tf.initialize_all_variables()

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        pichx = sess.run(images)
        pichy = sess.run(labels)
        pichy = np.reshape(pichy,(-1,1))
        _, loss_value = sess.run([train_step, loss],feed_dict={xs:pichx,ys:pichy})

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 100 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for  %d steps.' % (step))
      pass
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

def test():
    train_file = 'train.txt' #训练图片
    name='train'      #生成train.tfrecords
    output_directory='./tfrecords'
    resize_height=28 #存储图片高度
    resize_width=28 #存储图片宽度
    train=output_directory+'/'+name+'.tfrecords'
    transform2tfrecord(train_file, name , output_directory,  resize_height, resize_width) #转化函数   
    #img,label=disp_tfrecords(output_directory+'/'+name+'.tfrecords') #显示函数
    #img,label=read_tfrecord(output_directory+'/'+name+'.tfrecords') #读取函数
    #img,label=read_and_decode(output_directory+'/'+name+'.tfrecords') #读取函数
    #img,label = inputs(train,10,10000)
    run_training(train) 

if __name__ == '__main__':
    test()
