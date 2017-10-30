#encoding=utf-8

import sys
S_PATH=sys.path[0]
C_PATH=S_PATH + "/../1_pic_process"
DATA_PATH=S_PATH+"/../../data"
sys.path.append(C_PATH)
import convert
import time
import numpy as np
import tensorflow as tf
import argparse
import nnUtils

# Basic model parameters as external flags.
FLAGS = None

def preprocess_pic(file_path,name,output_directory, resize_height, resize_width):
    """
    file_path:file path of picpath and label info 
    name:tfrecordname's prefix name . ex: train
    resize_*: img resize.
    """
    convert.transform2tfrecord(file_path,name,output_directory,resize_height, resize_width)   

def placeholder_inputs(batch_size,image_size):
    """Generate placeholder variables to represent the input tensors.
  
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
  
    Args:
      batch_size: The batch size will be baked into both placeholders.
  
    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           image_size))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl): #nouse now!!!!!
    """Fills the feed_dict for training the given step.
  
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
  
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
  
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def pipeline(train_records,image_height=28,image_width=28,classes_num=10,units_num=10,batch_size=100):

    image_size = image_height*image_width*3

    # holder
    images_holder, labels_holder =  placeholder_inputs(batch_size,image_size)

    # add hidden layer 
    l1 = nnUtils.add_layer(images_holder,image_height*image_width*3, units_num, n_layer=1, activation_function=tf.nn.relu)

    # add output layer 
    prediction = nnUtils.add_layer(l1, units_num, classes_num, n_layer=2, activation_function=None)

    # loss
    loss =  nnUtils.get_loss(prediction,labels_holder)

    # train
    train_op =  nnUtils.training(loss, learning_rate=0.01, optimizer = tf.train.GradientDescentOptimizer)

    # eval
    eval_op =  nnUtils.evaluation(prediction, labels_holder)

    return images_holder,labels_holder,train_op, loss, eval_op

def run():

    # Input images and labels.
    images, labels = convert.inputs(train=FLAGS.train_records, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs, image_height = FLAGS.image_height, image_width = FLAGS.image_width)
    # pipeline
    images_holder,labels_holder, train_op, loss_op, eval_op = pipeline(FLAGS.train_records,
                                          image_height=FLAGS.image_height,
                                          image_width=FLAGS.image_width,
                                          classes_num=FLAGS.classes_num,
                                          batch_size=FLAGS.batch_size)

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
        #pichy = np.reshape(pichy,(-1,1))
        #print(pichx)
        #print(pichy)
        _, loss, evals = sess.run([train_op, loss_op, eval_op],
                                   feed_dict={images_holder:pichx,labels_holder:pichy})

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 100 == 0:
          #print('Step %d: loss = %.2f eval = $.2f (%.3f sec)' % (step, loss,evals,duration))
          print('Step {}: loss = {} eval = {} ({} sec)'.format(step, loss,evals,duration))
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for  %d steps.' % (step))
      pass
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    #Wait for threads to finish.
    coord.join(threads)
    sess.close()

def main(_):
    run()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_records',
      type=str,
      default='/data/clz_workspace/pic_process/data/train.tfrecords',
      help='tfRecords path.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
#  parser.add_argument(
#      '--max_steps',
#      type=int,
#      default=2000,
#      help='Number of steps to run trainer.'
#  )
  parser.add_argument(
      '--units_num',
      type=int,
      default=10,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--classes_num',
      type=int,
      default=10,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=0,
      help='epochs.'
  )
  parser.add_argument(
      '--image_height',
      type=int,
      default=28,
      help='image_height.'
  )
  parser.add_argument(
      '--image_width',
      type=int,
      default=28,
      help='image_width.'
  )
#  parser.add_argument(
#      '--input_data_dir',
#      type=str,
#      default='/tmp/tensorflow/mnist/input_data',
#      help='Directory to put the input data.'
#  )
#  parser.add_argument(
#      '--log_dir',
#      type=str,
#      default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
#      help='Directory to put the log data.'
#  )

  FLAGS, unparsed = parser.parse_known_args()
  argv = [sys.argv[0]] +unparsed
  main(argv)
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

#def run_training(trainfile):
#    """
#    Train MNIST for a number of steps.
#    trainsize is the tfrecordfile path.
#    """
#    # Input images and labels.
#    images, labels = convert.inputs(train=trainfile, batch_size=123,
#                            num_epochs=0)
#    print("===input in run_training====")
#    print(images)
#    print(labels)
#    #xs,ys = tf.train.batch([images, labels], batch_size=1)
#    xs = tf.placeholder(tf.float32, [None, 28*28*3], name='x_input')
#    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
#
#    # add hidden layer 
#    l1 = layyer.add_layer(xs, 28*28*3, 10, n_layer=1, activation_function=tf.nn.relu)
#    # add output layer 
#    prediction = layyer.add_layer(l1, 10, 1, n_layer=2, activation_function=None)
#
#    # the error between prediciton and real data
#    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                                        reduction_indices=[1]))
#    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
#    # The op for initializing the variables.
#    init_op = tf.initialize_all_variables()
#
#    # Create a session for running operations in the Graph.
#    sess = tf.Session()
#
#    # Initialize the variables (the trained variables and the
#    # epoch counter).
#    sess.run(init_op)
#
#    # Start input enqueue threads.
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#    try:
#      step = 0
#      while not coord.should_stop():
#        start_time = time.time()
#        pichx = sess.run(images)
#        pichy = sess.run(labels)
#        pichy = np.reshape(pichy,(-1,1))
#        _, loss_value = sess.run([train_step, loss],feed_dict={xs:pichx,ys:pichy})
#
#        duration = time.time() - start_time
#
#        # Print an overview fairly often.
#        if step % 100 == 0:
#          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
#                                                     duration))
#        step += 1
#    except tf.errors.OutOfRangeError:
#      print('Done training for  %d steps.' % (step))
#      pass
#    finally:
#      # When done, ask the threads to stop.
#      coord.request_stop()
#    #Wait for threads to finish.
#    coord.join(threads)
#    sess.close()
#
#if __name__ == "__main__":
#    file_path=sys.argv[1]
#    name=sys.argv[2]
#    output_directory=DATA_PATH
#    train_file="{}/{}.tfrecords".format(DATA_PATH,name)
#    resize_height=28
#    resize_width=28
#    #transform
#    convert.transform2tfrecord(file_path,name,output_directory,resize_height, resize_width)   
#    run_training(train_file)
