import tensorflow as tf
import numpy      as np
import math

IMAGE_WIDTH  = 160
IMAGE_HEIGHT = 120
IMAGE_DEPTH  = 1
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
NUM_CLASSES  = 2

STEPS         = 50000
STEP_PRINT    = 100
STEP_VALIDATE = 100
LEARN_RATE    = 0.0014
DECAY_RATE    = 0.4
BATCH_SIZE    = 5

def read_my_list( minId, maxId, folder ):
    """ create list with train/no and train/go from 1 to maxid
        max maxId = 50000
    """ 

    filenames = []
    labels    = []
    #labels = np.zeros( ( ( maxId - minId ) * 2, 2 ) )
    for num in range( minId, maxId ):

        filenames.append( "/media/boss/2C260F93260F5CE8/tensor/" + folder + "/go/" + str( num ) + ".jpg" )
        #labels[ ( num - minId ) * 2 ][ 1 ] = 1
        labels.append( int( 1 ) )

        filenames.append( "/media/boss/2C260F93260F5CE8/tensor/" + folder + "/no/" + no_go_name( num ) + ".jpg" )
        #labels[ ( ( num - minId ) * 2 ) + 1 ][ 0 ] = 1
        labels.append( int( 0 ) )

        # return list with all filenames
    print( "label: " + str( len( labels ) ) )
    print( "image: " + str( len( filenames ) ) )
    return filenames, labels

def no_go_name( id ):

    # create string where id = 5 becomes 00005

    ret = str( id )
    while ( len( ret ) < 5 ):
      ret = "0" + ret;

    return ret;

# Create model
def conv_net(x):

    img_width  = IMAGE_WIDTH
    img_height = IMAGE_HEIGHT
    img_depth  = IMAGE_DEPTH

    weights    = tf.Variable( tf.random_normal( [ img_width * img_height * img_depth, NUM_CLASSES ] ) )
    biases     = tf.Variable( tf.random_normal( [ NUM_CLASSES ] ) )

    # softmax layer
    out        = tf.add( tf.matmul( x, weights ), biases )
    return out 

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    print( "read file "  )
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg( file_contents, channels = 1 )

    example = tf.reshape( example, [ IMAGE_PIXELS ] )
    example.set_shape( [ IMAGE_PIXELS ] )

    example = tf.cast( example, tf.float32 )
    example = tf.cast( example, tf.float32 ) * ( 1. / 255 ) - 0.5

    label = tf.cast( label, tf.int64 )

    label = tf.one_hot( label, 2, 0, 1 )
    label = tf.cast( label, tf.float32 )

    print( "file read " )
    return  example, label

with tf.Session() as sess:

    ########################################
    # get filelist and labels for training
    image_list, label_list = read_my_list( 501, 50000, "train" )

    # create queue for training
    input_queue = tf.train.slice_input_producer( [ image_list, label_list ],
                                                num_epochs = 100,
                                                shuffle = True )

    # read files for training
    image, label = read_images_from_disk( input_queue )

    # `image_batch` and `label_batch` represent the "next" batch
    # read from the input queue.
    image_batch, label_batch = tf.train.batch( [ image, label ], batch_size = BATCH_SIZE )

    # input output placeholders

    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # create the network
    y = conv_net( x )

    # loss
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( y, y_) )

    learning_rate = tf.placeholder(tf.float32, shape=[])

    # train step
    train_step   = tf.train.AdamOptimizer( 1e-3 ).minimize( cost )


    ########################################
    # get filelist and labels for validation
    image_list_test, label_list_test = read_my_list( 1, 500, "validation" )

    # create queue for validation
    input_queue_test = tf.train.slice_input_producer( [ image_list_test, label_list_test ],
                                                shuffle=True )

    # read files for validation
    image_test, label_test = read_images_from_disk( input_queue_test )

    # `image_batch_test` and `label_batch_test` represent the "next" batch
    # read from the input queue test.
    image_batch_test, label_batch_test = tf.train.batch( [ image_test, label_test ], batch_size=200 )

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    sess.run(init)

    # N.B. You must run this function before `sess.run(train_step)` to
    # start the input pipeline.
    #tf.train.start_queue_runners(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(STEPS):
        # No need to feed, because `x` and `y_` are already bound to
        # the next input batch.   
        if i % STEP_PRINT == 0:
            LEARN_RATE = LEARN_RATE * DECAY_RATE
            print( str( i ) + " " + str( LEARN_RATE ) )

        if i % STEP_VALIDATE == 0:

            imgs, lbls = sess.run([image_batch_test, label_batch_test])

            print(sess.run(accuracy, feed_dict={
                    x: imgs,
                    y_: lbls}))

        imgs, lbls = sess.run([image_batch, label_batch])

        sess.run(train_step, feed_dict={
         x: imgs,
         y_: lbls}) 
#         ,learning_rate:LEARN_RATE})      

    imgs, lbls = sess.run([image_batch_test, label_batch_test])

    print(sess.run(accuracy, feed_dict={
         x: imgs,
         y_: lbls}))

    coord.request_stop()
    coord.join(threads)
