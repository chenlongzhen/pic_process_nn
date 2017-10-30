#encoding=utf-8
#author:clz
#add fullconnection layyer
import math
import tensorflow as tf
def add_layer(inputs, in_size, out_size, n_layer,activation_function=None):
    """Build the fullconnect layyer model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      in_size: Size of the input units (feature num).
      out_size: Size of the this layyer's units.
 
    Returns:
      outputs: this layyer output tensor
    """
    # add one more layer and return the output of this layer
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(
                tf.truncated_normal([in_size,out_size],
                                    stddev = 1.0 / math.sqrt(float(in_size)))
                )
            tf.histogram_summary(layer_name+'/weights',Weights)
    
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]), name='b')
            tf.histogram_summary(layer_name+'/biases',biases)
    
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
    
        tf.histogram_summary(layer_name+'/outputs',outputs)
    
        return outputs

def get_loss(outputs, labels):
    """Calculat the loss from the logits and the labels.

    Args: 
        outputs: tensor,float - [batch_size, Num_classes].
        labels:  Labels tensor, int32.

    returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=outputs, name='xentropy')
    return tf.reduce_mean(cross_entropy, name="xentropy_mean")

def training(loss, learning_rate, optimizer=tf.train.GradientDescentOptimizer):
    """Sets up the training Ops.
  
    Creates a summarizer to track the loss over time in TensorBoard.
  
    Creates an optimizer and applies the gradients to all trainable variables.
  
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
  
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
  
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    #tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = optimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(outputs, labels):
    """Evaluate the quality of the logits ad predicting the label

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
  #  classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
    correct = tf.nn.in_top_k(outputs, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


