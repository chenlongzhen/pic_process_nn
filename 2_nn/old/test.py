
import tensorflow as tf


def get_filename_list(filename_list):
    ''' get filenamelist type: path,label'''
    filepath_list=[]
    label_list=[]
    count = 0
    with open(filename_list,'w') as wfile:
        for line in wfile:
            segs = line.strip().split(',')
            if len(segs) != 2:
                print('[error] wrongtype:{}'.format(line))
                continue
            count += 1
            filepath_list.append(segs[0])
            label_list.append(segs[1])
    return filepath_list,label_list,count


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# images and labels array as input
def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
      raise ValueError("Images size %d does not match label size %d." %
                       (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(rows),
          'width': _int64_feature(cols),
          'depth': _int64_feature(depth),
          'label': _int64_feature(int(labels[index])),
          'image_raw': _bytes_feature(image_raw)}))
      writer.write(example.SerializeToString()

if __name__ == "__main__":
    filename_list=sys.argv[1]
    file_path,label,num = get_filename_list(filename_list)

    filename_queue = tf.train.string_input_producer(file_path) #  list of files to read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("value\n")
    print(value)
    
    my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
    convert_to(my_img,lables,"testdata")
    
    init_op = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init_op)
    
        # Start populating the filename queue.
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
    #    for i in range(2): #length of your filename list
    #        image = my_img.eval() #here is your image Tensor :) 
    #        print(image.shape)
    #    #Image.show(Image.fromarray(np.asarray(image)))
        
        coord.request_stop()
        coord.join(threads)
    
    
