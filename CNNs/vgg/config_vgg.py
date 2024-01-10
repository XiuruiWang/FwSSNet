import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "../data/cifar-10-batches-py", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", "vgg_cifar_new_11_2.csv", "Output directory where output such as acc are saved.")
tf.app.flags.DEFINE_string("log_save_path", "log/vgg_19",  '''log dir path ''')
tf.app.flags.DEFINE_string("model_save_path", "models/vgg19",  '''Output directory where output such as acc are saved. ''')
tf.app.flags.DEFINE_string("build_graph", "build_graph_FwSS",  '''type of model ''')


tf.app.flags.DEFINE_integer('total_epoch',500,
                            '''Total times that you want to train''')
tf.app.flags.DEFINE_integer('batch_size', 250, '''Train batch size''')
tf.app.flags.DEFINE_integer('image_size', 32, ''' size of image''')
tf.app.flags.DEFINE_integer('img_channels', 3, ''' channels of image''')
tf.app.flags.DEFINE_integer('iterations', 100, ''' iterations of train''')


tf.app.flags.DEFINE_integer('class_num', 10, '''output dim''')

tf.app.flags.DEFINE_float('weight_decay', 0.001, '''decay of weight''')
tf.app.flags.DEFINE_float('momentum_rate', 0.9, '''momentum_rate''')





