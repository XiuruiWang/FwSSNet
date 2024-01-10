import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "data/CNAE.mat", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("data_name", "CNAE", "data column name")
tf.app.flags.DEFINE_string("output_dir", "./results/file_CNAE_10_10_change_paper.csv", "Output directory where output such as acc are saved.")
tf.app.flags.DEFINE_string("output_dir_model", "./tmp/temp-bn-save-CNAE", "Output directory where output such as model are saved.")
tf.app.flags.DEFINE_string("build_graph", "build_graph_normal", "type of model")

tf.app.flags.DEFINE_integer('train_times', 100,
                            '''Total times that you want to train''')
tf.app.flags.DEFINE_integer('train_epoch', 1000,
                            '''Total times that you want to train''')
tf.app.flags.DEFINE_integer('train_batch_size', 108, '''Train batch size''')
tf.app.flags.DEFINE_integer('input_dim', 856, '''input dim or feature dim''')
tf.app.flags.DEFINE_integer('output_dim', 9, '''output dim''')
tf.app.flags.DEFINE_boolean('is_two_cls', False,
                            '''is two cls''')
tf.app.flags.DEFINE_boolean('is_two_data', False,
                            '''is two data train and test''')
tf.app.flags.DEFINE_integer('hidden_one', 400, '''num of hidden one layer''')
tf.app.flags.DEFINE_integer('hidden_two', 200, '''num of hidden two layer''')
tf.app.flags.DEFINE_integer('hidden_three', 50, '''num of hidden three layer''')
tf.app.flags.DEFINE_float('lr_weight', 0.1, '''learning rate of weight''')
tf.app.flags.DEFINE_float('lr_ss', 0.1, '''learning rate of scale and shift''')
tf.app.flags.DEFINE_integer('train_val_num', 1080, '''nums of train and val''')
tf.app.flags.DEFINE_integer('train_num', 864, '''nums of train''')
tf.app.flags.DEFINE_integer('val_time', 50, '''val internal''')


