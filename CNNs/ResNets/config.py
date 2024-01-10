import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", ".", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", ".", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("build_model", "build_model_FwSS_compare_2", '''type of model ''')

# tf.app.flags.DEFINE_string("log_dir", ".", "Model directory where final model files are saved.")

## define hyper-parameters related to training procedure and network
tf.app.flags.DEFINE_integer('train_steps', 31280,
                            '''Total steps that you want to train''')

tf.app.flags.DEFINE_boolean('is_full_validation', True,
                            '''Validation w/ full validation set or a random batch''')

tf.app.flags.DEFINE_integer('train_batch_size', 64, '''Train batch size''')

tf.app.flags.DEFINE_integer('validation_batch_size', 125,
                            '''Validation batch size, better to be a divisor of 
                            10000 for this task''')

tf.app.flags.DEFINE_integer('test_batch_size', 64, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.1, '''Initial learning rate''')

tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the 
                          learning rate each time''')

tf.app.flags.DEFINE_integer('decay_step0', 20000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 30000, '''At which step to decay the learning rate''')

tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')

tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

## The following flags are related to save paths, tensorboard outputs and screen outputs
tf.app.flags.DEFINE_string('version', 'ResNet_110',
                           '''A version number defining the directory to save
                            logs and checkpoints''')

tf.app.flags.DEFINE_integer('report_freq', 2, '''Steps takes to output errors on the screen
                             and write summaries''')

tf.app.flags.DEFINE_float('train_ema_decay', 0.95,
                          '''The decay factor of the train error's
                          moving average shown on tensorboard''')

## The following flags are related to data-augmentation
tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero 
                            padding on each side of the image''')

## If you want to load a checkpoint and continue training
tf.app.flags.DEFINE_string('ckpt_path', 'cache/logs_repeat20/model.ckpt-100000',
                           '''Checkpoint directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-39999', '''Checkpoint
directory to restore''')

tf.app.flags.DEFINE_string("train_dir_csv", "resnet_normal_1.csv",
                           "Output directory where output such as csv are saved.")
tf.app.flags.DEFINE_string("train_dir", "logs_normal/", "Output directory where output such as csv are saved.")

# train_dir = 'logs_raw' +'_'+FLAGS.T +'_' +FLAGS.version + '/'
# train_dir_csv = FLAGS.T