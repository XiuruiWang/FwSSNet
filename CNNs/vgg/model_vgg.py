# -*- coding:utf-8 -*-
import tensorflow as tf

from utils_vgg import *
from config_vgg import *


# 瀹氫箟棰勫鐞嗗眰
def preprocess_layer(x,W):
    output = conv2d(x, W)
    return output
# 瀹氫箟鏅�氬叏杩炴帴灞�
def cnn_layer(x, W, train_flag):
    output = conv2d(x, W)
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    return output

# 瀹氫箟FwSS鍏ㄨ繛鎺ュ眰
def FwSS_cnn_layer(x, W, train_flag):
    output = batch_norm_3(x,train_flag)
    output = conv2d(output, W)
    output = tf.nn.relu(output)
    return output

# 瀹氫箟鏅�氭渶鍚庝竴灞傦細鍏ㄨ繛鎺�
def last_layer(x,W,b):
    output = tf.matmul(x, W) + b
    return output



def build_graph_normal():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(x, W_conv1_1, train_flag)

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv1_2, train_flag)
    output = max_pool(output, 2, 2, "pool1")

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv2_1, train_flag)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv2_2, train_flag)
    output = max_pool(output, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv3_1, train_flag)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv3_2, train_flag)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv3_3, train_flag)

    W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv3_4, train_flag)
    output = max_pool(output, 2, 2, "pool3")

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv4_1, train_flag)

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv4_2, train_flag)

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv4_3, train_flag)

    W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv4_4, train_flag)
    output = max_pool(output, 2, 2)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv5_1, train_flag)

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv5_2, train_flag)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv5_3, train_flag)

    W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = cnn_layer(output, W_conv5_4, train_flag)

    output = max_pool(output, 2, 2)

    global_pool = tf.reduce_mean(output, [1, 2])

    W_fc1 = tf.get_variable('fc1', shape=[512, 10], initializer=tf.contrib.keras.initializers.he_normal())
    fc_b = tf.get_variable(name='fc_b', shape=[10], initializer=tf.zeros_initializer())

    output = last_layer(global_pool, W_fc1, fc_b)


    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag




def build_graph_FwSS():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = preprocess_layer(x,W_conv1_1)


    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv1_2, train_flag)

    output = max_pool(output, 2, 2, "pool1")

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv2_1, train_flag)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv2_2, train_flag)

    output = max_pool(output, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv3_1, train_flag)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv3_2, train_flag)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv3_3, train_flag)


    W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv3_4, train_flag)

    output = max_pool(output, 2, 2, "pool3")

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv4_1, train_flag)


    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv4_2, train_flag)


    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv4_3, train_flag)


    W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv4_4, train_flag)

    output = max_pool(output, 2, 2)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv5_1, train_flag)


    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv5_2, train_flag)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv5_3, train_flag)


    W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = FwSS_cnn_layer(output, W_conv5_4, train_flag)

    output = max_pool(output, 2, 2)

    global_pool = tf.reduce_mean(output, [1, 2])

    W_fc1 = tf.get_variable('fc1', shape=[512, 10], initializer=tf.contrib.keras.initializers.he_normal())
    fc_b = tf.get_variable(name='fc_b', shape=[10], initializer=tf.zeros_initializer())

    output = last_layer(global_pool, W_fc1, fc_b)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag






