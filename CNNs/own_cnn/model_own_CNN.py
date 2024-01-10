# -*- coding:utf-8 -*-
import tensorflow as tf

from utils_own_CNN import *
from config_own_CNN import *

def build_graph_normal():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_3(conv2d(x, W_conv1_1),train_flag))



    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_3(conv2d(output, W_conv2_1),train_flag))


    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_3(conv2d(output, W_conv2_2),train_flag))

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_3(conv2d(output, W_conv2_3),train_flag))


    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_3(conv2d(output, W_conv2_4),train_flag))

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_3(conv2d(output, W_conv2_5),train_flag))

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_1(tf.matmul(output, W_fc1),train_flag))

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm_1(tf.matmul(output, W_fc2),train_flag))

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(tf.matmul(output, W_fc3),train_flag)
    # output = tf.matmul(output, W_fc3)
    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_change_paper():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(tf.nn.relu(conv2d(x, W_conv1_1)),train_flag)



    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(tf.nn.relu(conv2d(output, W_conv2_1)),train_flag)


    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(tf.nn.relu(conv2d(output, W_conv2_2)),train_flag)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(tf.nn.relu(conv2d(output, W_conv2_3)),train_flag)


    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(tf.nn.relu(conv2d(output, W_conv2_4)),train_flag)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(tf.nn.relu(conv2d(output, W_conv2_5)),train_flag)

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(tf.nn.relu(tf.matmul(output, W_fc1)),train_flag)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(tf.nn.relu(tf.matmul(output, W_fc2)),train_flag)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc3)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag


def build_graph_change_addnormal():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network
    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(x,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv1_1)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_1)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_2)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_3)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_4)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_5)
    # output = tf.contrib.layers.flatten(output)

    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc2)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc3)
    output = batch_norm_1(output,train_flag)


    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_change_addnew():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(x,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv1_1)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_1)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_2)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_3)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_4)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = tf.nn.relu(output)
    output = conv2d(output, W_conv2_5)
    # output = tf.contrib.layers.flatten(output)
    output = batch_norm_3(output,train_flag)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc2)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc3)
    output = batch_norm_1(output,train_flag)


    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_new():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(x,train_flag)
    output = conv2d(output, W_conv1_1)
    output = tf.nn.relu(output)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_1)
    output = tf.nn.relu(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_2)
    output = tf.nn.relu(output)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_3)
    output = tf.nn.relu(output)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_4)
    output = tf.nn.relu(output)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_5)
    output = tf.nn.relu(output)
    # output = tf.contrib.layers.flatten(output)
    output = batch_norm_3(output,train_flag)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(output,train_flag)
    output = tf.matmul(output, W_fc2)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(output,train_flag)
    output = tf.matmul(output, W_fc3)
    # output = tf.nn.relu(output)

    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_new_addnormal():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network


    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(x,train_flag)
    output = conv2d(output, W_conv1_1)
    output = tf.nn.relu(output)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_1)
    output = tf.nn.relu(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_2)
    output = tf.nn.relu(output)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_3)
    output = tf.nn.relu(output)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_4)
    output = tf.nn.relu(output)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_5)
    output = tf.nn.relu(output)
    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc2)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc3)
    output = batch_norm_1(output,train_flag)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_change_new():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = conv2d(x, W_conv1_1)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_1)
    output = tf.nn.relu(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_2)
    output = tf.nn.relu(output)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_3)
    output = tf.nn.relu(output)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_4)
    output = tf.nn.relu(output)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_5)
    output = tf.nn.relu(output)
    # output = tf.contrib.layers.flatten(output)
    output = batch_norm_3(output,train_flag)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(output,train_flag)
    output = tf.matmul(output, W_fc2)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(output,train_flag)
    output = tf.matmul(output, W_fc3)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_change_new_addnormal():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network


    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = conv2d(x, W_conv1_1)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_1)
    output = tf.nn.relu(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_2)
    output = tf.nn.relu(output)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_3)
    output = tf.nn.relu(output)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_4)
    output = tf.nn.relu(output)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_5)
    output = tf.nn.relu(output)
    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc2)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc3)
    output = batch_norm_1(output,train_flag)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_change_new_plus():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network
    W_conv0_1 = tf.get_variable('conv0_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = conv2d(x, W_conv0_1)

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv1_1)
    output = tf.nn.relu(output)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_1)
    output = tf.nn.relu(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_2)
    output = tf.nn.relu(output)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_3)
    output = tf.nn.relu(output)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_4)
    output = tf.nn.relu(output)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_5)
    output = tf.nn.relu(output)
    # output = tf.contrib.layers.flatten(output)
    output = batch_norm_3(output,train_flag)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(output,train_flag)
    output = tf.matmul(output, W_fc2)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_1(output,train_flag)
    output = tf.matmul(output, W_fc3)    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag

def build_graph_change_new_addnormal_plus():
    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.class_num])
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network
    W_conv0_1 = tf.get_variable('conv0_1', shape=[5, 5, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = conv2d(x, W_conv0_1)

    W_conv1_1 = tf.get_variable('conv1_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv1_1)
    output = tf.nn.relu(output)


    W_conv2_1 = tf.get_variable('conv2_1', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_1)
    output = tf.nn.relu(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_2)
    output = tf.nn.relu(output)

    W_conv2_3 = tf.get_variable('conv2_3', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_3)
    output = tf.nn.relu(output)

    W_conv2_4 = tf.get_variable('conv2_4', shape=[5, 5, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_4)
    output = tf.nn.relu(output)

    W_conv2_5 = tf.get_variable('conv2_5', shape=[5, 5, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    output = batch_norm_3(output,train_flag)
    output = conv2d(output, W_conv2_5)
    output = tf.nn.relu(output)
    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 32 * 32 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[65536, 384], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc1)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc2 = tf.get_variable('fc7', shape=[384, 192], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc2)
    output = batch_norm_1(output,train_flag)
    output = tf.nn.relu(output)

    W_fc3 = tf.get_variable('fc3', shape=[192, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.matmul(output, W_fc3)
    output = batch_norm_1(output,train_flag)

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag


