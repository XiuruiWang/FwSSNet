import tensorflow as tf
import numpy as np
import argparse

from utils import norm,onehot,scale_and_shift
from config.config import *

# 定义预处理层
def preprocess_layer(x,W):
    y = tf.matmul(x, W)
    return y
# 定义普通全连接层
def full_layer(x, W, is_training, list_ss):
    y_mid1 = tf.matmul(x, W)
    y_mid2 = norm(y_mid1, is_training)
    y_mid3 = scale_and_shift(y_mid2,list_ss)
    y=tf.nn.relu(y_mid3)
    return y
# 定义FwSS全连接层

def FwSS_full_layer(x, W, is_training, list_ss):
    y_mid1 = norm(x, is_training)
    y_mid2 = scale_and_shift(y_mid1,list_ss)
    y_mid3 = tf.matmul(y_mid2, W)
    y=tf.nn.relu(y_mid3)
    return y


# 定义普通最后一层
def last_layer(x,W):
    y_mid1 = tf.matmul(x, W)
    y = tf.nn.softmax(y_mid1)
    return y

# 定义FwSS最后一层
def FwSS_last_layer(x,W, is_training):
    y_mid1 = norm(x, is_training)
    y_mid2 = tf.matmul(y_mid1, W)
    y = tf.nn.softmax(y_mid2)
    return y

# 定义FwSS最后一层保留FwSS
def FwSS_last_layer_compare(x,W, is_training, list_ss):
    y_mid1 = norm(x, is_training)
    y_mid2 = scale_and_shift(y_mid1,list_ss)

    y_mid3 = tf.matmul(y_mid2, W)
    y = tf.nn.softmax(y_mid3)
    return y


def build_graph_normal(is_training,list_ss):

    x1 = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
    y_ = tf.placeholder(tf.float32, [None,FLAGS.output_dim])

    w1_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.input_dim, FLAGS.hidden_one)).astype(np.float32)
    w2_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_one, FLAGS.hidden_two)).astype(np.float32)
    w3_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_two, FLAGS.hidden_three)).astype(np.float32)
    w4_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_three, FLAGS.output_dim)).astype(np.float32)

    # Layer 1
    w1 = tf.Variable(w1_initial)
    l1 =  full_layer(x1, w1, is_training, list_ss)

    # Layer 2
    x2 = l1
    w2 = tf.Variable(w2_initial)
    l2 =  full_layer(x2, w2, is_training, list_ss)

    #Layer 3
    x3 = l2
    w3 = tf.Variable(w3_initial)
    l3 = full_layer(x3, w3, is_training, list_ss)

    # Softmax
    x4 = l3
    w4 = tf.Variable(w4_initial)
    y  = last_layer(x4,w4)

    # Loss, Optimizer and Predictions
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

    gr2=[w1,w2,w3,w4]

    train_op1 = tf.train.GradientDescentOptimizer(FLAGS.lr_ss).minimize(cross_entropy, var_list=list_ss)
    train_op2 = tf.train.GradientDescentOptimizer(FLAGS.lr_weight).minimize(cross_entropy, var_list=gr2)
    train_step=tf.group(train_op1,train_op2)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    return (x1, y_), train_step, accuracy, y, tf.train.Saver()


def build_graph_FwSS(is_training,list_ss):

    x1 = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
    y_ = tf.placeholder(tf.float32, [None,FLAGS.output_dim])

    w1_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.input_dim, FLAGS.hidden_one)).astype(np.float32)
    w2_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_one, FLAGS.hidden_two)).astype(np.float32)
    w3_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_two, FLAGS.hidden_three)).astype(np.float32)
    w4_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_three, FLAGS.output_dim)).astype(np.float32)

    # Layer 1
    w1 = tf.Variable(w1_initial)
    l1 = preprocess_layer(x1, w1)

    # Layer 2
    x2 = l1
    w2 = tf.Variable(w2_initial)
    l2 = FwSS_full_layer(x2, w2, is_training, list_ss)

    #Layer 3
    x3 = l2
    w3 = tf.Variable(w3_initial)
    l3 = FwSS_full_layer(x3, w3, is_training, list_ss)

    # Softmax
    x4 = l3
    w4 = tf.Variable(w4_initial)

    y  = FwSS_last_layer(x4, w4, is_training)

    # Loss, Optimizer and Predictions
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

    gr2=[w1,w2,w3,w4]

    train_op1 = tf.train.GradientDescentOptimizer(FLAGS.lr_ss).minimize(cross_entropy, var_list=list_ss)
    train_op2 = tf.train.GradientDescentOptimizer(FLAGS.lr_weight).minimize(cross_entropy, var_list=gr2)
    train_step=tf.group(train_op1,train_op2)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    return (x1, y_), train_step, accuracy, y, tf.train.Saver()

def build_graph_FwSS_compare_1(is_training,list_ss):

    x1 = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
    y_ = tf.placeholder(tf.float32, [None,FLAGS.output_dim])

    w1_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.input_dim, FLAGS.hidden_one)).astype(np.float32)
    w2_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_one, FLAGS.hidden_two)).astype(np.float32)
    w3_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_two, FLAGS.hidden_three)).astype(np.float32)
    w4_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_three, FLAGS.output_dim)).astype(np.float32)

    # Layer 1
    w1 = tf.Variable(w1_initial)
    l1 = preprocess_layer(x1, w1)


    # Layer 2
    x2 = l1
    w2 = tf.Variable(w2_initial)
    l2 =  full_layer(x2, w2, is_training, list_ss)

    #Layer 3
    x3 = l2
    w3 = tf.Variable(w3_initial)
    l3 = full_layer(x3, w3, is_training, list_ss)

    # Softmax
    x4 = l3
    w4 = tf.Variable(w4_initial)
    y  = last_layer(x4,w4)

    # Loss, Optimizer and Predictions
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

    gr2=[w1,w2,w3,w4]

    train_op1 = tf.train.GradientDescentOptimizer(FLAGS.lr_ss).minimize(cross_entropy, var_list=list_ss)
    train_op2 = tf.train.GradientDescentOptimizer(FLAGS.lr_weight).minimize(cross_entropy, var_list=gr2)
    train_step=tf.group(train_op1,train_op2)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    return (x1, y_), train_step, accuracy, y, tf.train.Saver()

def build_graph_FwSS_compare_2(is_training,list_ss):

    x1 = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
    y_ = tf.placeholder(tf.float32, [None,FLAGS.output_dim])

    # w1_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.input_dim, FLAGS.hidden_one)).astype(np.float32)
    w2_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.input_dim, FLAGS.hidden_two)).astype(np.float32)
    w3_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_two, FLAGS.hidden_three)).astype(np.float32)
    w4_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_three, FLAGS.output_dim)).astype(np.float32)

    # Layer 1
    # w1 = tf.Variable(w1_initial)
    # l1 = FwSS_full_layer(x1, w1, is_training, list_ss)

    # Layer 2
    x2 = x1
    w2 = tf.Variable(w2_initial)
    l2 = FwSS_full_layer(x2, w2, is_training, list_ss)

    #Layer 3
    x3 = l2
    w3 = tf.Variable(w3_initial)
    l3 = FwSS_full_layer(x3, w3, is_training, list_ss)

    # Softmax
    x4 = l3
    w4 = tf.Variable(w4_initial)

    y  = FwSS_last_layer(x4, w4, is_training)

    # Loss, Optimizer and Predictions
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

    gr2=[w2,w3,w4]

    train_op1 = tf.train.GradientDescentOptimizer(FLAGS.lr_ss).minimize(cross_entropy, var_list=list_ss)
    train_op2 = tf.train.GradientDescentOptimizer(FLAGS.lr_weight).minimize(cross_entropy, var_list=gr2)
    train_step=tf.group(train_op1,train_op2)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    return (x1, y_), train_step, accuracy, y, tf.train.Saver()

def build_graph_FwSS_compare_3(is_training,list_ss):

    x1 = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
    y_ = tf.placeholder(tf.float32, [None,FLAGS.output_dim])

    w1_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.input_dim, FLAGS.hidden_one)).astype(np.float32)
    w2_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_one, FLAGS.hidden_two)).astype(np.float32)
    w3_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_two, FLAGS.hidden_three)).astype(np.float32)
    w4_initial = np.random.normal(loc=0, scale=0.1, size=(FLAGS.hidden_three, FLAGS.output_dim)).astype(np.float32)

    # Layer 1
    w1 = tf.Variable(w1_initial)
    l1 = preprocess_layer(x1, w1)

    # Layer 2
    x2 = l1
    w2 = tf.Variable(w2_initial)
    l2 = FwSS_full_layer(x2, w2, is_training, list_ss)

    #Layer 3
    x3 = l2
    w3 = tf.Variable(w3_initial)
    l3 = FwSS_full_layer(x3, w3, is_training, list_ss)

    # Softmax
    x4 = l3
    w4 = tf.Variable(w4_initial)

    y  = FwSS_last_layer_compare(x4, w4, is_training, list_ss)

    # Loss, Optimizer and Predictions
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

    gr2=[w1,w2,w3,w4]

    train_op1 = tf.train.GradientDescentOptimizer(FLAGS.lr_ss).minimize(cross_entropy, var_list=list_ss)
    train_op2 = tf.train.GradientDescentOptimizer(FLAGS.lr_weight).minimize(cross_entropy, var_list=gr2)
    train_step=tf.group(train_op1,train_op2)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    return (x1, y_), train_step, accuracy, y, tf.train.Saver()
