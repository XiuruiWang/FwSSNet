import tensorflow as tf
import numpy as np

def norm(inputs,is_training,decay = 0.999):
    epsilon = 0.00000000001
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return (inputs - batch_mean) / tf.sqrt(batch_var + epsilon)
    else:
        return (inputs - pop_mean) / tf.sqrt(pop_var + epsilon)





def onehot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels)
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels-1] = 1
    return onehot_labels

def onehot_cifar(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = 9 #max(labels)
    onehot_labels = np.zeros((n_sample, n_class+1))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

def scale_and_shift(in_data,list_ss):
    beta = tf.Variable(tf.zeros((1, in_data.get_shape()[-1])), trainable=True)  # inputX.shape[1],
    scale = tf.Variable(tf.zeros((1, in_data.get_shape()[-1])), trainable=True)
    list_ss.append(beta)
    list_ss.append(scale)
    x1 = tf.multiply(in_data, scale)
    x2 = tf.add(in_data, x1) + beta
    return x2


def scale_and_shift_3_all(in_data,list_ss_all):

    scale = tf.Variable(np.zeros((1, in_data.get_shape()[-1])), trainable=True, dtype=tf.float32)
    shift = tf.Variable(np.zeros((1, in_data.get_shape()[-1])),trainable=True, dtype=tf.float32)
    y1_ = tf.multiply(in_data, scale)
    y2_ = tf.add(in_data, y1_) + shift
    list_ss_all.append(scale)
    list_ss_all.append(shift)

    return y2_

def scale_and_shift_3(in_data,list_ss_single):

    scale = tf.Variable(np.zeros((1, in_data.get_shape()[-1])), trainable=True, dtype=tf.float32)
    shift = tf.Variable(np.zeros((1, in_data.get_shape()[-1])),trainable=True, dtype=tf.float32)
    y1_ = tf.multiply(in_data, scale)
    y2_ = tf.add(in_data, y1_) + shift
    list_ss_single.append(scale)
    list_ss_single.append(shift)
    return y2_


def bn_3(inputs,is_training,decay = 0.999):
    epsilon = 0.00000000001
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return (inputs - batch_mean) / tf.sqrt(batch_var + epsilon)
    else:
        return (inputs - pop_mean) / tf.sqrt(pop_var + epsilon)


def bn_3_all(inputs,is_training,decay = 0.999):
    epsilon = 0.00000000001
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-2],inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-2],inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2,3])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return (inputs - batch_mean) / tf.sqrt(batch_var + epsilon)
    else:
        return (inputs - pop_mean) / tf.sqrt(pop_var + epsilon)




def loss_func(logits,labels):
    labels = tf.cast(labels,tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                           labels=labels,name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(tf.reduce_sum(cross_entropy))
    tf.add_to_collection("losses",cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"),name="total_loss")


def variable_with_weight_loss(shape,std,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=std),dtype=tf.float32)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var