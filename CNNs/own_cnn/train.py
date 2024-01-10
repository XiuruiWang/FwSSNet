# -*- coding:utf-8 -*-
import numpy
import tensorflow as tf
import numpy as np
import time

from utils_own_CNN import *
from config_own_CNN import *
from model_own_CNN import *

class_num = FLAGS.class_num
image_size = FLAGS.image_size
img_channels = FLAGS.img_channels
iterations = FLAGS.iterations
batch_size = FLAGS.batch_size
total_epoch = FLAGS.total_epoch
weight_decay = FLAGS.weight_decay
log_save_path =  FLAGS.log_save_path
model_save_path = FLAGS.model_save_path




def run_testing(sess, ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index = pre_index + add
        loss_, acc_  = sess.run([cross_entropy, accuracy],
                                feed_dict={x: batch_x, y_: batch_y, train_flag: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss, summary


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    if FLAGS.build_graph == "build_graph_normal":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_normal()
    elif FLAGS.build_graph == "build_graph_change_paper":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_paper()
    elif FLAGS.build_graph == "build_graph_change_addnormal":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_addnormal()
    elif FLAGS.build_graph == "build_graph_change_addnew":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_addnew()
    elif FLAGS.build_graph == "build_graph_new":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_new()
    elif FLAGS.build_graph == "build_graph_new_addnormal":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_new_addnormal()
    elif FLAGS.build_graph == "build_graph_change_new":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_new()
    elif FLAGS.build_graph == "build_graph_change_new_addnormal":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_new_addnormal()
    elif FLAGS.build_graph == "build_graph_change_new_plus":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_new_plus()
    elif FLAGS.build_graph == "build_graph_change_new_addnormal_plus":
        (x, y_), train_step, accuracy, learning_rate, cross_entropy, train_flag = build_graph_change_new_addnormal_plus()
    else:

        print("error of model type")

    # initial an saver to save model
    saver = tf.train.Saver()
    train_loss_list = np.zeros(FLAGS.total_epoch)
    train_acc_list = np.zeros(FLAGS.total_epoch)
    test_loss_list = np.zeros(FLAGS.total_epoch)
    test_acc_list = np.zeros(FLAGS.total_epoch)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)

        for ep in range(1, total_epoch+1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\n epoch %d/%d:" % (ep, total_epoch))

            for it in range(1, iterations+1):
                batch_x = train_x[pre_index:pre_index+batch_size]
                batch_y = train_y[pre_index:pre_index+batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y,
                                                    learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    loss_, acc_ = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_x, y_: batch_y, train_flag: True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                               tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess, ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (it, iterations, int(time.time()-start_time), train_loss, train_acc, val_loss, val_acc))
                    train_loss_list[ep-1] = train_loss
                    train_acc_list[ep-1] = train_acc
                    test_loss_list[ep-1] = val_loss
                    test_acc_list[ep-1] = val_acc
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                          % (it, iterations, train_loss / it, train_acc / it))
            # save_path = saver.save(sess, model_save_path)
            # print("Model saved in file: %s" % save_path)
        result = [train_loss_list, train_acc_list, test_loss_list, test_acc_list]
        result = np.swapaxes(result, 0, 1)
        np.savetxt(FLAGS.output_dir, result, fmt='%g', delimiter=',')

