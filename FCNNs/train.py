
import argparse

import tensorflow as tf
import numpy as np
import random
import scipy.io as scipy
import tqdm
from utils import onehot
from model import *

from config.config import *

def read_datasets():
    if FLAGS.is_two_data:
        balance_train = scipy.loadmat(FLAGS.input_dir_train)
        data_train = balance_train[FLAGS.data_name_train]
        inputx_train = data_train[:, 0:FLAGS.input_dim]
        label_train = data_train[:, FLAGS.input_dim].astype(int)

        balance_test = scipy.loadmat(FLAGS.input_dir_test)
        data_test = balance_test[FLAGS.data_name_test]
        inputx_test = data_test[:, 0:FLAGS.input_dim]
        label_test = data_test[:, FLAGS.input_dim].astype(int)
        if FLAGS.is_two_cls:
            label_train[label_train == 1] = 2
            label_train[label_train == -1] = 1
            label_test[label_test == 1] = 2
            label_test[label_test == -1] = 1
        label_train = onehot(label_train)
        label_test = onehot(label_test)
        return inputx_train,label_train,inputx_test,label_test
    else:
        balance = scipy.loadmat(FLAGS.input_dir)
        data = balance[FLAGS.data_name]
        inputx = data[:, 0:FLAGS.input_dim]
        label = data[:, FLAGS.input_dim].astype(int)
        if FLAGS.is_two_cls:
            label[label == 1] = 2
            label[label == -1] = 1
        label = onehot(label)
        return inputx,label

if FLAGS.is_two_data:
    inputx_train, label_train, inputx_test, label_test = read_datasets()
else:
    inputx,label = read_datasets()
print('============')
print(FLAGS.build_graph)

list_ss = []
#Build training graph, train and save the trained model
tf.reset_default_graph()
if FLAGS.build_graph == "build_graph_normal":
    (x, y_), train_step, accuracy, _, saver = build_graph_normal(True, list_ss)
elif FLAGS.build_graph == "build_graph_FwSS_compare_3":
    (x, y_), train_step, accuracy, _, saver = build_graph_FwSS_compare_3(True, list_ss)
elif FLAGS.build_graph == "build_graph_FwSS_compare_2":
    (x, y_), train_step, accuracy, _, saver = build_graph_FwSS_compare_2(True, list_ss)
elif FLAGS.build_graph == "build_graph_FwSS_compare_1":
    (x, y_), train_step, accuracy, _, saver = build_graph_FwSS_compare_1(True, list_ss)
elif FLAGS.build_graph == "build_graph_FwSS":
    (x, y_), train_step, accuracy, _, saver = build_graph_FwSS(True, list_ss)
else:

    print("error of model type")
# 保存每次训练结果
staticsmean=np.zeros(FLAGS.train_times)

with tf.Session() as sess:
    # 训练循环次数
    for iq in range(FLAGS.train_times):
        acc = []
        # 初始化全局参数
        sess.run(tf.global_variables_initializer())
        if FLAGS.is_two_data:
            list1 = range(0, FLAGS.train_val_num)
            list2 = random.sample(list1, FLAGS.train_val_num)
            input_data = inputx_train[list2[0:FLAGS.train_val_num], :]
            output_data = label_train[list2[0:FLAGS.train_val_num], :]
            test_data = inputx_test
            test_output = label_test
        else:
            list1 = range(0, FLAGS.train_val_num)
            list2 = random.sample(list1, FLAGS.train_val_num)
            input_data = inputx[list2[0:FLAGS.train_num], :]
            output_data = label[list2[0:FLAGS.train_num], :]

            test_data = inputx[list2[FLAGS.train_num:FLAGS.train_val_num]]
            test_output = label[list2[FLAGS.train_num:FLAGS.train_val_num]]

        for i in tqdm.tqdm(range(FLAGS.train_epoch)):
            list3 = range(0, FLAGS.train_num)
            list4 = random.sample(list3, FLAGS.train_num)
            for j in range(int(FLAGS.train_num/FLAGS.train_batch_size)):
                train_step.run(feed_dict={x: input_data[list4[j*FLAGS.train_batch_size:(j+1)*FLAGS.train_batch_size],:], y_: output_data[list4[j*FLAGS.train_batch_size:(j+1)*FLAGS.train_batch_size],:]})
            if i % FLAGS.val_time is 0:
                res = sess.run([accuracy],feed_dict={x: test_data, y_: test_output})
                acc.append(res[0])
        print("max acc:", max(acc))
        print("Final accuracy:", acc[-1])
        staticsmean[iq] =  max(acc)
        if max(acc) > max(staticsmean):
            saved_model = saver.save(sess,FLAGS.output_dir_model)
np.savetxt(FLAGS.output_dir, staticsmean, fmt='%g', delimiter=',')

print(staticsmean)
print(np.mean(staticsmean))