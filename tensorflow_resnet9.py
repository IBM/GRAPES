# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:55:43 2021

@author: 7G5158848
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import argparse
import os
import sys


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    # ask for arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-grapes_fc','--grapes_fc', 
                    default=False, action='store_true',
                    help="Choose to apply GRAPES to fc layers")
    parser.add_argument('-grapes_fc_local','--grapes_fc_local', 
                    default=False, action='store_true',
                    help="Choose to apply GRAPES to fc layers in the local version")
    parser.add_argument('-grapes_conv','--grapes_conv', 
                    default=False, action='store_true',
                    help="Choose to apply GRAPES to conv layers")
    parser.add_argument('-grapes_conv_local','--grapes_conv_local', 
                    default=False, action='store_true',
                    help="Choose to apply GRAPES to conv layers in the local version")
    parser.add_argument('-mn', '--mnist', action='store_true',
                    default=False, 
                    help="use mnist as dataset")
    parser.add_argument('-cif', '--cifar10', action='store_true',
                        default=False,
                        help="use cifar10 as dataset")
    parser.add_argument('-cif100', '--cifar100', action='store_true',
                        default=False,
                        help="use cifar100 as dataset")
    parser.add_argument('-bs', '--batch_size',
                    default=64,type=int,
                    help="Batch size during training. Choose an integer")
    parser.add_argument('-lr', '--learning_rate',
                    type=float, default=0.001, 
                    help="Learning rate")
    parser.add_argument('-trep', '--num_epochs',
                    type=int, default= 3,
                    help="Number of training epochs")
    parser.add_argument('-run', '--run',
                    type=int, default= 0,
                    help="Run number")
    parser.add_argument('-rin', '--random_init', action='store_true',
                        default=False,
                        help="use random initialization")
    parser.add_argument('-dec', '--decay',
                    type=float, default=1, 
                    help="Learning rate decay")
    parser.add_argument('-decstep', '--decay_step',
                    type=int, default=50, 
                    help="Learning rate decay step")
    parser.add_argument('-opt', '--optimizer',
                    type=str, default='GD', 
                    help="Optimizer")
    parser.add_argument('-bn', '--batch_norm', action='store_true',
                        default=False,
                        help="use batch normalization")
    args = parser.parse_args()
    grapes_fc = args.grapes_fc
    grapes_fc_local = args.grapes_fc_local
    grapes_conv = args.grapes_conv
    grapes_conv_local = args.grapes_conv_local
    dataset_mnist = args.mnist
    dataset_cifar10 = args.cifar10
    dataset_cifar100 = args.cifar100
    mini_batch_size = args.batch_size
    learning_rate_start = args.learning_rate
    num_epochs = args.num_epochs
    run = args.run
    if args.random_init:
        w_init = 'rand'
    else:
        w_init = 'he'
    decay = args.decay
    decay_step = args.decay_step
    optim = args.optimizer
    batch_norm = args.batch_norm
    
    # create folder to save results
    if dataset_mnist:
        print("Data set MNIST chosen")
        savepath = "res_resnet9_512_"+optim+"_mnist"
        input_dim=(28, 28, 1)
        nout = 10
    elif dataset_cifar10:
        print("Data set CIFAR-10 chosen")
        savepath = "res_resnet9_512_"+optim+"_cifar"
        input_dim=(32, 32, 3)
        nout = 10
    elif dataset_cifar100:
        print("Data set CIFAR-100 chosen")
        savepath = "res_resnet9_512_"+optim+"_cifar100"
        input_dim=(32, 32, 3)
        nout = 100
    else:
        print("Error: no data set chosen")
        sys.exit(1)
    
    if grapes_fc:
        savepath += "_grapesFC"
        if grapes_fc_local:
            savepath += "local"
        else:
            savepath += "prop"
    else:
        savepath += "_nograpesFC"
    if grapes_conv:
        savepath += "_grapesCONV"
        if grapes_conv_local:
            savepath += "local"
        else:
            savepath += "prop"
    else:
        savepath += "_nograpesCONV"
    savepath += "_mbs"+str(mini_batch_size)
    savepath += "_trep"+str(num_epochs)
    savepath += "_lr"+str(learning_rate_start)[2:]
    savepath += "_"+w_init+"true"
    if decay is not 1:
        savepath += "_dec"+str(decay)[2:]
    else:
        savepath += "_nodec"
    savepath += '_r'+str(run)
    if batch_norm:
        savepath += "_bn"
    try:
        os.mkdir(savepath)
    except OSError:
        print ("Directory %s already exists" % savepath)
    else:
        print ("Successfully created the directory %s " % savepath)
    
    # only for mnist 
    # TODO: add for CIFAR 
    if dataset_mnist:
        mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
        X_train, y_train           = mnist.train.images, mnist.train.labels
        X_validation, y_validation = mnist.validation.images, mnist.validation.labels
        X_test, y_test             = mnist.test.images, mnist.test.labels
        assert(len(X_train) == len(y_train))
        assert(len(X_validation) == len(y_validation))
        assert(len(X_test) == len(y_test))
        print()
        print("Image Shape: {}".format(X_train[0].shape))
        print()
        print("Training Set:   {} samples".format(len(X_train)))
        print("Validation Set: {} samples".format(len(X_validation)))
        print("Test Set:       {} samples".format(len(X_test)))
        # Pad images with 0s
        X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        print("Updated Image Shape: {}".format(X_train[0].shape))
        X_train, y_train = shuffle(X_train, y_train)
        
    elif dataset_cifar10:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype(np.float32)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype(np.float32)
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])
        X_validation = X_test
        y_validation = y_test
        assert(len(X_train) == len(y_train))
        assert(len(X_validation) == len(y_validation))
        assert(len(X_test) == len(y_test))
        print()
        print("Image Shape: {}".format(X_train[0].shape))
        print()
        print("Training Set:   {} samples".format(len(X_train)))
        print("Validation Set: {} samples".format(len(X_validation)))
        print("Test Set:       {} samples".format(len(X_test)))
        
    elif dataset_cifar100:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype(np.float32)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype(np.float32)
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])
        X_validation = X_test
        y_validation = y_test
        assert(len(X_train) == len(y_train))
        assert(len(X_validation) == len(y_validation))
        assert(len(X_test) == len(y_test))
        print()
        print("Image Shape: {}".format(X_train[0].shape))
        print()
        print("Training Set:   {} samples".format(len(X_train)))
        print("Validation Set: {} samples".format(len(X_validation)))
        print("Test Set:       {} samples".format(len(X_test)))
        
    
    def ResNet9(x, input_channels,
              grapes_fc=False, grapes_fc_local=False, grapes_conv=False , grapes_conv_local=False,
              is_training=False):    
        # Hyperparameters
        mu = 0
        sigma = 1.0
        if w_init == 'rand':
            sigma = 0.01
        

        ################# initialize the parameters  ####################
         # Layer 1: Convolutional. Input = 32x32x1/3. Output = ?x?x64.
        conv1_w = tf.Variable(tf.truncated_normal(shape = [3,3,input_channels,64],mean = mu, stddev = sigma))
        if w_init == 'he':
            conv1_w = conv1_w * math.sqrt(2/(input_channels*3*3))
        elif w_init == 'xav':
            conv1_w = conv1_w * math.sqrt(2/(input_channels*3*3+64))
        # Layer 2: Convolutional. Output = ?x?x128.
        conv2_w = tf.Variable(tf.truncated_normal(shape = [3,3,64,128], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv2_w = conv2_w * math.sqrt(2/(64*3*3))
        elif w_init == 'xav':
            conv2_w = conv2_w * math.sqrt(2/(64*3*3+128))
        # Layer 3: Convolutional. Output = ?x?x128.
        conv3_w = tf.Variable(tf.truncated_normal(shape = [3,3,128,128], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv3_w = conv3_w * math.sqrt(2/(128*3*3))
        elif w_init == 'xav':
            conv3_w = conv3_w * math.sqrt(2/(128*3*3+128))
        # Layer 4: Convolutional. Output = ?x?x128.
        conv4_w = tf.Variable(tf.truncated_normal(shape = [3,3,128,128], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv4_w = conv4_w * math.sqrt(2/(128*3*3))
        elif w_init == 'xav':
            conv4_w = conv4_w * math.sqrt(2/(128*3*3+128))
        # Layer 5: Convolutional. Output = ?x?x256.
        conv5_w = tf.Variable(tf.truncated_normal(shape = [3,3,128,256], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv5_w = conv5_w * math.sqrt(2/(128*3*3))
        elif w_init == 'xav':
            conv5_w = conv5_w * math.sqrt(2/(128*3*3+256))
        # Layer 6: Convolutional. Output = ?x?x256.
        conv6_w = tf.Variable(tf.truncated_normal(shape = [3,3,256,512], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv6_w = conv6_w * math.sqrt(2/(256*3*3))
        elif w_init == 'xav':
            conv6_w = conv6_w * math.sqrt(2/(256*3*3+512))
        # Layer 7: Convolutional. Output = ?x?x256.
        conv7_w = tf.Variable(tf.truncated_normal(shape = [3,3,512,512], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv7_w = conv7_w * math.sqrt(2/(512*3*3))
        elif w_init == 'xav':
            conv7_w = conv7_w * math.sqrt(2/(512*3*3+512))
        # Layer 8: Convolutional. Output = ?x?x256.
        conv8_w = tf.Variable(tf.truncated_normal(shape = [3,3,512,512], mean = mu, stddev = sigma))
        if w_init == 'he':
            conv8_w = conv8_w * math.sqrt(2/(512*3*3))
        elif w_init == 'xav':
            conv8_w = conv8_w * math.sqrt(2/(512*3*3+512))
            
        #sigma = 1.0
        # Layer 9: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape = (512,nout), mean = mu, stddev = sigma))
        if w_init == 'he':
            fc1_w = fc1_w * math.sqrt(2/512)
        elif w_init == 'xav':
            fc1_w = fc1_w * math.sqrt(2/(512+nout))
        #sigma_fc1 = np.sqrt(6.0 / 400)
        #fc1_w = tf.Variable(tf.random.uniform(shape = (400,120), mean = mu, stddev = sigma_fc1))
        
            
           
        ##############  Build the graph  ###############
        # Layer 1    
        conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv1 = tf.nn.batch_normalization(conv1, mean=tf.reduce_mean(conv1),  variance=tf.math.reduce_variance(conv1),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv1 = tf.nn.relu(conv1)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv1_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv1_w), [0,1,2])),1,2))
            conv1 = conv1 * correction_factor + tf.stop_gradient(-conv1 * correction_factor + conv1)
       
        # Layer2
        conv2 = tf.nn.conv2d(conv1, conv2_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv2 = tf.nn.batch_normalization(conv2, mean=tf.reduce_mean(conv2),  variance=tf.math.reduce_variance(conv2),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv2 = tf.nn.relu(conv2)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv2_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv2_w), [0,1,2])),1,2))
            conv2 = conv2 * correction_factor + tf.stop_gradient(-conv2 * correction_factor + conv2)
    
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')     
    
        # Layer 3    
        conv3 = tf.nn.conv2d(pool_2, conv3_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv3 = tf.nn.batch_normalization(conv3, mean=tf.reduce_mean(conv3),  variance=tf.math.reduce_variance(conv3),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv3 = tf.nn.relu(conv3)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv3_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv3_w), [0,1,2])),1,2))
            conv3 = conv3 * correction_factor + tf.stop_gradient(-conv3 * correction_factor + conv3)
    
        # Layer 4
        conv4 = tf.nn.conv2d(conv3, conv4_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv4 = tf.nn.batch_normalization(conv4, mean=tf.reduce_mean(conv4),  variance=tf.math.reduce_variance(conv4),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv4 = tf.nn.relu(conv4)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv4_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv4_w), [0,1,2])),1,2))
            conv4 = conv4 * correction_factor + tf.stop_gradient(-conv4 * correction_factor + conv4)

        # Residual connection 1
        conv4 += pool_2
        
        # Layer5
        conv5 = tf.nn.conv2d(conv4, conv5_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv5 = tf.nn.batch_normalization(conv5, mean=tf.reduce_mean(conv5),  variance=tf.math.reduce_variance(conv5),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv5 = tf.nn.relu(conv5)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv5_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv5_w), [0,1,2])),1,2))
            conv5 = conv5 * correction_factor + tf.stop_gradient(-conv5 * correction_factor + conv5)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_5 = tf.nn.max_pool(conv5, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')     

        # Layer6
        conv6 = tf.nn.conv2d(pool_5, conv6_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv6 = tf.nn.batch_normalization(conv6, mean=tf.reduce_mean(conv6),  variance=tf.math.reduce_variance(conv6),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv6 = tf.nn.relu(conv6)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv6_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv6_w), [0,1,2])),1,2))
            conv6 = conv6 * correction_factor + tf.stop_gradient(-conv6 * correction_factor + conv6)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_6 = tf.nn.max_pool(conv6, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')     


        # Layer7
        conv7 = tf.nn.conv2d(pool_6, conv7_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv7 = tf.nn.batch_normalization(conv7, mean=tf.reduce_mean(conv7),  variance=tf.math.reduce_variance(conv7),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv7 = tf.nn.relu(conv7)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv7_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv7_w), [0,1,2])),1,2))
            conv7 = conv7 * correction_factor + tf.stop_gradient(-conv7 * correction_factor + conv7)
        
        # Layer8
        conv8 = tf.nn.conv2d(conv7, conv8_w, strides = [1,1,1,1], padding = 'SAME')
        # Batch Norm
        if batch_norm:
            conv8 = tf.nn.batch_normalization(conv8, mean=tf.reduce_mean(conv8),  variance=tf.math.reduce_variance(conv8),
                                              offset=None, scale=None, variance_epsilon=1e-6)
        # Activation.
        conv8 = tf.nn.relu(conv8)
        # GRAPES conv 
        if grapes_conv:
            correction_factor = tf.stop_gradient(tf.clip_by_value(2 * tf.reduce_sum(tf.abs(conv8_w), [0,1,2]) / tf.reduce_max(tf.reduce_sum(tf.abs(conv8_w), [0,1,2])),1,2))
            conv8 = conv8 * correction_factor + tf.stop_gradient(-conv8 * correction_factor + conv8)
        
        # Residual connection 1
        conv8 += pool_6
        
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_8 = tf.nn.max_pool(conv8, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')     
   
        # Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_8)
        
        # HERE DROPOUT?
        fc1 = tf.nn.dropout(fc1, keep_prob=0.8)
        
        # Layer 9
        logits = tf.matmul(fc1,fc1_w) 
        
        return logits
    
    
    # training pipeline
    if dataset_mnist:
        input_channels = 1
    elif dataset_cifar10:
        input_channels = 3
    elif dataset_cifar100:
        input_channels = 3
    x = tf.placeholder(tf.float32, (None, 32, 32, input_channels))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, nout)
    
    is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
    logits = ResNet9(x,input_channels,grapes_fc,grapes_fc_local,grapes_conv,grapes_conv_local,
                   is_training=is_training)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    if optim == "GD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optim == "mom":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif optim == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #saver = tf.train.Saver()
    
    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, mini_batch_size):
            batch_x, batch_y = X_data[offset:offset+mini_batch_size], y_data[offset:offset+mini_batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, is_training:False, learning_rate: learning_rate_start})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    
    val_acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(num_epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, mini_batch_size):
                end = offset + mini_batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training:True, learning_rate: learning_rate_start*(decay ** (i // decay_step))})
                
            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.4f}".format(validation_accuracy))
            val_acc.append(validation_accuracy)
            # save results to file
            np.savetxt(savepath+'/val_acc_curve.txt',val_acc)
            
        #saver.save(sess, './resnet9')
        #print("Model saved")
        
    #with tf.Session() as sess:
        #saver.restore(sess, tf.train.latest_checkpoint('.'))
        print()
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.4f}".format(test_accuracy))
        # save results to file
        np.savetxt(savepath+'/test_acc.txt',np.array([test_accuracy]))
    
