# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:02:22 2020

@author: GiorgiaDellaferrera
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy_grapes_functions import *
import argparse
from sklearn.model_selection import train_test_split

plt.close('all')
start_time = time.time()


# ask for the arguments
parser = argparse.ArgumentParser()

parser.add_argument('-en', '--exp_name',
                    type=str, default='exp', 
                    help="Experiment name")
parser.add_argument('-nf', '--nested_folder',
                    type=str, default='ignore', 
                    help="nested folder where to save the saving folder")
parser.add_argument('-r', '--n_runs',
                    type=int, default=1,
                    help="Number of simulations for each model")
parser.add_argument('-trep', '--train_epochs',
                    type=int, default= 1,
                    help="Number of training epochs")
parser.add_argument('-sp', '--sample_passes',
                    type=int, default=1,
                    help="Number of consecutive passes for each sample")
parser.add_argument('-ns', '--n_samples',
                    default='all',
                    help="Size of training set. Choose between an integer or 'all' ")
parser.add_argument('-eta', '--eta',
                    type=float, default=0.1, 
                    help="Learning rate")
parser.add_argument('-do', '--dropout',
                    type=float, default=0.9, 
                    help="Dropout")
parser.add_argument('-no_shuffling','--no_shuffling', 
                    default=False, action='store_true',
                    help="Choose not to do any shuffling")
parser.add_argument('-zm', '--zeromean',
                    action='store_true', 
                    help="Rescale the dataset to the interval [-1,1]")
parser.add_argument('-eta_m', '--eta_m',
                    action='store_true', 
                    help="Multiply learning rate in BP by mean of BP7 factor")
parser.add_argument('-eta_one', '--eta_one',
                    action='store_true', 
                    help="Divide learning rate in BP7 by mean of BP7 factor, to keep the average")
parser.add_argument('-eta_d', '--eta_decay',
                    action='store_true', 
                    help="If True, eta is decreased by 25% every 15 epochs")
parser.add_argument('-def', '--deformation',
                    action='store_true', 
                    help="If True, deformations are applied to the images at each epoch")
parser.add_argument('-notav', '--no_test_as_val',
                    action='store_true', 
                    help="If True, test set is used as validation set")
parser.add_argument('-ct', '--continue_training',
                    action='store_true', 
                    help="If True, training is continued from some saved weights")
parser.add_argument('-ct_path', '--continue_training_path',
                    default='res_exp_BP_v7_SGD_1_cir_un_do100_mnist_rep1_tr20_pass1',type=str, 
                    help="Path containing weights to continue training from")
parser.add_argument('-seed', '--seed',
                    default=None, 
                    help="Random seed. Set to None or to integer")
parser.add_argument('-mn', '--mnist', action='store_true',
                    help="use mnist as dataset")
parser.add_argument('-emn', '--emnist', action='store_true',
                    help="use emnist (balanced) as dataset")
parser.add_argument('-fmn', '--fmnist', action='store_true',
                    help="use fashion mnist as dataset")
parser.add_argument('-cif', '--cifar10', action='store_true',
                    help="use cifar10 as dataset")
parser.add_argument('-V', '--VERBOSE', action='store_true',
                    help="print some extra variables and results")
parser.add_argument('-pl', '--plots', action='store_true',
                    help="plot extra figures")
parser.add_argument('-val', '--validation', action='store_true',
                    help="perform validation")
parser.add_argument('-lt', '--learn_type',
                    type=str, default='BPgrapes', 
                    help="Learning rule. Choose between BP,BPgrapes,FA,FAgrapes,DFA,DFAgrapes")
parser.add_argument('-ut', '--update_type',
                    type=str, default='SGD', 
                    help="Update type: SGD, mom(entum), NAG, rmsprop, Adam ...")
parser.add_argument('-bs', '--batch_size',
                    default=64,type=int,
                    help="Batch size during training. Choose an integer")
parser.add_argument('-kv', '--keep_variants',
                    type=str, default='un', 
                    help="Keep variants: ul (until learning), ue (until end) or un (until normalization)")
parser.add_argument('-win', '--w_init',
                    type=str, default='he_uniform', 
                    help="Weight initialization type. Options: rnd, zero, xav, he, he_uniform, nok, cir")
parser.add_argument('-build', '--build',
                    type=str, default='auto', 
                    help="Building mode: auto or custom")
parser.add_argument('-arch', '--architecture',
                    default=[784,2000,1500,1000,500,10], 
                    help="Network layers size")
parser.add_argument('-act', '--act_list',
                    default=[relu,relu,relu,relu,softmax], 
                    help="Network layers activations")
parser.add_argument('-struct', '--struct',
                    type=str, default='uniform', 
                    help="Network structure: pyramidal or uniform")
parser.add_argument('-ss', '--start_size',
                    type=int, default=256, 
                    help="Size of 1st hidden layer")
parser.add_argument('-nh', '--n_hlayers',
                    type=int, default=2, 
                    help="Number of hidden layers")
parser.add_argument('-act_h', '--act_hidden',
                    default='relu',type=str, 
                    help="Activation of hidden layers")
parser.add_argument('-act_o', '--act_out',
                    default='softmax',type=str, 
                    help="Activation of output layer")
args = parser.parse_args()

mnist = True

# save the arguments
# simulation set-up
exp_name = args.exp_name
nested_folder = args.nested_folder
n_runs = args.n_runs
train_epochs = args.train_epochs
sample_passes = args.sample_passes
n_samples = args.n_samples
eta = args.eta
dropout = args.dropout
no_shuffling = args.no_shuffling
zeromean = args.zeromean
#zeromean = True
if no_shuffling == False:
    shuffling = True
    print("shuffling")
else:
    shuffling = False
    print("no shuffling")
dropout_perc = int(dropout*100)
eta_m = args.eta_m
eta_one = args.eta_one
eta_decay = args.eta_decay
deformation = args.deformation
no_test_as_val = args.no_test_as_val
if no_test_as_val:
    test_as_val = False
else:
    test_as_val = True
continue_training = args.continue_training
continue_training_path = args.continue_training_path
seed = args.seed
mnist = args.mnist
emnist = args.emnist
fmnist = args.fmnist
cifar10 = args.cifar10
# check that one dataset has been chosen
if mnist == emnist == True or mnist == fmnist == True or emnist == fmnist == True or mnist == cifar10 == True or fmnist == cifar10 == True or emnist == cifar10 == True:
    print('Warning, two datasets have been chosen')
    print('Setting mnist as dataset')
    mnist = True
    fmnist = False
    emnist = False
    cifar10= False
if mnist == emnist == fmnist == cifar10 == False:
    print('Warning, one dataset should be chosen')
    print('Setting mnist as dataset')
    mnist = True

w_init = args.w_init
VERBOSE = args.VERBOSE
plots = args.plots
validation = args.validation
# network set-up
learn_type = args.learn_type # current options are BP,BPgrapes,FA,FAgrapes,DFA,DFAgrapes
update_type = args.update_type # current options are SGD, mom(entum), NAG, rmsprop, Adam
batch_size = args.batch_size
keep_variants = args.keep_variants
build = args.build
if build == 'auto':
    struct = args.struct # pyramidal or uniform
    start_size = args.start_size # e.g. 256
    n_hlayers = args.n_hlayers # e.g. 2
    act_hidden = args.act_hidden
    act_out = args.act_out
    if act_hidden == 'sigm':
        act_hidden = sigm
    elif act_hidden == 'relu':
        act_hidden = relu
    elif act_hidden == 'Lrelu':
        act_hidden = Lrelu
    elif act_hidden == 'tanh':
        act_hidden = tanh
    elif act_hidden == 'tanh_ciresan':
        act_hidden = tanh_ciresan
    elif act_hidden == 'step_f':
        act_hidden = step_f
    elif act_hidden == 'softmax':
        act_hidden = softmax
    if act_out == 'sigm':
        act_out = sigm
    elif act_out == 'relu':
        act_out = relu
    elif act_out == 'Lrelu':
        act_out = Lrelu
    elif act_out == 'tanh':
        act_out = tanh
    elif act_out == 'tanh_ciresan':
        act_out = tanh_ciresan
    elif act_out == 'step_f':
        act_out = step_f
    elif act_out == 'softmax':
        act_out = softmax
    if mnist or emnist or fmnist:
        layers_size = [784]
    elif cifar10:
        layers_size = [3072]
    act_list = []
    size_next = start_size
    for h in range(n_hlayers):
        layers_size.append(size_next)
        act_list.append(act_hidden)
        if struct == 'pyramidal':
            size_next = int(size_next/2)
        elif struct == 'uniform':
            pass
    if mnist or fmnist or cifar10:
        layers_size.append(10)
    elif emnist:
        layers_size.append(47)
    act_list.append(act_out)
    
elif build == 'custom':
    layers_size = args.architecture
    act_list = args.act_list

print(act_list)
print(layers_size)
#check size and create list of derivatives of activations
try:  
    a = len(layers_size)-1
    b = len(act_list)
    assert a == b
except AssertionError:  
        print ("Assertion Exception Raised.")
else:  
    print ("layer size and number of activations correctly set up!")
d_act_list = []
for idx,a in enumerate(act_list):
    if a == sigm:
        d_act_list.append(d_sigm)
    elif a == relu:
        d_act_list.append(step_f)
    elif a == Lrelu:
        d_act_list.append(d_Lrelu)
    elif a == tanh:
        d_act_list.append(d_tanh)
    elif a == tanh_ciresan:
        d_act_list.append(d_tanh_ciresan)
    elif a == step_f:
        d_act_list.append(d_step_f)
    elif a == softmax:
        d_act_list.append(None)

# create folder to save all results
if mnist:
    arch_name = 'mnist'
elif emnist:
    arch_name = 'emnist'
elif fmnist:
    arch_name = 'fmnist'
elif cifar10:
    arch_name = 'cifar10'
savepath = "res_"+exp_name+"_"+learn_type+"_"+update_type+"_"+str(batch_size)+"_"+w_init+"_"+keep_variants+"_"+"do"+str(dropout_perc)+"_"+arch_name+"_rep"+str(n_runs)+"_tr"+str(train_epochs)+"_pass"+str(sample_passes)
if eta_decay:
    savepath = savepath + "_etadec"
if nested_folder is not "ignore":
    savepath = nested_folder + '/' + savepath
if deformation:
    savepath = savepath + "_def"
if test_as_val:
    savepath = savepath + "_tav"
if continue_training:
    savepath = savepath + "_ct"
try:
    os.mkdir(savepath)
except OSError:
    print ("Creation of the directory %s failed" % savepath)
else:
    print ("Successfully created the directory %s " % savepath)
    
# create variables to store results
train_acc_all = np.zeros((n_runs,train_epochs))
val_acc_all = np.zeros((n_runs,train_epochs))
test_acc_all = []
    
# loop over the number of simulations
for r in range(n_runs):
    print('####### RUN {} #######'.format(r))
    t0 = time.time()
    net = general_network(layers_size,act_list,d_act_list,learn_type,batch_size,update_type,keep_variants,w_init,VERBOSE) 
    if continue_training:
        for i in range(len(layers_size)-1):
            weights = np.loadtxt(continue_training_path+'/weights_layer'+str(i)+'.txt')
            net.layers[i].w = weights
    if mnist:
        x_list, target_list, x_list_test, target_list_test = dataset_mnist(n_samples,seed,plots=False)
    elif emnist:
        x_list, target_list, x_list_test, target_list_test = dataset_emnist(n_samples,seed,plots=False)
    elif fmnist:
        x_list, target_list, x_list_test, target_list_test = dataset_fmnist(n_samples,seed,plots=False)
    elif cifar10:
        x_list, target_list, x_list_test, target_list_test = dataset_cifar(n_samples,seed,plots=False)
                
    # train the model 
    E_curve, train_acc, val_acc = net.train(x_list,target_list,x_list_test,target_list_test,train_epochs,sample_passes,eta,dropout,shuffling,eta_decay,deformation,test_as_val,zeromean,plots,validation,savepath,r)
    t1 = time.time()
    print('Running time for train: {}'.format(np.round(t1-t0,2)))
    # test the model
    t0 = time.time()
    test_acc = net.test(x_list_test,target_list_test,plots)
    test_acc = np.array([test_acc])
    print('Final accuracy = {}'.format(test_acc))
    t1 = time.time()
    print('Running time for test: {}'.format(np.round(t1-t0,2)))
    # save the results for this network
    np.savetxt(savepath+'/train_acc_run'+str(r)+'.txt',train_acc)
    np.savetxt(savepath+'/test_acc_run'+str(r)+'.txt',test_acc)
    
    train_acc_all[r,:] = train_acc
    test_acc_all.append(test_acc)
    if validation:
        val_acc_all[r,:] = val_acc
    
    # save the results of the runs until the current one
    np.savetxt(savepath+'/train_acc_tot_rep'+str(r)+'.txt',train_acc_all[0:r+1,:])
    np.savetxt(savepath+'/test_acc_tot_rep'+str(r)+'.txt',test_acc_all)
    if validation:
        np.savetxt(savepath+'/val_acc_tot_rep'+str(r)+'.txt',val_acc_all[0:r+1,:])
                
# save the final train and test curves
np.savetxt(savepath+'/train_acc_tot.txt',train_acc_all)
np.savetxt(savepath+'/test_acc_tot.txt',test_acc_all)
train_acc_mean = np.mean(train_acc_all,axis=0)
train_acc_std = np.std(train_acc_all,axis=0)
test_acc_mean = np.mean(test_acc_all)
test_acc_std = np.std(test_acc_all)
if validation:
    np.savetxt(savepath+'/val_acc_tot.txt',val_acc_all)
    val_acc_mean = np.mean(val_acc_all,axis=0)
    val_acc_std = np.std(val_acc_all,axis=0)
    

