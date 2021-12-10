# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:02:06 2020

@author: GiorgiaDellaferrera
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# activation functions
def sigm(x):
    return 1/(1+np.exp(-x))
def d_sigm(x):
    return sigm(x) * (1-sigm(x))
def relu(x):
    return np.maximum(x,0)
def step_f(x,bias=0):
    return np.heaviside(x,bias)
def Lrelu(x,leakage=0.1):
    output = np.copy(x)
    output[output<0] *= leakage
    return output
def d_Lrelu(x,leakage=0.1):
    return np.clip(x>0,leakage,1.0)
def d_step_f(x):
    return 1-np.square(np.tanh(x))
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1-np.square(tanh(x))
def tanh_ciresan(x):
    A = 1.7159
    B = 0.6666
    return np.tanh(B*x)*A
def d_tanh_ciresan(x):
    A = 1.7159
    B = 0.6666
    return A*B*(1-np.square(tanh(B*x)))
def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
def onehotenc(idx,size):
    arr = np.zeros((size,1))
    arr[idx] = 1
    return arr
# to compute alignment of matrix
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1 = 2.*(v1 - np.min(v1))/np.ptp(v1)-1
    v2 = 2.*(v2 - np.min(v2))/np.ptp(v2)-1
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

# prepare dataset
def dataset_mnist(n_samples,seed=None,plots=False):
    print('Loading mnist')
    x_max = 255
    x_min = 0
    # import mnist
    import keras
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using digits:")
        print(idx_list)
    else:
        print("using the full mnist dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 10
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    # train
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
    # test
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_emnist(n_samples,seed=None,plots=False):
    print('Loading emnist')
    x_max = 255
    x_min = 0
    # import emnist balanced
    from extra_keras_datasets import emnist
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')

    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using letters:")
        print(idx_list)
    else:
        print("using the full emnist dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 47
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_fmnist(n_samples,seed=None,plots=False):
    print('Loading fmnist')
    x_max = 255
    x_min = 0
    # import mnist
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using clothes:")
        print(idx_list)
    else:
        print("using the full fmnist dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 10
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure()
            plt.imshow(x.reshape((int(np.sqrt(n_input)),int(np.sqrt(n_input)))))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def dataset_cifar(n_samples,seed=None,plots=False):
    print('Loading cifar10')
    x_max = 255
    x_min = 0
    # import cifar10
    import keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    class_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
    if n_samples is not 'all':
        if seed is not None:
            np.random.seed(seed)
        i = np.random.choice(len(y_train)-n_samples-1)
        #print(i)
        x_l = x_train[i:n_samples+i,:,:]
        idx_list = y_train[i:n_samples+i]
        print("using classes:")
        print(idx_list)
        print("corresponding to labels:")
        for id_ in idx_list:
            print(class_labels[id_[0]])
    else:
        print("using the full cifar10 dataset")
        x_l = x_train
        idx_list = y_train
        x_l_test = x_test
        idx_list_test = y_test
    # one input neuron encodes one pixel    
    n_input = np.size(x_l[0])
    n_output = 10
    # flattening for samples and one-hot encoding for targets
    target_list = []
    x_list = []
    target_list_test = []
    x_list_test = []
    for idx,t in enumerate(idx_list):
        x = x_l[idx].reshape((np.size(x_l[idx]),1))
        x = (x-x_min)/(x_max-x_min)
        x_list.append(x)
        target = np.zeros((n_output,1))
        target[t] = 1
        target_list.append(target)
        if plots:        
            plt.figure(figsize=(2,2))
            plt.imshow(x.reshape((int(np.sqrt(n_input/3)),int(np.sqrt(n_input/3)),3)))
    if n_samples is not 'all':
        x_list_test = x_list
        target_list_test = target_list
    else:
        for idx,t in enumerate(idx_list_test):
            x = x_l_test[idx].reshape((np.size(x_l_test[idx]),1))
            x = (x-x_min)/(x_max-x_min)
            x_list_test.append(x)
            target = np.zeros((n_output,1))
            target[t] = 1
            target_list_test.append(target)
        
    return x_list, target_list, x_list_test, target_list_test

def distortion(x_train, y_train):
    #from keras.preprocessing.image import ImageDataGenerator
    from image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        elastic_RGB=[34,6],
        fill_mode='constant',
        vertical_flip=False,
        horizontal_flip=False)
    #print('before',np.shape(x_train))
    #print('before mean',np.mean(x_train))
    x_train = np.reshape(x_train,(np.shape(x_train) + (1,)))
    x_train = np.reshape(x_train,(np.shape(x_train)[0],int(np.sqrt(np.shape(x_train)[1])),int(np.sqrt(np.shape(x_train)[1])),1))
    #print('input shape',np.shape(x_train))
    datagen.fit(x_train)
    batches = 0
    for x_deformed, y_deformed in datagen.flow(x_train, y_train, batch_size=60000):
        batches += 1
        if batches >= 1:
            break
    #x_plot = np.reshape(x_deformed,(np.shape(x_deformed)[0],np.shape(x_deformed)[1],np.shape(x_deformed)[2]))
    #plt.figure()
    #plt.imshow(x_plot[0,:,:])
    #plt.colorbar()
    x_deformed = np.reshape(x_deformed,(np.shape(x_deformed)[0],np.shape(x_deformed)[1]*np.shape(x_deformed)[2],1))
    #print('after',np.shape(x_deformed))
    #print('after mean',np.mean(x_deformed))
    x_max = np.max(x_deformed)
    x_min = np.min(x_deformed)
    x_deformed = (x_deformed-x_min)/(x_max-x_min)
    #print('after normaliz',np.mean(x_deformed))
    return x_deformed, y_deformed

# layer class
class general_layer():
    def __init__(self,input_size,output_size,act,d_act,w_init):
        self.input_size = input_size
        self.output_size = output_size
        self.act = act
        self.d_act = d_act
        self.sum_accumulator = []
        self.w_accumulator = []
        self.w_pre_sum = []
        self.w_next_sum = []
        if w_init == 'rnd':
            print('Random weight initialization')
            self.w = np.random.rand(output_size,input_size)/ \
                        np.sqrt(input_size)
        elif w_init == 'zero':
            print('Zero weight initialization')
            self.w = np.zeros((output_size,input_size))
        elif w_init == 'xav':
            print('Xavier weight initialization (uniform)')
            nin = input_size; nout = output_size
            sd = np.sqrt(6.0 / (nin + nout))
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-sd, sd))
                    self.w[j,i] = x
        elif w_init == 'he':
            print('Kaiming He weight initialization (normal)')
            nin = input_size; nout = output_size
            sd = np.sqrt(2.0 / nin)
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.normal(loc = mu, scale = sd))
                    self.w[j,i] = x
        elif w_init == 'he_uniform':
            print('Kaiming He weight initialization (uniform)')
            nin = input_size; nout = output_size
            limit = np.sqrt(6.0 / nin)
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-limit, limit))
                    self.w[j,i] = x
        elif w_init == 'nok':
            print('Nokland weight initialization')
            nin = input_size; nout = output_size
            sd = 1.0 / np.sqrt(nin)
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-sd,sd))
                    self.w[j,i] = x
        if w_init == 'cir':
            print('Ciresan weight initialization')
            nin = input_size; nout = output_size
            sd = 0.05
            mu = 0.0
            self.w = np.zeros((output_size,input_size))
            for i in range(nin):
                for j in range(nout):
                    x = np.float32(np.random.uniform(-sd,sd))
                    self.w[j,i] = x  
        #self.delta_w = np.zeros_like(self.w)
        self.delta_w_batch = np.zeros_like(self.w)
        
        #self.sqr_grad = np.zeros_like(self.w)
    def forward(self,x,dropout,training,update_type):
        self.x = x
        if update_type is not 'NAG':
            self.a = self.w @ x
        elif update_type == 'NAG':
            self.a = self.w_lookahead @ x
        self.output = self.act(self.a)
        # apply dropout
        if dropout != 1.0:
            if training:
                # commented is the old version
                # self.drop_mask = np.random.binomial(1,dropout,size = np.shape(self.a))
                self.drop_mask = np.random.binomial(1,dropout,size = np.shape(self.a))/dropout
            else:
                # commented is the old version
                # self.drop_mask = dropout
                self.drop_mask = 1.
            self.output *= self.drop_mask
        return self.output
        
# network class
class general_network():
    def __init__(self,layers_size,act_list,d_act_list,learn_type,batch_size,update_type,keep_variants,w_init,VERBOSE=False):
        self.layers_size = layers_size
        self.n_layers = len(self.layers_size)
        print(self.n_layers)
        self.learn_type = learn_type
        self.batch_size = batch_size
        self.update_type = update_type
        self.keep_variants = keep_variants
        self.layers = []
        for i in range(self.n_layers-1):
            new_layer = general_layer(self.layers_size[i],self.layers_size[i+1],act_list[i],d_act_list[i],w_init)
            self.layers.append(new_layer)
        # if needed in the model, initialize the B matrix(-ces)
        if self.learn_type in ['BP','BPgrapes']:
            pass
        elif self.learn_type in ['FA','FAgrapes']:
            for idx in range(len(self.layers)-1,0,-1):
                if w_init == 'nok':
                    print('Nokland weight initialization for FA')
                    nin = self.layers[idx].input_size
                    nout = self.layers[idx].output_size
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers[idx].output_size))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                elif w_init == 'he_uniform':
                    print('Kaiming He weight initialization (uniform) for feedback weights')
                    nin = self.layers[idx].input_size
                    nout = self.layers[idx].output_size
                    limit = np.sqrt(6.0 / nin)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers[idx].output_size))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-limit, limit))
                            self.layers[idx].B[i,j] = x  
                else:
                    print('Nokland weight initialization for FA')
                    nin = self.layers[idx].input_size
                    nout = self.layers[idx].output_size
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers[idx].output_size))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                            
                        
            #self.layers[idx].B = np.random.rand(self.layers[idx].input_size,self.layers[idx].output_size)    
        elif self.learn_type in ['DFA','DFAgrapes']:
            for idx in range(len(self.layers)-1,0,-1):
                if w_init == 'nok':
                    print('Nokland weight initialization for DFA')
                    nin = self.layers[idx].input_size
                    nout = self.layers_size[-1]
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers_size[-1]))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                elif w_init == 'he_uniform':
                    print('Kaiming He weight initialization for DFA')
                    nin = self.layers[idx].input_size
                    nout = self.layers_size[-1]
                    limit = np.sqrt(6.0 / nin)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers_size[-1]))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-limit, limit))
                            self.layers[idx].B[i,j] = x         
                else:
                    print('Nokland weight initialization for DFA')
                    nin = self.layers[idx].input_size
                    nout = self.layers_size[-1]
                    sd = 1.0 / np.sqrt(nout)
                    mu = 0.0
                    self.layers[idx].B = np.zeros((self.layers[idx].input_size,self.layers_size[-1]))
                    for i in range(nin):
                        for j in range(nout):
                            x = np.float32(np.random.uniform(-sd, sd))
                            self.layers[idx].B[i,j] = x
                #self.layers[idx].B = np.random.rand(self.layers[idx].input_size,self.layers_size[-1])        
        elif self.learn_type in ['IFA','IFAgrapes']:
            if w_init == 'nok':
                print('Nokland weight initialization for IFA')
                nin = self.layers[1].input_size
                nout = self.layers_size[-1]
                sd = 1.0 / np.sqrt(nout)
                mu = 0.0
                self.layers[0].B = np.zeros((self.layers[1].input_size,self.layers_size[-1]))
                for i in range(nin):
                    for j in range(nout):
                        x = np.float32(np.random.uniform(-sd, sd))
                        self.layers[0].B[i,j] = x
            elif w_init == 'he_uniform':
                print('Kaiming He weight initialization for IFA')
                nin = self.layers[1].input_size
                nout = self.layers_size[-1]
                limit = np.sqrt(6.0 / nin)
                mu = 0.0
                self.layers[0].B = np.zeros((self.layers[1].input_size,self.layers_size[-1]))
                for i in range(nin):
                    for j in range(nout):
                        x = np.float32(np.random.uniform(-limit, limit))
                        self.layers[0].B[i,j] = x
            else:
                print('Nokland weight initialization for IFA')
                nin = self.layers[1].input_size
                nout = self.layers_size[-1]
                sd = 1.0 / np.sqrt(nout)
                mu = 0.0
                self.layers[0].B = np.zeros((self.layers[1].input_size,self.layers_size[-1]))
                for i in range(nin):
                    for j in range(nout):
                        x = np.float32(np.random.uniform(-sd, sd))
                        self.layers[0].B[i,j] = x
            #self.layers[0].B = np.random.rand(self.layers[1].input_size,self.layers_size[-1])
        # initialize variables according to the optimizer
        print('Update type is:'+self.update_type)
        if self.update_type == 'SGD':
            pass
        elif self.update_type =='mom':
            print('setting up parameters for momentum optimizer')
            self.gamma = 0.9
            for l in self.layers:
                l.v_w = np.zeros((l.output_size,l.input_size))
        elif self.update_type == 'NAG':
            print('setting up parameters for NAG optimizer')
            self.gamma = 0.9
            for l in self.layers:
                l.v_w = np.zeros((l.output_size,l.input_size))
                l.w_lookahead = np.copy(l.w)
        elif self.update_type =='rmsprop':
            print('setting up parameters for rmsprop optimizer')
            for l in self.layers:
                l.sqr_grad = np.zeros_like(l.w)
        elif self.update_type =='Adam':
            print('setting up parameters for Adam optimizer')
            for l in self.layers:
                #initialize l.vel, l.sqr, l.t
                l.vel = np.zeros_like(l.w)
                l.sqr = np.zeros_like(l.w)
                l.timestep = 1
        
    def forward(self,x,target,dropout,training):
        self.target=target
        for l in self.layers[:-1]:
            x = l.forward(x,dropout,training,self.update_type)
        x = self.layers[-1].forward(x,dropout=1.0,training=training,update_type=self.update_type)
        self.output = x
        self.error = self.output-target
        return self.output,self.error

    def learning(self,error,eta,dropout):
        # compute delta_a (depends on the method)
        
        self.layers[-1].delta_a = error
        
        if self.learn_type=='BP':
            for idx in range(len(self.layers)-2,-1,-1):
                if self.update_type is not 'NAG':
                    self.layers[idx].delta_a = (self.layers[idx+1].w.T @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                elif self.update_type == 'NAG':
                    self.layers[idx].delta_a = (self.layers[idx+1].w_lookahead.T @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                                
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask                
        elif self.learn_type=='FA':
            for idx in range(len(self.layers)-2,-1,-1):
                self.layers[idx].delta_a = (self.layers[idx+1].B @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask 
        elif self.learn_type=='DFA':
            for idx in range(len(self.layers)-2,-1,-1):
                self.layers[idx].delta_a = (self.layers[idx+1].B @ error) * self.layers[idx].d_act(self.layers[idx].a)
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask 
        elif self.learn_type=='IFA':
            self.layers[0].delta_a = (self.layers[0].B @ error) * self.layers[0].d_act(self.layers[0].a)
            for idx in range(1,len(self.layers)-1):
                if self.update_type is not 'NAG':
                    self.layers[idx].delta_a = (self.layers[idx].w @ self.layers[idx-1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                elif self.update_type == 'NAG':
                    self.layers[idx].delta_a = (self.layers[idx].w_lookahead @ self.layers[idx-1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask 
                    

        elif self.learn_type=='IFAgrapes':
            # takes into account the sum of the incoming weights to the postsynaptic population of the current layer
            # incorporates the local information in the error and propagates it
            # FIRST LAYER
            if self.new_batch:
                if self.update_type is not 'NAG':
                    if dropout != 1.0:
                        self.w_next_sum = np.array([np.sum(np.abs(self.layers[0].drop_mask * self.layers[0].w),axis = 1),])
                    else:
                        self.w_next_sum = np.array([np.sum(np.abs(self.layers[0].w),axis = 1),])
                elif self.update_type == 'NAG':
                    if dropout != 1.0:
                        self.w_next_sum = np.array([np.sum(np.abs(self.layers[0].drop_mask * self.layers[0].w_lookahead),axis = 1),])
                    else:
                        self.w_next_sum = np.array([np.sum(np.abs(self.layers[0].w_lookahead),axis = 1),])
                self.w_next_sum = 2*(self.w_next_sum)/(np.max(self.w_next_sum))
            
            self.layers[0].delta_a = self.w_next_sum.T * (self.layers[0].B @ error) * self.layers[0].d_act(self.layers[0].a)
            #print('I_v1 delta a: max={} min={} mean={}'.format(np.max(self.layers[0].delta_a),np.min(self.layers[0].delta_a),np.mean(self.layers[0].delta_a)))
            # HIDDEN LAYERS
            for idx in range(1,len(self.layers)-1):
                if self.new_batch:
                    if self.update_type is not 'NAG':
                        if dropout != 1.0:
                            self.w_next_sum = np.array([np.sum(np.abs(self.layers[idx].drop_mask * self.layers[idx].w),axis = 1),])
                        else:
                            self.w_next_sum = np.array([np.sum(np.abs(self.layers[idx].w),axis = 1),])
                    elif self.update_type == 'NAG':        
                        if dropout != 1.0:
                            self.w_next_sum = np.array([np.sum(np.abs(self.layers[idx].drop_mask * self.layers[idx].w_lookahead),axis = 1),])
                        else:
                            self.w_next_sum = np.array([np.sum(np.abs(self.layers[idx].w_lookahead),axis = 1),])        
                    self.w_next_sum = 2*(self.w_next_sum)/(np.max(self.w_next_sum))
                
                if self.update_type is not 'NAG':
                    self.layers[idx].delta_a = self.w_next_sum.T *(self.layers[idx].w @ self.layers[idx-1].delta_a) * self.layers[idx].d_act(self.layers[idx].a) 
                elif self.update_type == 'NAG':
                    self.layers[idx].delta_a = self.w_next_sum.T *(self.layers[idx].w_lookahead @ self.layers[idx-1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)                 
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask 
            
                    
        elif self.learn_type=='DFAgrapes':
            if self.new_batch:
                self.layers[-1].w_pre_sum = np.ones_like(np.array([np.sum(np.abs(self.layers[-1].w),axis = 0),]))
            for idx in range(len(self.layers)-2,0,-1):
                if self.new_batch:
                    if dropout != 1.0:
                        self.layers[idx].w_pre_sum = np.array([np.sum(np.abs(self.layers[idx].drop_mask * self.layers[idx].w),axis = 0),])
                    else:
                        self.layers[idx].w_pre_sum = np.array([np.sum(np.abs(self.layers[idx].w),axis = 0),])
                    self.layers[idx].w_pre_sum = 2*(self.layers[idx].w_pre_sum)/(np.max(self.layers[idx].w_pre_sum))
                self.layers[idx].delta_a = (self.layers[idx+1].w_pre_sum.T * self.layers[idx+1].B @ error) * self.layers[idx].d_act(self.layers[idx].a) * self.layers[idx].w_pre_sum
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask  
            self.layers[0].delta_a = (self.layers[0+1].B @ error) * self.layers[0].d_act(self.layers[0].a) 
            if dropout != 1.0:
                self.layers[0].delta_a *= self.layers[0].drop_mask  
  
            
        elif self.learn_type=='BPgrapes':
            for idx in range(len(self.layers)-2,-1,-1):
                if self.new_batch:
                    if self.update_type is not 'NAG':
                        if dropout != 1.0:
                            self.layers[idx].w_next_sum = np.array([np.sum(np.abs(self.layers[idx].drop_mask * self.layers[idx].w),axis = 1),])
                        else:
                            self.layers[idx].w_next_sum = np.array([np.sum(np.abs(self.layers[idx].w),axis = 1),])
                    elif self.update_type == 'NAG':
                        if dropout != 1.0:
                            self.layers[idx].w_next_sum = np.array([np.sum(np.abs(self.layers[idx].drop_mask * self.layers[idx].w_lookahead),axis = 1),])
                        else:
                            self.layers[idx].w_next_sum = np.array([np.sum(np.abs(self.layers[idx].w_lookahead),axis = 1),])
                    self.layers[idx].w_next_sum = 2*(self.layers[idx].w_next_sum)/(np.max(self.layers[idx].w_next_sum))               
                    if self.keep_variants == 'un':
                        self.layers[idx].w_next_sum = np.clip(self.layers[idx].w_next_sum,1.0,2.0)
                if self.update_type is not 'NAG':
                    self.layers[idx].delta_a =  self.layers[idx].w_next_sum.T *(self.layers[idx+1].w.T @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                elif self.update_type == 'NAG':
                    self.layers[idx].delta_a =  self.layers[idx].w_next_sum.T *(self.layers[idx+1].w_lookahead.T @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask 
                #print("w next sum ", np.shape(self.layers[idx].w_next_sum))
                #print("weights ", np.shape(self.layers[idx].w))
                #print("delta a ", np.shape(self.layers[idx].delta_a))
                #print("drop mask ",np.shape(self.layers[idx].drop_mask))
                
                    
        elif self.learn_type=='FAgrapes':
            for idx in range(len(self.layers)-2,-1,-1):
                if self.new_batch:
                    if dropout != 1.0:
                        self.layers[idx].w_next_sum = np.array([np.sum(np.abs(self.layers[idx].drop_mask * self.layers[idx].w),axis = 1),])
                    else:
                        self.layers[idx].w_next_sum = np.array([np.sum(np.abs(self.layers[idx].w),axis = 1),])
                    self.layers[idx].w_next_sum = 2*(self.layers[idx].w_next_sum)/(np.max(self.layers[idx].w_next_sum))               
                self.layers[idx].delta_a =  self.layers[idx].w_next_sum.T *(self.layers[idx+1].B @ self.layers[idx+1].delta_a) * self.layers[idx].d_act(self.layers[idx].a)
                if dropout != 1.0:
                    self.layers[idx].delta_a *= self.layers[idx].drop_mask 

        
        # weight_update (depends on the chosen optimization method)
        for l in self.layers:
            # dropout 
            #if dropout != 1.0:
            #    l.delta_a *= l.drop_mask                
            # weight update
            if self.update_type == 'SGD':
                #l.delta_w = 
                l.delta_w_batch += -l.delta_a * l.x.T  
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    #print("performing update at {}".format(self.s+1))
                    l.w += eta * l.delta_w_batch / self.batch_size
                    #print("mean before zeroing {}".format(np.mean(l.delta_w_batch)))
                    #print("mean last dw {}".format(np.mean(l.delta_w)))
                    l.delta_w_batch = np.zeros_like(l.w)
                    #print("mean after zeroing {}".format(np.mean(l.delta_w_batch)))
                else:
                    self.new_batch = False
                   
            elif self.update_type == 'rmsprop':
                # step always performed
                l.delta_w_batch += -l.delta_a * l.x.T 
                # step performed only at the end of the minibatch
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    #print("performing update at {}".format(self.s+1))
                    gamma = 0.9
                    eps_stable = 1e-8
                    l.sqr_grad = gamma * l.sqr_grad + (1. - gamma) * np.square(l.delta_w_batch / self.batch_size)
                    l.w += eta * l.delta_w_batch / self.batch_size / np.sqrt(l.sqr_grad + eps_stable)
                    #print("mean before zeroing {}".format(np.mean(l.delta_w_batch)))
                    #print("mean last dw {}".format(np.mean(l.delta_w)))
                    l.delta_w_batch = np.zeros_like(l.w)
                    #print("mean after zeroing {}".format(np.mean(l.delta_w_batch)))
                else:
                    self.new_batch = False
            
            
            elif self.update_type == 'Adam':
                # step always performed
                l.delta_w_batch += -l.delta_a * l.x.T 
                # step performed only at the end of the minibatch
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    #print("performing update at {}".format(self.s+1))
                    beta1 = 0.9
                    beta2 = 0.999
                    eps_stable = 1e-8
                    
                    #print('timestep',l.timestep)
                    grad = l.delta_w_batch / self.batch_size
                    #print('grad',np.mean(grad))
                    l.vel = beta1 * l.vel + (1. - beta1) * grad
                    #print('vel',np.mean(l.vel))
                    l.sqr = beta2 * l.sqr + (1. - beta2) * np.square(grad)
                    #print('sqr',np.mean(l.sqr))
                    vel_bias_corr = l.vel / (1. - beta1**l.timestep)
                    #print('vel bias',np.mean(vel_bias_corr))
                    sqr_bias_corr = l.sqr / (1. - beta2**l.timestep)
                    #print('sqr bias',np.mean(sqr_bias_corr))
                    update = eta * vel_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
                    #print('update',np.mean(update))
                    l.w += update
                    l.timestep += 1
                    
                    #print("mean before zeroing {}".format(np.mean(l.delta_w_batch)))
                    #print("mean last dw {}".format(np.mean(l.delta_w)))
                    l.delta_w_batch = np.zeros_like(l.w)
                    #print("mean after zeroing {}".format(np.mean(l.delta_w_batch)))
                else:
                    self.new_batch = False
                    
                   
            elif self.update_type == 'mom':
                # step always performed
                l.delta_w_batch += l.delta_a * l.x.T 
                # step performed only at the end of the minibatch
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    #print("performing update at {}".format(self.s+1))                    
                    l.v_w = self.gamma * l.v_w + eta * l.delta_w_batch / self.batch_size
                    l.w += -l.v_w
                    #print("mean before zeroing {}".format(np.mean(l.delta_w_batch)))
                    #print("mean last dw {}".format(np.mean(l.delta_w)))
                    l.delta_w_batch = np.zeros_like(l.w)
                    #print("mean after zeroing {}".format(np.mean(l.delta_w_batch)))
                else:
                    self.new_batch = False
                
                
            elif self.update_type == 'NAG':
                # step always performed
                l.delta_w_batch += l.delta_a * l.x.T 
                # step performed only at the end of the minibatch
                if (self.s+1)%self.batch_size == 0:
                    self.new_batch = True
                    #print("performing update at {}".format(self.s+1))   
                    l.v_w = self.gamma * l.v_w + eta * l.delta_w_batch / self.batch_size
                    l.w += -l.v_w
                    #print("mean before zeroing {}".format(np.mean(l.delta_w_batch)))
                    #print("mean last dw {}".format(np.mean(l.delta_w)))
                    l.delta_w_batch = np.zeros_like(l.w)
                    #print("mean after zeroing {}".format(np.mean(l.delta_w_batch)))
                    l.w_lookahead = l.w - self.gamma * l.v_w # da mettere prima
                else:
                    self.new_batch = False
        
    def train(self,x_list,target_list,x_list_test,target_list_test,train_epochs,sample_passes,eta,dropout,shuffling,eta_decay,deformation,test_as_val,zeromean,plots,validation,savepath,r):
        # set validation set differently depending on distortion or not
        if validation:
            if deformation == False and test_as_val == False:
                x_list, x_val, target_list, target_val = train_test_split(x_list, target_list, test_size=0.1, random_state=1)    
            else:
                print('Use full testing set for validation')
                x_val = x_list_test
                target_val = target_list_test
            val_size = len(x_val)
            val_pred_all = []    
            val_targs = []
        self.val_acc_all = []
        dataset_size = len(x_list)
        self.acc_all = []
        self.dropout = dropout
        E_curve = []
        pred_all = []
        stop_training = False
        targs = []
        points = 20
        if deformation:
            x_list_old = x_list
            target_list_old = target_list
        
        # perform the training loop
        for e in range(train_epochs):
            # shuffle the training set
            if shuffling==True and deformation==False:
                x_list, target_list = shuffle(x_list, target_list, random_state = 0)
            # apply deformation
            if deformation:
                x_list, target_list = distortion(x_list_old,target_list_old)
                # normalize to the interval [-1,1] if zeromean is True
                if zeromean:
                    #print("before: min = {} , max = {}".format(np.min(x_list),np.max(x_list)))
                    for i in range(len(x_list)):
                        x_list[i] = x_list[i]*2 - 1
            if train_epochs>9:
                if e%int(train_epochs/10)==0:
                    print('Training epoch {}/{}'.format(e,train_epochs))
            else:
                print('Training epoch {}/{}'.format(e,train_epochs))
            if eta_decay:
                if e%5==0 and e>5:
                    if validation:
                        print('checking val at epoch {}'.format(e))
                        if np.mean(self.val_acc_all[-5:-1]) <= np.mean(self.val_acc_all[-10:-5])+0.0002:
                            eta = np.min((eta*(0.75),1e-6))
                            print("Learning rate at epoch {} decreased to {}".format(e,eta))
                            filename = savepath+'/eta_decrease_epoch'+str(e)+'.txt'
                            file = open(filename,'w')
                            file.write(str(eta))
                            file.close()
                    else:
                        eta = eta*(0.75)
                        print("Learning rate at epoch {} decreased to {}".format(e,eta))
            acc = []
            val_accuracy = []
            self.new_batch = True
            
            for s in range(dataset_size): 
                self.s = s
                x = x_list[s]
                target = target_list[s]
                self.target = target
                for p in range(sample_passes): 
                    if stop_training:
                        break
                    # save the weights
                    this_pass = e*dataset_size + s*sample_passes + p
                    if this_pass%int(train_epochs*dataset_size*sample_passes/points)==0:
                        #print('saving weights at pass {}'.format(this_pass))
                        for idx in range(len(self.layers)-2,-1,-1):
                            self.layers[idx].w_accumulator.append(np.copy(self.layers[idx].w))
              
                    y,self.error = self.forward(x,target,dropout,training=True)
                    targs.append(np.argmax(target))
                    self.learning(self.error,eta,dropout)
                    # save the error
                    E_curve.append(np.sum(abs(self.error)))
                    self.pred = onehotenc(np.argmax(y),np.size(y))
                    pred_all.append(np.argmax(self.pred))
                    if np.argmax(y) == np.argmax(target):
                        acc.append(1)
                    else:
                        acc.append(0)
                      
            self.acc_all.append(np.mean(acc))
            if train_epochs>9:
                if e%int(train_epochs/10)==0:
                    print('Training accuracy = {}'.format(self.acc_all[-1]))
            else:
                print('Training accuracy = {}'.format(self.acc_all[-1]))
             
            if e%1 == 0:    
                np.savetxt(savepath+'/train_acc_tot.txt',np.array([self.acc_all]))
                
            # save the weights
            for i in range(self.n_layers-1):
                #np.savetxt(savepath+'/weights_layer'+str(i)+'.txt',self.layers[i].w)
                pass
            # perform validation
            if validation:
                for s in range(val_size):                           
                    x = x_val[s]
                    target = target_val[s]
                    self.target = target
                    y,self.error = self.forward(x,target,dropout,training=False)
                    # save the error
                    self.pred = onehotenc(np.argmax(y),np.size(y))
                    #print('target {} pred {}'.format(np.argmax(target),np.argmax(self.pred)))
                    val_pred_all.append(np.argmax(self.pred))
                    if np.argmax(y) == np.argmax(target):
                        val_accuracy.append(1)
                    else:
                        val_accuracy.append(0)
                    val_targs.append(np.argmax(target))
                
                self.val_acc_all.append(np.mean(val_accuracy))
                if train_epochs>9:
                    if e%int(train_epochs/10)==0:
                        print('Validation accuracy = {}'.format(self.val_acc_all[-1]))
                else:
                    print('Validation accuracy = {}'.format(self.val_acc_all[-1]))
                if e%1 == 0: 
                    np.savetxt(savepath+'/val_acc_tot.txt',np.array([self.val_acc_all]))
                
            if plots:    
                plt.figure()
                plt.plot(val_pred_all,'*',label='Prediction')
                plt.plot(val_targs,'.',label='Target')
                plt.title(str(self.learn_type)+' - Validation')
                plt.legend() 
        
                
        if plots:
            plt.figure()
            plt.plot(pred_all,'*',label='Prediction')
            plt.plot(targs,'.',label='Target')
            plt.title(str(self.learn_type))
            plt.legend() 
            
        
        return E_curve, self.acc_all, self.val_acc_all
            
    
    def test(self,x_list,target_list,plots,plots_test=False):
        dataset_size = len(x_list)
        pred_all = []    
        targs = []
        accuracy = []
                
        for s in range(dataset_size):   
            if dataset_size>9:
                if s%int(dataset_size/10)==0:
                    print('Testing sample {}/{}'.format(s,dataset_size))
            else:
                print('Testing sample {}/{}'.format(s,dataset_size))
                
            x = x_list[s]
            target = target_list[s]
            self.target = target
            y,self.error = self.forward(x,target,self.dropout,training=False)
            # save the error
            self.pred = onehotenc(np.argmax(y),np.size(y))
            #print('target {} pred {}'.format(np.argmax(target),np.argmax(self.pred)))
            pred_all.append(np.argmax(self.pred))
            if np.argmax(y) == np.argmax(target):
                accuracy.append(1)
            else:
                accuracy.append(0)
            targs.append(np.argmax(target))
        
        accuracy_mean = np.mean(accuracy)
        
        if plots or plots_test:    
            plt.figure()
            plt.plot(pred_all,'*',label='Prediction')
            plt.plot(targs,'.',label='Target')
            plt.title(str(self.learn_type))
            plt.legend() 
        
        return accuracy_mean

