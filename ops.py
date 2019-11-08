import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
import numpy as np
import math

epsilon = 1e-5
def conv(numout,kernel_size=3,strides=1,kernel_regularizer=0.0005,padding='same',use_bias=False,name='conv'):
    return tf.keras.layers.Conv2D(name=name,filters=numout, kernel_size=kernel_size,strides=strides, padding=padding,use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),kernel_initializer=tf.random_normal_initializer(stddev=0.1))

	
def convt(numout,kernel_size=3,strides=1,kernel_regularizer=0.0005,padding='same',use_bias=False,name='conv'):
    return tf.keras.layers.Conv2DTranspose(name=name,filters=numout, kernel_size=kernel_size,strides=strides, padding=padding,use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),kernel_initializer=tf.random_normal_initializer(stddev=0.1))	
def bn(name,momentum=0.9):
    return tf.keras.layers.BatchNormalization(name=name,momentum=momentum)


class c7s1_k(keras.Model):
    def __init__(self,scope: str="c7s1_k",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(c7s1_k, self).__init__(name=scope)
        self.conv1 = conv(numout=k,kernel_size=7,kernel_regularizer=reg,padding='valid',name='conv')
        self.norm =norm
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False,activation='Relu'):
        x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        if activation is 'Relu':
            x = tf.nn.relu(x)
        else:
            x = tf.nn.tanh(x)
        return x  

class dk(keras.Model):
    def __init__(self,scope: str="dk",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(dk, self).__init__(name=scope)
        self.norm =norm
        self.conv1 = conv(numout=k,kernel_size=3,strides=[2, 2],kernel_regularizer=reg,padding='same',name='conv')
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False):
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        return x 

		
class Rk(keras.Model):
    def __init__(self,scope: str="Rk",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(Rk, self).__init__(name=scope)
        self.norm =norm
        self.conv1 = conv(numout=k,kernel_size=3,kernel_regularizer=reg,padding='valid',name='layer1/conv')
        if norm is 'instance':
            self.scale1 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer1/scale')
            self.offset1 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer1/offset')
            self.scale2 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer2/scale')
            self.offset2 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer2/offset')
        elif norm is 'bn':
            self.bn1 = bn(name='layer1/bn')
            self.bn2 = bn(name='layer2/bn')
        self.conv2 = conv(numout=k,kernel_size=3,kernel_regularizer=reg,padding='valid',name='layer2/conv')

    def call(self,x,training=False):
        inputs = x
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale1 * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset1
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        x = self.conv2(x)		
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale2 * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset2
        elif self.norm is 'bn':
            x = self.bn2(x,training=training)		
        return x + inputs 		
		
		
	
		
class n_res_blocks(keras.Model):
    def __init__(self,scope: str="n_res_blocks",n:int =6,k:int=16,reg:float=0.0005,norm:str="instance"):
        super(n_res_blocks, self).__init__(name=scope)
        self.group=[]
        self.norm =norm
        for i in range(n):
            self.group.append(Rk(scope='Rk_'+str(i+1),k=k,reg=reg,norm=norm))
    def call(self,x,training=False):
        for i in range(len(self.group)):
            x = self.group[i](x,training=training)
        return x 
		
class uk(keras.Model):
    def __init__(self,scope: str="uk",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(uk, self).__init__(name=scope)
        self.norm =norm
        #self.conv1 = conv(numout=k,kernel_size=3,kernel_regularizer=reg,padding='valid',name='conv')
        self.conv1 = convt(numout=k,kernel_size=3,strides=[ 2 , 2 ],kernel_regularizer=reg,padding='same',name='conv')
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False):
        #height = x.shape[1]
        #width = x.shape[2]
        #x=tf.compat.v1.image.resize_images(x, [2*height,2*width],method = 0, align_corners = True)
        #x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        return x 	        
		
class Ck(keras.Model):
    def __init__(self,scope: str="uk",k:int =16,stride:int=2,reg:float=0.0005,norm:str="instance"):
        super(Ck, self).__init__(name=scope)
        self.norm =norm
        self.conv1 = conv(numout=k,kernel_size=3,strides=[ stride, stride],kernel_regularizer=reg,padding='same',name='conv')
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False,slope=0.2):
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.leaky_relu(x,slope)
        return x 

class last_conv(keras.Model):
    def __init__(self,scope: str="last_conv",reg:float=0.0005):
        super(last_conv, self).__init__(name=scope)
        self.conv1 = conv(numout=1,kernel_size=4,kernel_regularizer=reg,padding='same',name='conv')
    def call(self,x,use_sigmoid=False):
        x = self.conv1(x)
        if use_sigmoid:
            output = tf.nn.sigmoid(x)
        return x 