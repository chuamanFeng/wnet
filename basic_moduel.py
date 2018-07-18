import tensorflow as tf
import numpy as np
import basic_tf
import sys
import os
import time


def input_down_size_unit(img,c_out1,cout_2,is_training,bn_decay,scope,bn=True):

    
    img1=basic_tf.conv2d(img,c_out1, [3,3],
                            padding='SAME', stride=[1,1], bn=True, is_training=is_training,
                            scope='conv_input0', bn_decay=bn_decay)       #down_size

    img1=basic_tf.max_pool2d(img1,[3,3],scope='max_pool_input',stride=[2,2],padding='SAME')

    img1=basic_tf.conv2d(img1,cout_2, [3,3],
                            padding='SAME', stride=[1,1], bn=True, is_training=is_training,
                            scope='conv_input1', bn_decay=bn_decay)       #down_size

    img1=basic_tf.max_pool2d(img1,[3,3],scope='max_pool_input',stride=[2,2],padding='SAME')

    return img1
def output_down_size_unit(input,cout,is_training,bn_decay,scope,bn=True):

    b=input.get_shape()[0].value    
    out =basic_tf.avg_pool2d(input ,[3,3],scope='ave_pool_output',stride=[1,1],padding='SAME') 
        
    out= basic_tf.conv2d(out,cout, [1,1],
                            padding='SAME', stride=[1,1], bn=True, is_training=is_training,
                            scope='conv_output1', bn_decay=bn_decay)       #down_size
    out=basic_tf.max_pool2d(out,[3,3],scope='max_pool_output',stride=[2,2],padding='SAME')  
      
    out = tf.reshape(out,(b,-1))#(-1,2048)
    return out

def wnet_model_unit(input,mlp_list,kernel_size_list,is_training,bn_decay,scope,bn=True,maxpool=True):
     
    bf= input._shape[-1]        
    with tf.variable_scope(scope) as sc:   
        output=basic_tf.max_pool2d(input,[3,3],'maxpool_input',[1,1],'SAME')           
        output=basic_tf.conv2d(output,bf//3, [1,1],
                                padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                scope='conv_input', bn_decay=bn_decay)       #filter(output1)
        for i ,ks in enumerate(kernel_size_list):
            kernel_h=ks[0]
            kernel_w=ks[1]        
            out=input            #conv from down_sized input
            c_out =mlp_list[i]            
            if kernel_h==1 and kernel_w==1:
                out = basic_tf.conv2d(out, c_out, [1,1],
                                    padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                    scope='conv_p1%d'%(i), bn_decay=bn_decay)    #conv from down_sized input
            else:
                out = basic_tf.conv2d(out, c_out//2, [1,1],
                                    padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                    scope='conv_p0%d'%(i), bn_decay=bn_decay)    #conv from down_sized input
                out = basic_tf.conv2d(out, c_out, [kernel_h,kernel_w],
                                        padding='SAME', stride=[1,1], bn=bn, is_training=is_training,
                                        scope='conv_p1%d'%(i), bn_decay=bn_decay)    #conv from down_sized input              
            #print(output.shape)     
            output=tf.concat([output,out],-1)
           
      
        if maxpool:
            output=basic_tf.max_pool2d(output,[3,3],'max_pool',[2,2],'SAME')
      
            #(b,h2,w2,sum[mlp])
    print(output.shape,'one turn finished')
   
    return output     

def skip_sbstract_layer(skip,input,mlp_list,kernel_size_list,is_training,bn_decay,scope,bn=True,maxpool=True):

   with tf.variable_scope(scope) as sc: 
       input=tf.concat((skip,input),-1)
       input=wnet_model_unit(input,mlp_list,kernel_size_list,is_training,bn_decay,'abstraction',bn,maxpool)
       if maxpool:
           skip=basic_tf.max_pool2d(skip,[3,3],'skip',[2,2],'SAME')

   return input,skip