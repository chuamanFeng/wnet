import tensorflow as tf
import numpy as np
import basic_tf
import sys
import os
import time

from basic_moduel import skip_sbstract_layer,input_down_size_unit,output_down_size_unit

def placeholder_inputs(b,h,w,c):
    img_pl = tf.placeholder(tf.float32, shape=(b,h,w,c),name="img_place")
    label_pl = tf.placeholder(tf.int32, shape=(b),name="label_place")
    
    return  img_pl, label_pl

def wnet_model(img,is_training,class_num,bn_decay=None):        

    with tf.variable_scope('input_layer'):

        out0=input_down_size_unit(img,64,192,is_training,bn_decay,'input_pre')
        skip0=out0

    with tf.variable_scope('intermidate_layer'):

        out1 ,skip1= skip_sbstract_layer(skip0,out0,[16,16,32],[[5,5],[3,3],[1,1]],is_training,bn_decay=bn_decay,scope='inception_1')      
        out2 ,skip2= skip_sbstract_layer(skip1,out1,[32,32,64],[[5,5],[3,3],[1,1]],is_training,bn_decay=bn_decay,scope='inception_2')       
        out3 ,skip3= skip_sbstract_layer(skip2,out2,[64,64,96],[[5,5],[3,3],[1,1]],is_training,bn_decay=bn_decay,scope='inception_3',maxpool=False) 
        
        out = output_down_size_unit(tf.concat((out3,skip3),-1),256,is_training,bn_decay,'out_downsize')
      
    with tf.variable_scope('output_layer'):        

        out = basic_tf.fully_connected(out,1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        out = basic_tf.dropout(out, keep_prob=0.5, is_training=is_training, scope='dp1')
        out = basic_tf.fully_connected(out,128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        out = basic_tf.dropout(out, keep_prob=0.5, is_training=is_training, scope='dp2')       
        out = basic_tf.fully_connected(out,class_num, bn=True, is_training=is_training, scope='fc3', activation_fn=None,bn_decay=bn_decay)  
        out_pred = tf.argmax(out,1,name="final")
   
    return out,out_pred

def get_loss(pred_out,label):
   
    with tf.variable_scope('loss_calc'):
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_out,labels=label)
        loss = tf.maximum(loss,1e-10)
        loss=tf.reduce_mean(loss,0)  

    return loss

#def inference_model(img_list,stand_size):

#    batch_size =img_list.len()
#    pred = np.argmax()

#if __name__=='__main__':
# with tf.Session() as sess:
#    img=np.random.rand(32,224,224,3).astype('float32')
#    img=tf.constant(img)
#    out=wnet_model(img,tf.constant(True),5)
#    print(out.shape)
