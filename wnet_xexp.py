import argparse
import math
import numpy as np
import tensorflow as tf

import socket
import importlib
import os
import sys
import time
import basic_tf
import pickle as pk
import WNet_model
import tf_dataLoader as tl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='WNet_model', help='Model name')
parser.add_argument('--stan_size', default=[56,56], help='lenth&width for croping&resizing')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='H:\Wtest\Best', help='Log dir [default: log]')
#---------------------------------------------------------------------
FLAGS = parser.parse_args()

#------------------------------------get those arguments from the parsel---------------------------------------------------#
BATCH_SIZE = FLAGS.batch_size
STAN_SIZE = FLAGS.stan_size



MODEL = importlib.import_module(FLAGS.model) # import network module
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)

    '''
    some sort of copy work
os.system('COPY %s %s' % ( FLAGS.model+'.py', LOG_DIR)) # bkp of model def
os.system('COPY color_net_train.py %s' % (LOG_DIR)) # bkp of train procedure
    '''
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#-------------------------------------other params,including hyper params & data address----------------------------------------------------#



IMG_FORM='bmp'
CHANNEL=1
CLASS_NUM=2



HOSTNAME = socket.gethostname()


#-----------------------------------------basic helpers for retrieving learnable params------------------------------------------------------#
#string helpler
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def train(): 
    '''
 elements:
   A part
    placeholder for inputs
    batch&batch_normalization decay
    model&its loss/accuracy
    optimizer(train_op)
    saver
   B part
    session
    summary writer(train&test)
    global variable initializer(use that sess run it before training)
    operations in a dictionary
   C part
    a loop for training (train&eval&summary)
    '''
    with tf.Graph().as_default():
     
        with tf.variable_scope('input_placeholder'):
           
             img_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, STAN_SIZE[0], STAN_SIZE[1], CHANNEL)
             is_training_pl = tf.placeholder(tf.bool, shape=())
             print(is_training_pl)
          
        _, pred_label = MODEL.wnet_model(img_pl, is_training_pl, CLASS_NUM)
       
        saver = tf.train.Saver()

       
        with tf.variable_scope('training_session'):
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True #allow grouing gpu usage
            config.allow_soft_placement = True   #allow tf  charge the devices when our charge failed
            config.log_device_placement = False  #print device configuration infos
            sess = tf.Session(config=config)
        
           
            init = (tf.global_variables_initializer())
            sess.run(init, {is_training_pl: False})  # kind of slow but passible
            # -------------------------------------------------------------------------
           
            initial_step=0
            ckpt=tf.train.get_checkpoint_state(LOG_DIR)     
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
               
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model-export"))
                return

if __name__=="__main__":
       
    train()
    LOG_FOUT.close()
