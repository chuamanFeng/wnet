import time
import tensorflow as tf
import numpy as np
import PIL.Image as IMG
import cv2
#foceus on scope rebuild technic
from tensorflow.python.platform import gfile

MODEL_DIR="H:/Wtest/expert-graph.pb"

class external():
    def __init__(self,pb_file_path=MODEL_DIR):
         
         self.graph = tf.Graph()
         with self.graph.as_default():
            output_graph_def = tf.GraphDef()
            
            with gfile.FastGFile(pb_file_path, "rb") as f:
                output_graph_def.ParseFromString(f.read(),tf.float32)
              
         tf.import_graph_def(output_graph_def)  
         self.sess=tf.Session()     

    def external_convert(self,img_string,img_size):     
         pimg = IMG.fromstring("input",img_size,img_string)  # pimg is a PIL image 
         array = np.array(pimg)    

         return array

    def convert_inputs(self,img_mat,stand_size):
        img_size=cv2.GetSize(img_mat)
        img_string=img_mat.tostring()
        img=self.external_convert(img_string,img_size)

        img=tf.image.resize_bilinear(np.expand_dims(img,0),stand_size)
        img = img/255.0

        return img      

    def push_final(self,img_mat,stand_size):
        img = np.array([])
        for img_m in img_mat:
            temp=self.convert_inputs(img_mat,stand_size)
            img = np.concatenate((img, temp))

        img = np.reshape(img,(-1,stand_size[0],stand_size[1],1))  
        return img

    def model_sumon(self,mat_serial,stand_size):

       imgs = self.push_final(mat_serial,stand_size)
       out = []
       with tf.Graph().as_default():
           for img in imgs:
               feed_dict = {img_pl: img,is_training_pl: is_training,}                  
               pred_lb= self.sess.run(pred_label, feed_dict=feed_dict)
               out.append(pred_lb)

       return out

if __name__ == '__main__':  

         with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            
            with gfile.FastGFile("H:/Wtest/test_115/frozen_model.pb", "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def)



