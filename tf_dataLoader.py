import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


def TF_loader(tfr_path,img_size,num_epochs=None):

    if not num_epochs:
        num_epochs = None

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(tfr_path),num_epochs)   
   
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)      
    features = tf.parse_single_example(serialized_example,
                                       features=
                                       {
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img' : tf.FixedLenFeature([], tf.string),
                                        }
                                       )

    image = tf.decode_raw(features['img'], tf.uint8)         
    image = tf.reshape(image, [img_size[0],img_size[1],img_size[2],-1])  
    image = tf.transpose(image,[3,0,1,2]) 
  
    label = tf.decode_raw(features['label'],tf.int32)
    label = tf.squeeze(label)

    return image,label
#def get_file_shape(image):
#   return int(image.shape[0])
def TF_loader_multi(root_path,img_size,num_epochs=None):
     
 
    list1=[]
    list2=[]
    filenames=os.listdir(root_path)
    print(filenames)
    for i,name in enumerate(filenames):
            path=os.path.join(root_path,name)
            print(path)
            img,label=TF_loader(path,img_size,num_epochs=None)
            list1.append(img)
            list2.append(label)
  
    return list1,list2

    
def shuffle_it(image,label,b=None):
   
    if b==None:
        b=int(image.shape[0])
    
    idx=np.arange(b)
    np.random.shuffle(idx)
    
   
    return image[idx],label[idx]


def data_loader(list1,list2):

    for i in range(len(list1)):
        img=list1[i]
        label=list2[i]
        if i==0:
            out_img=img
            out_label=label
        else:
            out_img=np.concatenate((out_img,img),0)
            out_label=np.concatenate((out_label,label),0)
   
    out_img,out_label=shuffle_it(out_img,out_label)
    out_img=out_img/255.0
    return out_img,out_label

 
def full_path_maker(dir_list,file_list):

    list_out=[]
    for i,dir in enumerate(dir_list):
        list_temp=[]
        names=file_list[i]
        for j,name in enumerate(names):
            list_temp.append(os.path.join(dir,name))

        list_out.append(list_temp)

    return list_out



if __name__=="__main__":

    with tf.Session() as sess:
        l1,l2=data_loader('H:/DLtest/C',[224,224,3],None)
        print(l1)
    #list1,list2=TF_loader_multi('H:/DLtest/C',[224,224,3],num_epochs=None)
    #sv = tf.train.Supervisor()  
    #with sv.managed_session() as sess: 
    #    list1,list2=sess.run([list1,list2])
            