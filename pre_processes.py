import numpy as np
import sys
import os
import tensorflow as tf
import PIL.Image as IM


#-------------------------------------------------------------global pack
def get_all_done(main_path,TF_savepath,file_size,img_size,raw_size,brt_hue_round=1,crop_round=1,resize=True,new_size=[112,112]):
   
    file_folders_class=getDataFiles(main_path)  

    total_num=0
      
    for j,folder in enumerate(file_folders_class):
              
        file_num=0
        writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')   

        img,img_num=input_prepare_pack(os.path.join(main_path,folder),img_size,raw_size,brt_hue_round,crop_round,resize,new_size)
        label=label_pre_pack(folder,img_num)
        print(label)
        #label=label_pre_pack(folder)    
        print(label.shape,img.shape)
        total_num+=img_num
        #curr_filesize+=img_num

        if  img_num>file_size:           
            sidx=file_size
            TFRecord_writer(writer,img[0:sidx],label[0:sidx])
            writer.close()
            file_num+=1            
            img_num-=file_size

            while img_num//file_size>0:
                writer = tf.python_io.TFRecordWriter(TF_savepath+str(j)+str(file_num)+'.tfrecords')  
                TFRecord_writer(writer,img[sidx:sidx+file_size],label[sidx:sidx+file_size])
                writer.close()

                sidx+=file_size
                img_num-=file_size
                file_num+=1
               
            writer = tf.python_io.TFRecordWriter(TF_savepath+str(file_num)+'.tfrecords')  
            TFRecord_writer(writer,img[sidx:],label[sidx:])
         
        
        else:                
            TFRecord_writer(writer,img,label)
        writer.close()    
        print(j+1,'folder(s) finished')

    print(total_num)
    writer.close()
    

       
      

  
#-------------------------------------------------------------local folder pack

def input_prepare_pack(filepath,img_size,raw_size,brt_hue_round,crop_round,resize=True,new_size=[112,112]):  #for one class

    filenames= getDataFiles(filepath)    #single class_folder
    img_rawlist=image_to_array(filenames,filepath,img_size)
    img=flip_it_randomly(img_rawlist)
    img = change_img_dim(img)
   
    if brt_hue_round>0:
        img = change_brightness_hue_randomly(img,brt_hue_round)
  
    img = change_contrast(img)
  
    img = noisy_img_randomly(img)
   
    #img = flip_it_randomly(img)
  
    img =return_to_or_shape(img,raw_size)
 
    if crop_round>0:
        img = crop_randomly(img,raw_size,crop_round)
 
    if resize:
        img = resize_it(img,new_size)
 
    img_num=img.get_shape()[0].value
   
    return img,img_num

def label_pre_pack(folder,img_num):

    label=folder.split('.')
    label=np.tile(label[0],[img_num])

    return label.astype('int32')

#------------------------------------------------------------File works
def getDataFiles(TEST_or_Train_TOPfloder):

    return os.listdir(TEST_or_Train_TOPfloder)

def image_to_array(filenames,filepath,img_size): #ok

    name_num=int(len(filenames))   

    h=img_size[0][0]
    w=img_size[0][1]
    area=img_size[1]
   
    img=[]
    print('transforming...')
    for i, name in enumerate(filenames):
        image = IM.open(os.path.join(filepath,name))
        r=image.split()[0];
        r_arr = np.array(r).reshape(area)
        r_arr=np.reshape(r_arr,(w,h,1)).astype('float32')
        
       
        img.append(r_arr)
 
 
    print('transformed',len(img)) 
    '''                                             #this part have permission error
    os.chdir(savepath) 
    with open(savepath, mode='wb') as f:
        pk.dump(result, f)
    '''        
    return img

def TFRecord_writer(writer,img,label):

     img=tf.cast(img,tf.uint8)
     label=tf.cast(label,tf.int32)
     #writer = tf.python_io.TFRecordWriter(filepath_save) 
     img=(img.eval()).tobytes()     
     label=(label.eval()).tobytes()
     example=tf.train.Example(features=
                              tf.train.Features(feature=
                                                {"img":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])), 
                                                 "label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
                                                 }
                                                )  
                              )
   
     writer.write(record=example.SerializeToString())

#-----------------------------------------------------------------------Image works
def change_img_dim(img):

    b=int(img.shape[0])
    c=int(img.shape[-1])
    img = tf.reshape(img,(b,-1,c))
    return img


def change_brightness_hue_randomly(img,change_round,clip=None):
  
    if clip==None:
        clip=np.random.randint(low=0,high=50)/100
    brt=np.clip(np.random.rand(change_round),-1.*clip,clip).astype('float32')
    #hue=np.clip(np.random.rand(change_round),-1.*clip,clip).astype('float32')

    
    for i in range(change_round):
        temp=tf.image.adjust_brightness(img,brt[i])
        #temp=tf.image.adjust_hue(img,hue[i])
        if i==0:
            out=temp
        else:
            out = tf.concat((out,temp),0)
      
  
    return out

def change_contrast(img,val=None):
   
    if val==None:
        val=np.random.randint(low=0,high=30)/100   
           
    out=tf.image.adjust_contrast(img,val)
    
    return tf.concat((out,img),0)

def flip_it_randomly(img_list):

    for i,img in enumerate(img_list):   
        sp=np.random.randint(low=0,high=2)  
        if sp == 0:
            img= tf.image.random_flip_left_right(img)
        elif sp == 1:
            img = tf.image.random_flip_up_down(img)
        elif sp==2:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
        img=tf.expand_dims(img,0)
      
        if i==0:
            out=img;
        elif i>0:
            out=tf.concat([out,img],0)
    print(out.shape)
    return out

def noisy_img_randomly(img,c=3,weight=2):   #ok
          
    noi=2*np.random.rand(1).astype('float32')
   
    noisy_image=img+noi
    if c==3:
        noisy_image = tf.clip_by_value(noisy_image, 0.0,255.0)
    else:
        noisy_image = tf.clip_by_value(noisy_image,0.0, 1.0)
    
    return  tf.concat((noisy_image,img),0)

def return_to_or_shape(img,img_size,c=None):
           
    b=int(img.shape[0])
    if c==None:
        c=int(img.shape[-1])

    return  tf.reshape(tf.expand_dims(img,1),(b,img_size[0],img_size[1],c))

def crop_randomly(img,imgsize=None,crop_round=6):

    if imgsize==None:
        h=img.shape[1]
        w=img.shape[2]
    else:
        h=imgsize[0]
        w=imgsize[1]

    short_edge=np.min([h,w],0)
    x0=(w-short_edge)//2
    y0=(h-short_edge)//2
    ct_img=img[:,y0:y0+short_edge,x0:x0+short_edge,:]

    crop_side=int(short_edge*0.8)
    rangex=np.random.randint(low=0,high=short_edge-crop_side,size=[crop_round])
    rangey=np.random.randint(low=0,high=short_edge-crop_side,size=[crop_round])

    for i in range(crop_round):      

        temp=ct_img[:,rangex[i]:rangex[i]+crop_side,rangey[i]:rangey[i]+crop_side,:]        
        if i==0:
            out=temp
        else:
            out=tf.concat((out,temp),0)
   
    return tf.reshape(out,(int(out.shape[0]),crop_side,crop_side,int(out.shape[-1])))

def resize_it(img,new_size):

     return tf.image.resize_images(img,new_size)

def newname(filepath,new_name): 
  
  os.chdir(filepath)
  filenames=os.listdir(filepath)
  for i,filename in enumerate(filenames):      
      portion = os.path.splitext(filename)
     
      newname = new_name+str(i)+portion[1]
          
      os.rename(filename,newname) 

if __name__=='__main__':

    #writer = tf.python_io.TFRecordWriter('H:/Wtest/saa') 
    with tf.Session() as sess:
      
        get_all_done('H:/OriginalPics/plus1','H:/tecplus11',1024,[[68,68],68**2],[68,68],brt_hue_round=1,crop_round=0,resize=True,new_size=[56,56])
        #get_all_done('H:/TRUEDATA/B/c/1','H:/c13',1024,[[1080,1920],1080*1920],[375,375],2,1)
     
        #img = input_prepare_pack('H:/Wtest/A/train1/0',[[1024,1024],1024**2],0,0)
        #img=img.eval()
        #img_byte=img.tostring()

        #example=tf.train.Example(
        #features=tf.train.Features
        #    feature={
        #        "image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_byte])),                
        #        })  )
   
        #writer.write(record=example.SerializeToString())
        #writer.close()