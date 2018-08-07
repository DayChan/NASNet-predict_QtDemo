import numpy as np
import random
import os
import keras
import tensorflow as tf
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model ,np_utils
from keras.layers import GlobalAveragePooling2D, Dense,Dropout
from keras.optimizers import Adam, SGD, Adadelta ,Nadam
from keras.models import Model
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input
from keras import backend as K



#DATA_DIR='./DATA/test'
NAS_DIR='./NASnet_data'
OUTPUT_DIR = 'predict_X_2.csv'
WIDTH=299
HEIGHT=299


NASnet=keras.applications.nasnet.NASNetMobile(input_shape=(WIDTH,HEIGHT,1), include_top=False, weights=None, pooling='avg')
x=NASnet.output
x=Dropout(0.3)(x)
x=Dense(2,activation='sigmoid')(x)
model=Model(inputs=NASnet.input,outputs=x)
model.load_weights(NAS_DIR+'/weights/NASnet_ver5.hdf5')

def getDatalist(filename):
    Datalist=[]
    img = Image.open(filename)  
    img = img.resize((WIDTH,HEIGHT),Image.ANTIALIAS) 
    img = img.convert('L')
    x = image.img_to_array(img)
    #print(x.shape)
    x=(x-127)/128
    y=[x,0]
    Datalist.append(x)
    return Datalist

def get_result(filename):
    
    Data_list =getDatalist(filename)
    
    data_x=np.zeros([len(Data_list),WIDTH,HEIGHT,1],np.float32)

    i=0
    for l in Data_list:
        data_x[i]=l
        i+=1

    pre = model.predict(data_x)

    print("pre:",pre)

    return pre[0][0]