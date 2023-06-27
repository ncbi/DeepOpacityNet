from tensorflow.keras.optimizers import *

from tensorflow.keras.layers import *

from tensorflow.keras.models import *

# from keras.utils import multi_gpu_model

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from tensorflow.keras.applications import *

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import *

from tensorflow.keras.regularizers import *

from tensorflow.keras.constraints import *

import tensorflow as tf

from losses import *

import numpy as np

from tensorflow.keras.layers import Input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D




def xception_transfer_learning(img_height, img_width, nClasses=2): ####
   
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='xception_transfer_learning')
    
    return model

def resnet_transfer_learning(img_height, img_width, nClasses=2): #####
   
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)    
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)
    
    x = Dropout(0.5)(x)
        
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='resnet_transfer_learning')
    
    return model




def inception_resnet_v2_transfer_learning(img_height, img_width, nClasses=2): ######
   
    base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    
    x = GlobalAveragePooling2D()(x)  
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='inception_resnet_v2')
    
    return model


def resnet152v2_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = ResNet152V2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)
        
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)
    
    x = Dropout(0.5)(x)
        
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='resnet152v2_transfer_learning')
    
    return model


def inception_v3_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)
        
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)
    
    x = Dropout(0.5)(x)
        
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='incpv3_transfer_learning')
    
    return model



def dense201_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)
        
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)
    
    x = Dropout(0.5)(x)
        
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='dense201_transfer_learning')
    
    return model



def vgg16_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='vgg16_transfer_learning') # base_model.inputs
    
    return model
