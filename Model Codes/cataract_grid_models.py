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




def ConvLayer(conv, nfilters=16, sz=3):
        
    conv = Conv2D(nfilters, (sz, sz), padding='same', kernel_initializer = 'he_normal')(conv)
    
    conv = BatchNormalization()(conv)
    
    conv = Activation('relu')(conv)
    
    return conv



def DownPool(conv):
            
    down = MaxPooling2D(pool_size=(2, 2))(conv)
    
    return down



def InitConvLayer(conv, nfilters=16, sz=3, activation='relu'):
        
    conv_blk1 = Conv2D(nfilters, (sz, sz), strides = (2, 2), padding='same', kernel_initializer = 'he_normal')(conv)
    
    conv_blk1_bn = BatchNormalization()(conv_blk1)
    
    conv_blk1_act = Activation(activation)(conv_blk1_bn)
    
    
    conv_blk2 = Conv2D(nfilters, (sz, sz), padding='same', kernel_initializer = 'he_normal')(conv_blk1_act)
    
    conv_blk2_bn = BatchNormalization()(conv_blk2)
    
    conv_blk2_act = Activation(activation)(conv_blk2_bn)
    
    
    return conv_blk2_act



def SepConvLayer(conv, nfilters=16, sz=3, activation='relu'):
                
    sepconv_blk1 = SeparableConv2D(nfilters, (sz, sz), padding='same', kernel_initializer = 'he_normal')(conv)
    
    sepconv_blk1_bn = BatchNormalization()(sepconv_blk1)
    
    sepconv_blk1_act = Activation(activation)(sepconv_blk1_bn)
    
    
    sepconv_blk2 = SeparableConv2D(nfilters, (sz, sz), padding='same', kernel_initializer = 'he_normal')(sepconv_blk1_act)
    
    sepconv_blk2_bn = BatchNormalization()(sepconv_blk2)
    
            
    # adding attention
    
#     sepconv_blk2_bn = CBAM(sepconv_blk2_bn)

#     sepconv_blk2_bn = autoencoder(sepconv_blk2_bn, pool_type='mean', pool_size=4, activation='relu')
    
#     sepconv_blk2_bn = BAM(sepconv_blk2_bn)    
    
    return sepconv_blk2_bn



def MaxPool(conv):
                
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
            
    return pool

def PointConvPool(conv, nfilters=16):
    
    pool = AveragePooling2D(pool_size=(2, 2))(conv)
    
#     pool = concatenate([pool, pool], axis=-1)

#     pool = MaxPooling2D(pool_size=(2, 2))(conv)
            
    pool = Conv2D(nfilters, (1, 1), strides=(1, 1), padding='same', kernel_initializer = 'he_normal')(pool)
    
    pool = BatchNormalization()(pool)
    
    return pool


def PointConv(conv, nfilters=16):
                
    conv = Conv2D(nfilters, (1, 1), strides=(1, 1), padding='same', kernel_initializer = 'he_normal')(conv)
    
    conv = BatchNormalization()(conv)
    
    return conv



def ConvPool(conv, nfilters=16, sz=3, activation='relu'):
        
    pool = Conv2D(nfilters, (sz, sz), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv)
    
    return pool

def init_conv(conv, nfilters=16, sz=3, activation='relu', doubled=False):
    
    n1 = nfilters
    
    n2 = 2*nfilters if doubled else nfilters

    x = Conv2D(n1, (sz, sz), strides = (2, 2), padding='same', kernel_initializer = 'he_normal')(conv)
    
    x = BatchNormalization()(x)
    
    x = Activation(activation)(x)
    
    
    x = Conv2D(n2, (sz, sz), padding='same', kernel_initializer = 'he_normal')(x)
    
    x = BatchNormalization()(x)
    
    x = Activation(activation)(x)
    
    return x



def squeeze_excite_block(conv, ratio=16):
    
    nfilters = conv.shape[-1]            
    
    se_shape = (1, 1, nfilters)

    se = GlobalAveragePooling2D()(conv)
    
    se = Reshape(se_shape)(se)
    
    se = Dense(nfilters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    
    se = Dense(nfilters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    x = multiply([conv, se])
    
    return x

def sep_conv_m_layer(conv, nfilters=16, sz=3, activation='relu', m_layers=2, use_se=False):

    x = conv

    for i in range(m_layers):

        x = SeparableConv2D(nfilters, (sz, sz), padding='same', kernel_initializer = 'he_normal')(x)
        
        x = BatchNormalization()(x)
        
        if i < (m_layers-1): 
           
           x = Activation(activation)(x)

    conv = PointConvPool(conv, nfilters=nfilters)
    
    x = MaxPool(x)
    
    if use_se:
        
        x = squeeze_excite_block(x)
        
    x = add([conv, x])
    
    x = Activation(activation)(x)
                    
    return x


def gelu(x):
    
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def XBnLm(img_width, img_height, nClasses=2, nChannels=3, nfilters=32, activation='relu', n_blocks=5, m_layers=2,
          doubled=False, use_se=False):
    
    # input
    
    inputs = Input((img_width, img_height, nChannels))
        
    if doubled:
        
        filters_list = [nfilters * 4 * (2**i) for i in range(n_blocks)]
    
    else:
        
        filters_list = [nfilters * 2 * (2**i) for i in range(n_blocks)]
    
    
    if activation=='gelu':
        
        activation = gelu
    
    # encoder
    
    conv = init_conv(inputs, nfilters=nfilters, sz=3, activation=activation, doubled=doubled)
    
    x = conv

    for idx in range(n_blocks):

        x = sep_conv_m_layer(x, nfilters=filters_list[idx], sz=3, activation=activation, m_layers=m_layers, use_se=use_se)

    
    x = GlobalAveragePooling2D()(x)

#     x = Dropout(0.5)(x)
    
    x = Dense(256, activation=activation)(x)
    
#     x = Dropout(0.5)(x)
    
    predictions = Dense(nClasses, activation='softmax', name='prediction')(x)

    model = Model(inputs=inputs, outputs=predictions)
    
    return model





def xception_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
#     x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)

    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='xception_transfer_learning')
    
    return model

def resnet_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
#     x = GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)    
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)
    
    x = Dropout(0.5)(x)
        
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='resnet_transfer_learning')
    
    return model


def inception_resnet_v2_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
#     x = base_model.output
    
#     x = GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)  
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='inception_resnet_v2') # base_model.inputs
    
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


def incpv3_transfer_learning(img_height, img_width, nClasses=2):
   
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



def incpresv2_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)
        
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)
    
    x = Dropout(0.5)(x)
        
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='incpresv2_transfer_learning')
    
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
    
#     x = base_model.output
    
#     x = GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='inception_resnet_v2') # base_model.inputs
    
    return model


def efficientnet_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
#     x = base_model.output
    
    x = GlobalAveragePooling2D()(x)    
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='inception_resnet_v2') # base_model.inputs
    
    return model



def efficientv2_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = EfficientNetV2L(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
#     x = base_model.output
    
    x = GlobalAveragePooling2D()(x)    
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='inception_resnet_v2') # base_model.inputs
    
    return model


def convxlarge_transfer_learning(img_height, img_width, nClasses=2):
   
    base_model = ConvNeXtXLarge(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
            
    base_model.trainable = False
    
    inputs = Input(shape=(img_height, img_width, 3))
                
    x = base_model(inputs, training=False)
    
#     x = base_model.output
    
    x = GlobalAveragePooling2D()(x)    
    
    x = Dropout(0.5)(x)
        
    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    
    outputs = Dense(nClasses, activation='softmax', name='prediction')(x)
            
    model = Model(inputs=inputs, outputs=outputs, name='inception_resnet_v2') # base_model.inputs
    
    return model
