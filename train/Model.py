#!/usr/bin/python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten

def create_model():
    Nin=Nout=256;
    
    model = Sequential()

    model.add(  Conv1D( input_shape=(Nin,1),
                        filters=8, 
                        kernel_size=5, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    #model.add(MaxPooling1D(pool_size=2))
    
    model.add(  Conv1D( filters=8, 
                        kernel_size=7, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    #model.add(MaxPooling1D(pool_size=2))
    
    model.add(  Conv1D( filters=8, 
                        kernel_size=9, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
             
    #model.add(MaxPooling1D(pool_size=2))
    
    model.add(  Conv1D( filters=4, 
                        kernel_size=11, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    #model.add(MaxPooling1D(pool_size=2))
    
    model.add(  Conv1D( filters=2, 
                        kernel_size=13, 
                        padding='same', 
                        activation='tanh'
                      )
             );
    
    #model.add(MaxPooling1D(pool_size=2))
    
    model.add(  Conv1D( filters=1, 
                        kernel_size=15, 
                        padding='same', 
                        activation='sigmoid'
                      )
             );
    
    model.add(Flatten())
    
    model.add(Dense(units=Nout, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    model.summary()
    
    return model 

def create_model2():
    Nin=Nout=256;
    
    model = Sequential(); 

    model.add(  Conv1D( input_shape=(Nout,1),
                        filters=16, 
                        kernel_size=7, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    model.add(MaxPooling1D(pool_size=2)); #128
    
    model.add(  Conv1D( filters=16, 
                        kernel_size=7, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    model.add(MaxPooling1D(pool_size=2)); #64
    
    model.add(  Conv1D( filters=16, 
                        kernel_size=7, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
             
    
    model.add(MaxPooling1D(pool_size=2)); #32
    
    model.add(  Conv1D( filters=16, 
                        kernel_size=7, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    model.add(MaxPooling1D(pool_size=2)); #16
    
    model.add(  Conv1D( filters=16, 
                        kernel_size=7, 
                        padding='same', 
                        activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                      )
             );
    
    model.add(Flatten())
    #model.add(Dense(units=int(Nin/8), activation='tanh'))
    model.add(Dense(units=Nout, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    model.summary()
    
    return model 


################################################################################

from tensorflow import Tensor
#from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
    
def residual_block( x:           Tensor, 
                    filters:     int = 16, 
                    kernel_size: int = 7) -> Tensor:
    
    y = Conv1D(kernel_size = kernel_size,
               strides     = 1,
               filters     = filters,
               padding     = "same")(x)
    
    y = LeakyReLU(alpha=0.1)(y)
    
    y = Conv1D(kernel_size = kernel_size,
               strides     = 1,
               filters     = filters,
               padding     = "same")(y)

    
    out = Add()([x, y])
    
    out = LeakyReLU(alpha=0.1)(out)
    
    out = BatchNormalization()(out)
    
    return out
    
def create_model_residual1():
    Nin=Nout=256;
    
    inputs = Input(shape=(Nin, 1))
    
    ########
    
    t = residual_block( inputs, filters = 32, kernel_size = 9);
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    #t = Conv1D(filters = 1, kernel_size = 9, padding= "same")(t)
    
    t = MaxPooling1D(pool_size=2)(t);
    
    ########
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    #t = Conv1D(filters = 1, kernel_size = 9, padding= "same")(t)
    
    t = MaxPooling1D(pool_size=2)(t);
    
    ########
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    #t = Conv1D(filters = 1, kernel_size = 9, padding= "same")(t)
    
    t = MaxPooling1D(pool_size=2)(t);
    
    ########
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    #t = Conv1D(filters = 1, kernel_size = 9, padding= "same")(t)
    
    t = MaxPooling1D(pool_size=2)(t);
    
    ########
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    t = residual_block( t     , filters = 32, kernel_size = 9);
    
    #t = Conv1D(filters = 1, kernel_size = 9, padding= "same")(t)
    
    t = Flatten()(t);
    
    outputs = Dense(Nout, activation='sigmoid')(t);
    
    model = Model(inputs, outputs);
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']);
    
    model.summary()
    
    return model 
