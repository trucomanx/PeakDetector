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

        model.add(  Conv1D( input_shape=(Nin, 1),
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
        
        model = Sequential()

        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=16, 
                            kernel_size=7, 
                            padding='same', 
                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                          )
                 );
        
        model.add(MaxPooling1D(pool_size=2))#128
        
        model.add(  Conv1D( filters=16, 
                            kernel_size=7, 
                            padding='same', 
                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                          )
                 );
        
        model.add(MaxPooling1D(pool_size=2))#64
        
        model.add(  Conv1D( filters=16, 
                            kernel_size=7, 
                            padding='same', 
                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                          )
                 );
                 
        
        model.add(MaxPooling1D(pool_size=2))#32
        
        model.add(  Conv1D( filters=16, 
                            kernel_size=7, 
                            padding='same', 
                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)#'relu'
                          )
                 );
        
        model.add(MaxPooling1D(pool_size=2))#16
        
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
        
