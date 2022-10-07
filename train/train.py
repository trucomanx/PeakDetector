#!/usr/bin/python

import tensorflow as tf
import os


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten

def create_model():
        Nin=Nout=256;
        
        model = Sequential()

        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=32, 
                            kernel_size=7, 
                            padding='same', 
                            activation='relu'
                          )
                 );
        
        #model.add(MaxPooling1D(pool_size=2))
        
        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=16, 
                            kernel_size=15, 
                            padding='same', 
                            activation='relu'
                          )
                 );
        
        #model.add(MaxPooling1D(pool_size=2))
        
        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=8, 
                            kernel_size=15, 
                            padding='same', 
                            activation='tanh'
                          )
                 );
                 
        #model.add(MaxPooling1D(pool_size=2))
        
        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=4, 
                            kernel_size=15, 
                            padding='same', 
                            activation='relu'
                          )
                 );
        
        #model.add(MaxPooling1D(pool_size=2))
        
        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=2, 
                            kernel_size=15, 
                            padding='same', 
                            activation='tanh'
                          )
                 );
        
        #model.add(MaxPooling1D(pool_size=2))
        
        model.add(  Conv1D( input_shape=(Nin, 1),
                            filters=1, 
                            kernel_size=15, 
                            padding='same', 
                            activation='sigmoid'
                          )
                 );
        
        model.add(Flatten())
        
        #model.add(Dense(units=Nout, activation='sigmoid'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        
        print(model.summary())
        
        return model 
        
model=create_model();
output_dir='output'
EPOCAS=100;
BATCH_SIZE=32;

try: 
    os.mkdir(output_dir)
except: 
    pass

from CustomDataGenerator import CustomDataGenerator

training_generator   = CustomDataGenerator(100000,batch_size=BATCH_SIZE);
validation_generator = CustomDataGenerator( 20000,batch_size=BATCH_SIZE,validation=True);

best_model_file=os.path.join(output_dir,'model.h5');


checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                verbose=1);

history_cb = tf.keras.callbacks.CSVLogger('./log.csv', separator=",", append=False)

history = model.fit(training_generator,
                    epochs=EPOCAS,
                    validation_data=validation_generator,
                    callbacks=[checkpoint,history_cb],
                    verbose=1
                   );

'''
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
'''
