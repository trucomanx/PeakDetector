#!/usr/bin/python

import tensorflow as tf
import os
from Model import create_model

        

output_dir='output'
EPOCAS=100;
BATCH_SIZE=32;

try: 
    os.mkdir(output_dir)
except: 
    pass
    
    
model=create_model();

from CustomDataGenerator import CustomDataGenerator
training_generator   = CustomDataGenerator(100000,batch_size=BATCH_SIZE);
validation_generator = CustomDataGenerator( 20000,batch_size=BATCH_SIZE,validation=True);

best_model_file=os.path.join(output_dir,'model.h5');


checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                verbose=1);

log_file=os.path.join(output_dir,'log.csv');
history_cb = tf.keras.callbacks.CSVLogger(log_file, separator=",", append=False)

history = model.fit(training_generator,
                    epochs=EPOCAS,
                    validation_data=validation_generator,
                    callbacks=[checkpoint,history_cb],
                    verbose=1
                   );

