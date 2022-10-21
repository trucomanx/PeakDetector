#!/usr/bin/python

import tensorflow as tf
import os

import sys
sys.path.append('../src')

from PeakDetector.Model import create_model
from PeakDetector.Model import create_model2
from PeakDetector.Model import create_model_residual1

"""# Global variables"""

output_dir='output'
EPOCAS=200;
BATCH_SIZE=64;
NPEAK=5;

try: 
    os.mkdir(output_dir)
except: 
    pass

"""#Load model

"""

model=create_model_residual1();
model.load_weights("data/model_residual1_601-800epochs.h5")


#from keras.utils.vis_utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

"""# Data generator"""

from CustomDataGenerator import CustomDataGenerator
training_generator   = CustomDataGenerator(131072,batch_size=BATCH_SIZE,max_peak_count=NPEAK);
validation_generator = CustomDataGenerator( 16384,batch_size=BATCH_SIZE,max_peak_count=NPEAK,validation=True);

"""# Callbacks"""

best_model_file=os.path.join(output_dir,'model.h5');


checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                verbose=1);

log_file=os.path.join(output_dir,'log.csv');
history_cb = tf.keras.callbacks.CSVLogger(log_file, separator=",", append=False)

"""# Training"""

history = model.fit(training_generator,
                    epochs=EPOCAS,
                    validation_data=validation_generator,
                    callbacks=[checkpoint,history_cb],
                    verbose=1
                   );

