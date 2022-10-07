
import numpy as np
import tensorflow as tf

import sys

sys.path.append('../src')

import PeakDetector as pkpet
from random import randint
from random import seed

class CustomDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(   self, 
                    L,
                    max_peak_count=5,
                    validation=False,
                    batch_size=32):
        'Initialization'
        self.Det=pkpet.PeakDetector1D();
        self.Nel=256;
        
        self.validation=validation;
        self.L=L;
        self.max_peak_count=max_peak_count;
        self.batch_size = batch_size;
        

    def __len__(self):
        'OBLIGATORIO: Denotes the number of batches per epoch'
        return int(np.floor(self.L / self.batch_size))


    def __getitem__(self, index):
        'OBLIGATORIO: Generate one batch of data'
        # Generate data
        if self.validation==False:
            seed(index);
        X, y = self.__temporal_data_generation()

        return X, y

    def on_epoch_end(self):
        'OPCIONAL: Updates indexes after each epoch'
        return;

    def __temporal_data_generation(self):
        'Generates data containing batch_size samples'  
        # Initialization
        X = np.zeros((self.batch_size, self.Nel))
        y = np.zeros((self.batch_size, self.Nel))

        # Generate data
        for i in range(self.batch_size):
            # Store sample
            peak_count=randint(1,self.max_peak_count);
            vec,bec=self.Det.GenerateRandomVector(  func_list=['exp1','gaussian'],
                                                    peak_count=peak_count,
                                                    sigma_min=0.1,
                                                    sigma_max=6,
                                                    amp_min=0.10,
                                                    amp_max=1.0,
                                                    noise_level=0.05
                                                );
            X[i,] = vec;

            # Store class
            y[i,] = bec;

        return X, y
