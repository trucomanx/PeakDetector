#!/usr/bin/python
# importing numpy
import numpy as np

class PeakDetector:
    
    def __init__(self):
        self.N = 256;
    
    def GenerateRandomVector(self):
        return np.random.rand(1,self.N);
