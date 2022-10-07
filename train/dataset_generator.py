#!/usr/bin/python

import sys

sys.path.append('../src')

import PeakDetector as pkpet
from random import randint


Det=pkpet.PeakDetector1D();

peak_count=randint(1,5);
vec,bec=Det.GenerateRandomVector(   func_list=['exp1','gaussian'],
                                    peak_count=peak_count,
                                    sigma_min=0.1,
                                    sigma_max=6,
                                    amp_min=0.10,
                                    amp_max=1.0,
                                    noise_level=0.05
                                );



