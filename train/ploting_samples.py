#!/usr/bin/python

import sys

sys.path.append('../src')

import PeakDetector as pkpet
import numpy as np

import matplotlib.pyplot as plt

Det=pkpet.PeakDetector1D();
vec,bec=Det.GenerateRandomVector(   func_list=['exp1','gaussian'],
                                    peak_cout=3,
                                    sigma_min=0.1,
                                    sigma_max=6,
                                    amp_min=0.10,
                                    amp_max=1.0,
                                    noise_level=0.05
                                );

nv=np.linspace(0,len(vec)-1,len(vec));


fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(nv,vec)
axs[1].plot(nv,bec)
plt.show()

