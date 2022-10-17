#!/usr/bin/python

import sys

sys.path.append('../src')

import PeakDetector as pkpet
import matplotlib.pyplot as plt

Det=pkpet.PeakDetector1D();
vec_input,vec_target=Det.GenerateRandomVector(  func_list=['exp1','gaussian'],
                                                peak_count=3,
                                                sigma_min=0.1,
                                                sigma_max=6,
                                                amp_min=0.10,
                                                amp_max=1.0,
                                                noise_level=0.05
                                             );

vec_predict = Det.FindPeaks(vec_input);

import numpy as np
nv=np.linspace(0,len(vec_input)-1,len(vec_input));


fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(nv,vec_input)
axs[0].set_title('Input');

axs[1].plot(nv,vec_target)
axs[1].set_title('Target');

axs[2].plot(nv,vec_predict);
axs[2].set_title('Predict');
plt.show()

