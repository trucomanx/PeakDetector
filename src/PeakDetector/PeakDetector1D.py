#!/usr/bin/python
# importing numpy
import numpy as np
from random import randint
from random import random
import PeakDetector.Model as model_lib
import os

class PeakDetector1D:
    
    def __init__(self):
        self.N = 256;
        
        path_lib=os.path.abspath(os.path.dirname(__file__));
        path_model=os.path.join(path_lib,"model","model_residual1.h5")

        self.model=model_lib.create_model_residual1(enable_summary=False);
        self.model.load_weights(path_model);
    
    def FindPeaks(   self, vec):
        dat=np.reshape(vec,(1,self.N,1));
        
        bec=self.model.predict(dat,verbose=False);
        bec=np.reshape(bec,(self.N,));
        
        return bec>0.5;
    
    
    def GenerateRandomVector(   self,
                                func_list=['exp1','gaussian'],
                                peak_count=3,
                                sigma_min=0.1,
                                sigma_max=6,
                                amp_min=0.15,
                                amp_max=1.0,
                                noise_level=0.05):
        
        if peak_count<=0:
            return sys.exit();
        if amp_min<=0:
            return sys.exit();
        if amp_max>1.0:
            return sys.exit();
        if amp_min>amp_max:
            return sys.exit();
        if sigma_min<=0:
            return sys.exit();
        if sigma_min>sigma_min:
            return sys.exit();
        if noise_level<=0:
            return sys.exit();
        if noise_level>=1:
            return sys.exit();
        
        vec=np.zeros(self.N);
        bec=np.zeros(self.N);
        
        peak_count=int(peak_count);
        
        for n in range(peak_count):
            ID=randint(0, len(func_list)-1);
            ti=randint(0, self.N-1);
            sigmai=sigma_min+random()*(sigma_max-sigma_min);
            ai=amp_min+random()*(amp_max-amp_min);
            
            for t in range(self.N):
                if np.abs(t-ti)<=max(0.5*sigmai,1):
                    bec[t]=1;
            
            if func_list[ID]=='exp1':
                for t in range(self.N):
                    vec[t]+=ai*np.exp(-np.abs(t-ti)/sigmai);
            elif func_list[ID]=='gaussian':
                for t in range(self.N):
                    vec[t]+=ai*np.exp(-(t-ti)*(t-ti)/(2*sigmai*sigmai));
            else:
                for t in range(self.N):
                    vec[t]+=ai*np.exp(-np.abs(t-ti)/sigmai);
            
        vec=vec/np.max(vec);
        
        for t in range(self.N):
            vec[t]+=noise_level*(random()-0.5)/0.5;
        
        return np.abs(vec),bec;
