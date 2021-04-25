#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:16:55 2019

@author: cxx
"""
import math

class model(object):
    
    def __init__(self, diffusion = 0.1, decay = 0.004, width = 1,scaling = 50):
        self.diffusion = diffusion
        self.decay = decay
        self.width = width
        self.scaling = scaling
        
    def expectedHit(self, dx,dy, isTrapped,isSide):
        width = self.width + dx * self.diffusion
        r_y = math.exp(-2 * (dy / width) ** 2)
        if dx >= 0 :
            r_x = 1 - self.decay * (dx ** 2)
            if r_x < 0 :
                r_x = 0
        else:
            r_x = 0
        return self.scaling*r_x * r_y*(1 - isTrapped) + self.scaling/2/(dx**2+dy**2+1)+ self.scaling/2*r_x*isSide
