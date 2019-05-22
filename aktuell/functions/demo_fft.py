# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:05:51 2019

@author: Admin_1
"""

import numpy as np
import matplotlib.pyplot as plt 

def demo_leakage(fsig, fS, tmax):
    t = np.arange(0,tmax,np.round(1/fS, 10))
    f = np.arange(0, fS, 1/tmax)
    x = np.sin(2*np.pi*fsig*t)
    X = abs(np.fft.fft(x))/len(t)
    X = np.log(X/max(X) + np.finfo(float).eps)
    plt.figure(figsize=(14,5))
    plt.subplot(121)
    plt.plot(t,x, ':', marker = 'o')
    plt.grid()
    plt.xlabel('t in s')
    plt.title('Zeitfensterinhalt')
    plt.subplot(122)
    plt.plot(f, X, ':', marker = 'o')
    plt.grid()
    plt.xlabel('f in Hz')
    plt.title('Spektrum')
    plt.show()
    
def demo_hamming(fsig, fS, tmax):
    t = np.arange(0,tmax,np.round(1/fS, 10))
    f = np.arange(0, fS, 1/tmax)
    x = np.sin(2*np.pi*fsig*t) * np.hamming(len(t))
    X = abs(np.fft.fft(x))/len(t)
    X = np.log(X/max(X) + np.finfo(float).eps)
    plt.figure(figsize=(14,5))
    plt.subplot(121)
    plt.plot(t,x, ':', marker = 'o')
    plt.grid()
    plt.xlabel('t in s')
    plt.title('Zeitfensterinhalt')
    plt.subplot(122)
    plt.plot(f, X, ':', marker = 'o')
    plt.grid()
    plt.xlabel('f in Hz')
    plt.title('Spektrum')
    plt.show()
    
    
#demo_leakage(10, 70, 0.1)
#demo_hamming(50, 1000, 0.1)