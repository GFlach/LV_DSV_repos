# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:33:24 2017

@author: Admin_1
"""

import numpy as np
import matplotlib.pyplot as plt


def create_DFT(N = 32):

    e = 1j*np.complex(-2*np.pi/N)
    #w = np.round(np.exp(e),4)
    w = np.exp(e)

    n = np.arange(N)
    for i in np.arange(N-1):
        n = np.append(n, np.arange(N))
    n = n.reshape(N,N)

    k = []
    k1 = np.zeros(N)
    for i in np.arange(N):
        k = np.append(k, k1+i)
    k = k.reshape(N,N)

    exponent = n*k

    DFT = 1/np.sqrt(N)*w**exponent

    return(DFT)

def create_sigvec(tS = 0.25, freq = [1,2], amp = [1, 0.5], phi = [0, np.pi/2], atw = 9):
    """
    Erzeugen eines Abtastwertevektors aus Kosinusschwingungen
    Parameter: 
        tS - Abtastintervall in s
        freq - Liste von Frequenzen in Hz
        amp - Liste von Amplitugen
        phi - Liste von Phasenwinkeln
        atw - Anzahl Abtastwerte
    Rückgabe:
        sigvec - Abtastwertevektor
        0 - im Fehlerfall
    """
    if len(freq) != len(amp) != len(phi):
        return(0)
    t = np.arange(0,tS*atw,tS*atw/1000)
    sig = np.zeros(len(t))
    for i in range(len(freq)):
        sig = sig + amp[i] * np.cos(2*np.pi*freq[i]*t + phi[i])
    
    t1 = np.arange(0,tS*atw,tS)
    t1 = np.array(t1)
    dsig = sig[(t1*1000/(tS*atw)).astype(int)]
    plt.plot(t,sig,'--')
    plt.stem(t1,dsig)
    plt.plot([-0.1,tS*atw+0.1],[0,0],'k')
    plt.plot([0, 0],[min(sig)-0.1, max(sig)+0.1],'k')
    plt.xlabel('Analysezeitfenster (s)')
    plt.title('Signalvektor')
    plt.grid()
    plt.axis([-0.1, tS*atw+0.1, min(sig)-0.1, max(sig)+0.1])
    plt.savefig('sigvec.jpg')
    plt.show()    
    sigvec = np.matrix(dsig).T

    return(sigvec)

def create_DFT_inv(DFT):
    N = DFT.shape[0]
    DFT = np.sqrt(N)*DFT
    DFT_inv = 1/np.sqrt(N)*(DFT**(-1))
    return (DFT_inv)
        
def amp_spec(X, tdft):
    f = np.arange(0, len(X)/tdft, 1/tdft)
    #print(f)
    b = np.round(np.abs(X),2)
    plt.stem(f, b)    
    plt.grid()
    plt.axis([-0.1, len(X)/tdft+1, min(b)-0.1, max(b)+0.1])
    plt.plot([-0.1, len(X)/tdft+1],[0,0],'k')
    plt.plot([0,0],[min(b)-0.1, max(b)+0.1],'k')
    plt.xlabel('Frequenz (Hz)')
    plt.title('Amplitudenspektrum')
    plt.show()
    
def power_spec(X, tdft):
    f = np.arange(0, len(X)/tdft, 1/tdft)
    #print(f)
    b = np.round(np.abs(X)**2,2)
    plt.stem(f, b)    
    plt.grid()
    plt.axis([-0.1, len(X)/tdft+1, min(b)-0.1, max(b)+0.1])
    plt.plot([-0.1, len(X)/tdft+1],[0,0],'k')
    plt.plot([0,0],[min(b)-0.1, max(b)+0.1],'k')
    plt.xlabel('Frequenz (Hz)')
    plt.title('Leistungsspektrum')
    plt.show()
    
def power_spec_norm(X, tdft):
    f = np.arange(0, len(X)/tdft, 1/tdft)
    #print(f)
    b = np.round((np.abs(X)/max(np.abs(X)))**2,2)
    plt.stem(f, b)    
    plt.grid()
    plt.axis([-0.1, len(X)/tdft+1, min(b)-0.1, max(b)+0.1])
    plt.plot([-0.1, len(X)/tdft+1],[0,0],'k')
    plt.plot([0,0],[min(b)-0.1, max(b)+0.1],'k')
    plt.xlabel('Frequenz (Hz)')
    plt.title('normiertes Leistungsspektrum')
    plt.show()

def power_spec_log(X, tdft):
    f = np.arange(0, len(X)/tdft, 1/tdft)
    #print(f)
    norm_90 = np.abs(X) + 10**(-4.5)*max(np.abs(X)) # -90dB als Minimum
    b = np.round(20 * np.log10(norm_90/max(np.abs(X))),2)
    plt.plot(f, b)    
    plt.grid()
    plt.axis([-0.1, len(X)/tdft+1, min(b)-0.1, max(b)+0.1])
    plt.plot([-0.1, len(X)/tdft+1],[0,0],'k')
    plt.plot([0,0],[min(b)-0.1, max(b)+0.1],'k')
    plt.xlabel('Frequenz (Hz)')
    plt.title('Logarithmiertes Leistungsspektrum')
    plt.show()
    
def power_spec_log_stem(X, tdft):
    f = np.arange(0, len(X)/tdft, 1/tdft)
    #print(f)
    norm_90 = np.abs(X) + 10**(-4.5)*max(np.abs(X)) # -90dB als Minimum
    b = np.round(20 * np.log10(norm_90/max(np.abs(X))),2)
    plt.stem(f, b)    
    plt.grid()
    plt.axis([-0.1, len(X)/tdft+1, min(b)-0.1, max(b)+0.1])
    plt.plot([-0.1, len(X)/tdft+1],[0,0],'k')
    plt.plot([0,0],[min(b)-0.1, max(b)+0.1],'k')
    plt.xlabel('Frequenz (Hz)')
    plt.title('Logarithmiertes Leistungsspektrum')
    plt.show()

def phase_spec(X, tdft):
    f = np.arange(0, len(X)/tdft, 1/tdft)
    phi = np.angle(X)
    phi[np.round(np.abs(X),1) < 0.01] = 0
    plt.stem(f, phi)    
    plt.grid()
    plt.axis([-0.1, len(X)/tdft+1, min(phi)-0.1, max(phi)+0.1])
    plt.plot([-0.1, len(X)/tdft+1],[0,0],'k')
    plt.plot([0,0],[min(phi)-0.1, max(phi)+0.1],'k')
    plt.title('Phasenspektrum')
    plt.xlabel('Frequenz (Hz)')
    plt.show()

def dsv_triangle(type=1, offs=0, fS=1000, dur=0.2, T0=0.1, amp=2, plot=1):
    """
    Erzeugen eines Dreiecksignals
    Parameter: 
        type - 0: gerade, 1: ungerade
        offset - Mittelwert
        fS - Abtastfrequenz
        dur - Dauer des erzeugten Signals
        T0 - Periodendauer
        amp - Amplitude
    Rückgabe:
        Dreiecksignal
        0 - Fehlerfall
    """
    if fS * T0 < 2:
        print('Abtasttheorem verletzt')
        return(0)
    atwp = T0 * fS
    x = np.linspace(-amp, amp, atwp/2, endpoint=False)
    y = np.zeros(int(atwp))
    #y = np.zeros(len(x))
    y1 = []
    if type == 1:
        y[0:int(atwp/4)] = x[int(atwp/4):int(atwp/2)] + offs
        y[int(atwp/4):int(3*atwp/4)] = -x[0:int(atwp/2)] + offs
        y[int(3*atwp/4):int(atwp)] = x[int(atwp/4):int(atwp/2)] -amp + offs
    else:
        y[0:int(atwp/2)] = -x[0:int(atwp/2)] + offs
        y[int(atwp/2):int(atwp)] = x[0:int(atwp/2)] + offs
    
    for i in range(int(dur//T0)):
        y1 = np.append(y1,y)
    
    y1 = np.append(y1, y[0:int((dur%T0)*fS)])
    if  plot == 1:
        t = np.linspace(0, dur, fS*dur)
        plt.plot(t,y1)
        plt.grid()
        plt.title('Dreiecksignal')
        plt.xlabel('t in s')
        plt.axis([0, max(t), min(y1)+0.1*min(y1), max(y1)+0.1*max(y1)])
        plt
        plt.show()
    return(y1)
    
def dsv_harm(type=1, offs=0, fS=1000, dur=0.2, T0=0.01, amp=0.4, plot=1):
    """
    Erzeugen einer harmonischen Schwingung
    Parameter: 
        type - 0: cos, 1: sin
        offset - Mittelwert
        fS - Abtastfrequenz
        dur - Dauer des erzeugten Signals
        T0 - Periodendauer
        amp - Amplitude
    Rückgabe:
        harmonische Schwingung
        0 - Fehlerfall
    """
    if fS * T0 < 2:
        print('Abtasttheorem verletzt')
        return(0)
    atwp = T0 * fS
    t = np.arange(0,atwp)
    y1 = []
    if type == 1:
        y = amp*np.sin(2*np.pi/T0*t/fS)
    else:
        y = amp*np.cos(2*np.pi/T0*t/fS)
    
    for i in range(int(dur//T0)):
        y1 = np.append(y1,y)
    
    y1 = np.append(y1, y[0:int((dur%T0)*fS)])
    if  plot == 1:
        t = np.linspace(0, dur, fS*dur)
        plt.plot(t,y1)
        plt.grid()
        plt.title('harmonische Schwingung')
        plt.xlabel('t in s')
        plt.axis([0, max(t), min(y1)+0.1*min(y1), max(y1)+0.1*max(y1)])
        plt
        plt.show()
        plt.show()
    return(y1)

 
 
