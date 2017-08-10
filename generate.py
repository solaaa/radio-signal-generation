import matplotlib.pyplot as plt
import numpy as np
import scipy
import commpy as cp
from commpy import modulation
import os
from multiprocessing import Process

from numpy import complex, sum, pi, arange, array, size, shape, real, sqrt
from numpy import matrix, sqrt, sum, zeros, concatenate, sinc
from numpy.random import randn, seed, random

CONST_BPSK = 1
CONST_QPSK = 2
CONST_8PSK = 3
CONST_QAM16 = 4
CONST_QAM64 = 6
SYMBOL_NUM = 64
SYMBOL_RATE = 8

SNRs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]

def interpolation(series, symbolRate):
    reshpSerInter = np.array([])
    for L in xrange(len(series)):
        reshpSerInter = np.append(reshpSerInter, series[L])
        for _ in xrange(SYMBOL_RATE-1):
            reshpSerInter = np.append(reshpSerInter, 0)
    return reshpSerInter

def awgn(input_signal, snr_dB):
    """
    from commpy.channel.awgn,
    modifying a bug
    ----------
    Addditive White Gaussian Noise (AWGN) Channel.

    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.

    snr_dB : float
        Output SNR required in dB.

    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(input_signal * input_signal)/len(input_signal)
    snr_linear = 10**(snr_dB/10.0)
    noise_variance = avg_energy/(2*snr_linear)

    
    noise = sqrt(noise_variance) * randn(len(input_signal)) * 1 + sqrt(noise_variance) * randn(len(input_signal)) * 1j


    output_signal = input_signal + noise

    return output_signal

def signalGeneration(SIGNAL_NUM = 50000):
    '''
    init params
    ----------------------
    params:
        X: signals with I and Q series
        Y: labels with the 5 classes, 0-1 5-dim vectors
        Z: SNRs
    classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    '''

    X = np.zeros([SIGNAL_NUM, 2, SYMBOL_NUM*SYMBOL_RATE])
    Y = np.zeros([SIGNAL_NUM, 5])
    Z = np.zeros([SIGNAL_NUM])
    #X2 = np.zeros([SIGNAL_NUM, 2, SYMBOL_NUM*SYMBOL_RATE])

    for iterator in xrange(SIGNAL_NUM):
        pid = os.getpid()
        print ('pid:%d, iterator:%d'%(pid, iterator))
        '''
        generate 5 modulationTypes randomly
        generate #SYMBOL_NUM# symbols randomly
        construct constellation
        '''
        modulationType = np.random.random_integers(1,5,1)
        label = np.zeros(5)
        label[modulationType-1] = 1
        Y[iterator] = label

        
        if modulationType == 1:
            '''BPSK'''
            inputSeries = np.random.random_integers(0,1,SYMBOL_NUM*CONST_BPSK)
            modulationSeries = modulation.PSKModem(2).modulate(inputSeries)
            modulationSeries = interpolation(modulationSeries, SYMBOL_RATE) 
        elif modulationType == 2:
            '''QPSK'''
            inputSeries = np.random.random_integers(0,1,SYMBOL_NUM*CONST_QPSK)
            modulationSeries = modulation.PSKModem(4).modulate(inputSeries)
            modulationSeries = interpolation(modulationSeries, SYMBOL_RATE)
        elif modulationType == 3:
            '''8PSK'''
            inputSeries = np.random.random_integers(0,1,SYMBOL_NUM*CONST_8PSK)
            modulationSeries = modulation.PSKModem(8).modulate(inputSeries)
            modulationSeries = interpolation(modulationSeries, SYMBOL_RATE)
        elif modulationType == 4:
            '''QAM16'''
            inputSeries = np.random.random_integers(0,1,SYMBOL_NUM*CONST_QAM16)
            modulationSeries = modulation.QAMModem(16).modulate(inputSeries)
            modulationSeries = interpolation(modulationSeries, SYMBOL_RATE)
        elif modulationType == 5:
            '''QAM64'''
            inputSeries = np.random.random_integers(0,1,SYMBOL_NUM*CONST_QAM64)
            modulationSeries = modulation.QAMModem(64).modulate(inputSeries)
            modulationSeries = interpolation(modulationSeries, SYMBOL_RATE)
        else:
            print 'incorrect modulationType'
            break
        
  
        '''use RRC'''
        [t, hRRC] = cp.filters.rrcosfilter(33, 0.4, 1, SYMBOL_RATE)
        signal = np.convolve(modulationSeries, hRRC, mode='same')
        

        
        '''add phrase, freq offset'''
        OFFSET_F = 0.01
        OFFSET_PH = float(np.random.randint(0,314))/100
        
        signal = cp.add_frequency_offset(signal, SYMBOL_RATE, OFFSET_F*float(np.random.randint(8,15))/10)
        signal = signal*np.exp(-OFFSET_PH*1j)
        
        '''channel'''
        '''rayleigh channel'''
        H_CHANNEL = [0.342997170285018 + 2.75097576175635j,0.342995668431465 + 2.75097418548155j,
                     0.342991162874393 + 2.75096945665969j,0.342983653624566 + 2.75096157529839j,
                     0.342973140699920 + 2.75095054141035j,0.342959624125568 + 2.75093635501332j,
                     0.342943103933798 + 2.75091901613016j,0.342923580164072 + 2.75089852478880j,
                     0.342901052863027 + 2.75087488102222j,0.342875522084475 + 2.75084808486853j]
        
        signal = np.convolve(signal, H_CHANNEL, mode='same')
        
        '''noise by quantitative SNR'''
        snr = SNRs[np.random.randint(0, len(SNRs))]
        
        Z[iterator] = snr

        #signal2 = signal
        
        signal = awgn(signal, snr)


        
        '''regulization'''
        length = np.zeros([len(signal)])
        for i in xrange(len(signal)):
            length[i] = np.sqrt(np.power(np.imag(signal[i]), 2) + np.power(np.real(signal[i]), 2))
            
        maxLen = length[np.where(length == np.max(length))]
        signal = signal/maxLen[0]             
        
          
        '''reconstruction'''
        signalI = np.real(signal)
        signalQ = np.imag(signal)

        X[iterator][0] = signalI
        X[iterator][1] = signalQ
        
        #X2[iterator][0] = np.real(signal2)
        #X2[iterator][1] = np.imag(signal2)
        

        
    return [X, Y, Z]

def process1():
    s = signalGeneration()
    np.save('train_set.npy', s[0])
    np.save('train_label.npy', s[1])
    np.save('train_snr.npy', s[2])
    #np.save('train_set_2.npy', s[3])
    

def process2():
    s = signalGeneration()
    np.save('test_set.npy', s[0])
    np.save('test_label.npy', s[1])
    np.save('test_snr.npy', s[2])

def test():
    X_train = np.load('train_set.npy')
    Z = np.load('train_snr.npy')
    Y = np.load('train_label.npy')
    X2 = np.load('train_set_2.npy')
    
    for i in range(1000):
        if Z[i] == 4:
            print i
    
    
    num = 835
    print(Z[num])
    print(Y[num])
    sample0 = X_train[num]
    sample1 = X2[num]
    
    plt.figure(1)
    #plt.plot(sample1[0])
    plt.plot(sample0[0])
    plt.plot(sample0[1])
    plt.figure(2)
    #plt.plot(sample1[0])
    plt.plot(sample1[0])
    plt.plot(sample1[1])    
    plt.figure(3)
    plt.plot(sample1[0], sample1[1])
    plt.plot(sample0[0], sample0[1])
    
    AWGN_FACTOR0 = sample0[0] - sample1[0]
    AWGN_FACTOR1 = sample0[1] - sample1[1]

    plt.figure(4)
    #plt.plot(AWGN_FACTOR[0], AWGN_FACTOR[1])  
    plt.plot(AWGN_FACTOR0)  
  
    
    plt.show()    

def main():

    p1 = Process(target = process1)
    p2 = Process(target = process2)
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()        
    
    #test()
    
    
    
if __name__ == '__main__':
    main()