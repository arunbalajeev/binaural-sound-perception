import pyaudio
import wave
import sys,cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
import wavio

def plot_with_markers(saudio1,rate=48000):
    samplePoints = float(saudio1.shape[0]);#samplePoints2 = float(saudio2.shape[0])
    signalDuration =  saudio1.shape[0] / rate;#signalDuration2 =  saudio2.shape[0] / rate
    timeArray = np.arange(0, samplePoints, 1);#timeArray2 = np.arange(0, samplePoints2, 1)
    timeArray = timeArray / rate;#timeArray2 = timeArray2 / rate
    timeArray = timeArray * 1000;#timeArray2 = timeArray2 * 1000
    fig,axs = plt.subplots(1)
    axs.plot(timeArray, saudio1, color='G')
    axs.set_title("Mic track without cuts")
    #wavio.write("outdoor/test/VIDEO_0266_cropped.WAV",saudio1[front_cut:-back_cut],rate,sampwidth=3)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_freq(fftArray,mySoundLength):
    numUniquePoints = np.ceil((mySoundLength ))
    freqArray = np.arange(0, numUniquePoints, 1.0) * (48000 / mySoundLength);
    print(len(freqArray))
    plt.plot(freqArray/1000, 10 * np.log10 (fftArray), color='B')
    plt.xlabel('Frequency (Khz)')
    plt.ylabel('Power (dB)')
    plt.show()

def mapi(x):
    return (x -np.min(x))+1e-4/(np.max(x)-np.min(x))

thresh = 0.0002
adict={};energyArr=[]

for sc in range(1,166):
    if sc<37:    fdir="/dataset/outdoor/scene"+str(sc)+"/"
    else:    fdir="/dataset/outdoor/scene%04d/"%sc
    adict[sc]=[];print(sc,len(glob.glob(fdir+"wavsplits/Track1/"+"*.npy")))
    for g in glob.glob(fdir+"wavsplits/Track1/"+"*.npy"):
        soundCH = np.load(g)
        #plot_with_markers(soundCH)
        mySoundLength = len(soundCH)
        fftArray = fft(soundCH)
        fftArray = abs(fftArray)
        fftArray = fftArray / float(mySoundLength)
        fftArray = fftArray **2
        #plot_freq(fftArray,mySoundLength)
        energy = np.sum(fftArray);energyArr.append(energy);
        if energy>thresh:    adict[sc].append(fdir+"spectrograms/Track1/"+g.split('/')[-1].split('.')[0]+".npy")


np.save('/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/audio/dataset/SoundEnergy_165scenes.npy',adict)
print("Saved sound scenes.")

### To plot the energy plot ###
"""
energyArr = np.load('/srv/beegfs02/scratch/language_vision/data/Sound_Event_Prediction/audio/dataset/SoundEnergy_165scenes.npy')
print(len(energyArr))
energyArr=[i for i in energyArr if i<0.1]
n, bins, patches = plt.hist(energyArr,'auto',density=True, facecolor='g', alpha=0.75)

num1 = sum(i > thresh for i in energyArr)
num0 = sum(i < thresh for i in energyArr)
print("Percent > thresh: %f"% (num1/float(num1+num0)))
print("Percent < thresh: %f"% (num0/float(num1+num0)))
plt.title("Audio Energy distribution of our dataset samples",fontsize=20)
plt.xlabel("Sound Energy (~Hz^-2)",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.xlim(0.0, 0.02)
plt.tick_params(axis="x", labelsize=18)
plt.tick_params(axis="y", labelsize=18)
plt.grid(True)
plt.show()
"""
