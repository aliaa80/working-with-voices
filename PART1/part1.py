import scipy.io.wavfile
import math
import glob
import os
import numpy as np
from numpy import fft
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt


def create_power_spectrum(filename):
    sampling_rate, data = scipy.io.wavfile.read(filename)
    sum_stereo = data.sum(axis=1) / 2
    freq_nq = len(sum_stereo) // 2
    power_spectrum = (abs(fft.fft(sum_stereo))[:freq_nq] / len(data) * 2) ** 2
    frequency = fft.fftfreq(len(data), 1/sampling_rate)[:freq_nq]
    return(frequency, power_spectrum)


def freq_for_maximum_power(filename):
    frequency, power_spectrum = create_power_spectrum(filename)
    max = frequency[np.argmax(power_spectrum)]
    print(filename[7:-4] + '  ' + str(max))
    return(max)


def sex_verification(folder):
    list = []
    for filename in glob.glob(os.path.join(folder, '*.wav')):
        max = freq_for_maximum_power(filename)
        sex = 'woman'
        if (max) <= 180:
            sex = 'man'
        list.append(filename[7:-4]+": " + sex)
        show_plot(filename, filename[7:-4], sex)

    for i in list:
        print(i)


def show_plot(filename, i, sex):
    frequency, power_spectrum = create_power_spectrum(filename)
    plt.plot(frequency, power_spectrum)
    plt.title(sex)
    plt.savefig('output/'+i+'.png')
    plt.close()


sex_verification('voices')
