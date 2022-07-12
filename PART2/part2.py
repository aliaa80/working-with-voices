from scipy.io import wavfile
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from numpy import random
from librosa import stft, istft


def AWGN_noise(sampling_rate, data, SNR):

    root_mean_Square_signal = sqrt(np.mean(data**2))
    root_mean_Square_noise = sqrt(root_mean_Square_signal**2/(pow(10, SNR/10)))
    noise = (random.normal(0, root_mean_Square_noise,
             data.shape[0])).astype(np.float32)
    noisy = np.add(data, noise)
    wavfile.write('noisy.wav', sampling_rate, noisy.astype(np.int16))
    return noisy, noise


def spectral_subtraction(sampling_rate, noisy, noise):
    fft_noisy_signal = (stft(noisy))
    absolute_value_fft_noisy_signal = np.abs(fft_noisy_signal)
    b = np.exp(1.0j * np.angle(fft_noisy_signal))

    fft_noise = stft(noise)
    absolute_value_fft_noise = np.abs(fft_noise)

    average_fft_noise = np.mean(absolute_value_fft_noise, axis=1)
    average_fft_noise = average_fft_noise.reshape(
        (average_fft_noise.shape[0], 1))
    denoised_signal = istft(
        (absolute_value_fft_noisy_signal - average_fft_noise) * b)
    wavfile.write('denoised.wav', sampling_rate,
                  denoised_signal.astype(np.int16))


def plot(filename):
    sampling_rate, complete_data = wavfile.read(filename)
    data = complete_data.astype(np.float32)
    n = data.shape[0] / sampling_rate
    t = np.linspace(0, n, data.shape[0])
    plt.title(filename[:-4])
    plt.plot(t, data)
    plt.savefig(filename[:-4]+'.png')
    plt.close()


sampling_rate, comlete_data = wavfile.read("Test.wav")
data = comlete_data.astype(np.float32)
noisy, noise = AWGN_noise(sampling_rate, data, 1)
spectral_subtraction(sampling_rate, noisy, noise)

plot('Test.wav')
plot('noisy.wav')
plot('denoised.wav')
