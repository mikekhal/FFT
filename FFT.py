# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 06:09:50 2024

@author: mike_
"""
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Read the audio file
sample_rate, data = wav.read('Bad_Sample_1 (1).wav')

# FFT the data
fft_data = np.fft.fft(data)
# Return sample frequencies
freqs = np.fft.fftfreq(len(data), d=1./sample_rate)

# Print frequencies and Fourier coefficients
print('Frequencies: ', freqs)
print('Fourier Coefficients: ', fft_data)

# Plot the frequency values as a function of array index
plt.plot(freqs)
plt.xlabel('Array index')
plt.ylabel('Frequency [Hz]')
plt.savefig("frequency1.png")

plt.show()

# compute the power spectral density (power of each frequency)
psd = np.abs(freqs) / len(data)


# Shift the zero-frequency component to the center
psd_shifted = np.fft.fftshift(np.real(fft_data*np.conj(fft_data)))


# Plotting the power spectrum of the shifted frequencies
plt.plot(np.fft.fftshift(freqs), psd_shifted)
plt.yscale('log')
plt.ylabel('Power Spectral Density')
plt.xlabel('Frequency [Hz]')
plt.savefig("PowerSpectrumPlot.png")

plt.show()

# Set thresholds for filtering
lower_threshold = -2000
upper_threshold = 2000

# C;ean PSD by zeroing out components
indices_to_zero = np.where((np.abs(freqs) < lower_threshold) | (np.abs(freqs) > upper_threshold))

fft_data_clean = np.copy(fft_data)
fft_data_clean[indices_to_zero] = 0

# Plot the filtered PSD/denoised signal
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.real(fft_data_clean * np.conj(fft_data_clean))))
plt.yscale('log')
plt.ylabel('Power Spectral Density')
plt.xlabel('Frequency [Hz]')
plt.savefig("filteredPSD.png")

plt.show()

# Inverse FFT to get back to the time domain
filtered_data = np.fft.ifft(fft_data_clean)
filtered_data = np.real(filtered_data)

# Amplify the audio
gain = 1000.0
amplified_data = filtered_data * gain

# Save the amplified and filtered audio to a new WAV file
wav.write('amplified_filtered_output.wav', sample_rate, amplified_data)
