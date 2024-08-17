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
# Get an array for the frequencies in correct units (Hz)
freqs = np.fft.fftfreq(len(data), d=1./sample_rate)

# Print frequencies and Fourier coefficients
print('Frequencies: ', freqs)
print('Fourier Coefficients: ', fft_data)

# Plot the frequency values as a function of array index
plt.plot(freqs)
plt.xlabel('Array index')
plt.ylabel('Frequency [Hz]')
plt.show()

# Plot the power spectral density
PSD_audio = np.fft.fftshift(np.real(fft_data * np.conj(fft_data)))
plt.plot(np.fft.fftshift(freqs), PSD_audio)
plt.yscale('log')
plt.ylabel('Power Spectral Density')
plt.xlabel('Frequency [Hz]')
plt.show()

# Set thresholds for filtering
lower_threshold = -2000
upper_threshold = 2000

# Find indices where freqs satisfy the condition
indices_to_zero = np.where((np.abs(freqs) < lower_threshold) | (np.abs(freqs) > upper_threshold))

# Copy FFT data and zero out the frequencies outside the thresholds
fft_data_clean = np.copy(fft_data)
fft_data_clean[indices_to_zero] = 0

# Plot the filtered power spectral density
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.real(fft_data_clean * np.conj(fft_data_clean))))
plt.yscale('log')
plt.ylabel('Power Spectral Density')
plt.xlabel('Frequency [Hz]')
plt.show()

# Plot the zoomed-in power spectral density
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.real(fft_data_clean * np.conj(fft_data_clean))))
plt.xlim(0, upper_threshold)
plt.yscale('log')
plt.ylabel('Power Spectral Density')
plt.xlabel('Frequency [Hz]')
plt.show()

# Inverse FFT to get back to the time domain
filtered_data = np.fft.ifft(fft_data_clean)
filtered_data = np.real(filtered_data)

# Amplify the audio
gain = 1000.0  # Adjust this value to increase or decrease the amplification
amplified_data = filtered_data * gain

# Normalize the amplified data to avoid clipping
amplified_data = np.int16(amplified_data / np.max(np.abs(amplified_data)) * 32767)

# Save the amplified and filtered audio to a new WAV file
wav.write('amplified_filtered_output.wav', sample_rate, amplified_data)

print("Amplified and filtered audio saved as 'amplified_filtered_output.wav'.")

