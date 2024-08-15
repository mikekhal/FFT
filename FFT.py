# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 06:09:50 2024

@author: mike_
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
opd = np.linspace(0, 10, 1000)  # Optical Path Difference (micrometers)
delta_opd = opd[1] - opd[0]     # Sampling interval

# Hydrogen Spectrum Simulation
def hydrogen_spectrum(opd, wavelengths, amplitudes):
    interferogram = np.zeros_like(opd)
    for wavelength, amplitude in zip(wavelengths, amplitudes):
        frequency = 1 / wavelength
        interferogram += amplitude * np.cos(2 * np.pi * frequency * opd)
    return interferogram

# White Light Spectrum Simulation
def white_light_spectrum(opd, mean_wavelength, sigma):
    frequencies = np.fft.fftfreq(len(opd), delta_opd)
    gaussian_spectrum = np.exp(-0.5 * (frequencies - 1 / mean_wavelength)**2 / (sigma**2))
    interferogram = np.fft.ifft(gaussian_spectrum)
    return np.real(interferogram)

# Hydrogen Spectral Lines (wavelengths in micrometers)
hydrogen_wavelengths = [0.4861, 0.6563, 0.4340]  # H-beta, H-alpha, H-gamma
hydrogen_amplitudes = [1.0, 0.8, 0.6]  # Arbitrary units
hydrogen_interferogram = hydrogen_spectrum(opd, hydrogen_wavelengths, hydrogen_amplitudes)

# White Light Spectrum
mean_wavelength = 0.55  # Central wavelength in micrometers
sigma = 0.1            # Standard deviation of the Gaussian (broadness of the spectrum)
white_light_interferogram = white_light_spectrum(opd, mean_wavelength, sigma)

# Plot Hydrogen Interferogram
plt.figure(figsize=(12, 6))
plt.plot(opd, hydrogen_interferogram)
plt.xlabel('Optical Path Difference (micrometers)')
plt.ylabel('Intensity (arbitrary units)')
plt.title('Simulated Hydrogen Spectral Lamp Interferogram')
plt.grid(True)
plt.show()

# Plot White Light Interferogram
plt.figure(figsize=(12, 6))
plt.plot(opd, white_light_interferogram)
plt.xlabel('Optical Path Difference (micrometers)')
plt.ylabel('Intensity (arbitrary units)')
plt.title('Simulated White Light Interferogram')
plt.grid(True)
plt.show()

# Perform FFT on Hydrogen Interferogram
fft_result_hydrogen = np.fft.fft(hydrogen_interferogram)
freqs_hydrogen = np.fft.fftfreq(len(hydrogen_interferogram), delta_opd)
magnitude_hydrogen = np.abs(fft_result_hydrogen)

# Perform FFT on White Light Interferogram
fft_result_white_light = np.fft.fft(white_light_interferogram)
freqs_white_light = np.fft.fftfreq(len(white_light_interferogram), delta_opd)
magnitude_white_light = np.abs(fft_result_white_light)

# Plot Frequency Spectrum of Hydrogen
plt.figure(figsize=(12, 6))
plt.plot(freqs_hydrogen[:len(freqs_hydrogen)//2], magnitude_hydrogen[:len(magnitude_hydrogen)//2])
plt.xlabel('Frequency (cycles per micrometer)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Hydrogen Spectral Lamp')
plt.grid(True)
plt.show()

# Plot Frequency Spectrum of White Light
plt.figure(figsize=(12, 6))
plt.plot(freqs_white_light[:len(freqs_white_light)//2], magnitude_white_light[:len(magnitude_white_light)//2])
plt.xlabel('Frequency (cycles per micrometer)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of White Light')
plt.grid(True)
plt.show()
