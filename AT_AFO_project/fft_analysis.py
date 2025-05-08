import numpy as np
import matplotlib.pyplot as plt

# FFT 분석
def fft_analysis(data, sample_rate, plot=True):
    n = len(data)
    x = np.fft.fftfreq(n, 1 / sample_rate)
    y = np.fft.fft(data) / n
    mask = x >= 0
    freqs = x[mask]
    amps = np.abs(y[mask])

    # 공진 주파수 (최대 진폭을 가지는 주파수)
    resonance_freq = freqs[np.argmax(amps)]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(freqs, amps)
        plt.title("FFT Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.legend()
        plt.show()

    return freqs, amps, resonance_freq
