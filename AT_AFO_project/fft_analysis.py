import numpy as np

# FFT 분석 함수
def fft_analysis(data, sample_rate):
    n = len(data)
    x = np.fft.fftfreq(n, 1 / sample_rate)
    y = np.fft.fft(data) / n
    mask = x >= 0
    return x[mask], np.abs(y[mask])
