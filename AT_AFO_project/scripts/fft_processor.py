import numpy as np

def apply_fft(signal, sampling_rate=200):  # Hz 단위
    # FFT 적용
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)

    # 양의 주파수만 사용
    positive_freqs = fft_freqs[np.where(fft_freqs >= 0)]
    positive_magnitudes = np.abs(fft_vals[np.where(fft_freqs >= 0)])

    return positive_freqs, positive_magnitudes
