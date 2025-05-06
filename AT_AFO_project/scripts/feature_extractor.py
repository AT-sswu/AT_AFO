import numpy as np
from scripts.fft_processor import apply_fft

def extract_dominant_freq(signal, sampling_rate=200):
    freqs, mags = apply_fft(signal, sampling_rate)
    dominant_idx = np.argmax(mags)
    return freqs[dominant_idx], mags[dominant_idx]
