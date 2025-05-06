import numpy as np
from scripts.fft_processor import apply_fft  # FFT 처리 함수 임포트

def extract_dominant_freq(signal, sampling_rate=200):
    """
    주어진 진동 신호에서 지배적인 주파수와 해당 진폭 추출

    Parameters:
        signal (array-like): 진동 신호 데이터
        sampling_rate (int): 샘플링 주파수 in Hz (기본값: 200Hz)

    Returns:
        tuple: (dominant_freq, dominant_magnitude)
    """

    if signal is None or len(signal) == 0:
        raise ValueError("입력 신호가 비어 있습니다.")

    # FFT 수행
    freqs, magnitudes = apply_fft(signal, sampling_rate)

    dominant_index = np.argmax(magnitudes)
    dominant_freq = freqs[dominant_index]
    dominant_magnitude = magnitudes[dominant_index]

    return dominant_freq, dominant_magnitude
