import numpy as np

def apply_fft(signal, sampling_rate=1):  # MATLAB의 Fs = 1과 동일
    """
    FFT 분석 함수
    :param signal: 입력 신호 (1D NumPy 배열)
    :param sampling_rate: 샘플링 주파수 (Hz)
    :return: 양의 주파수 성분과 해당 크기
    """

    L = len(signal)                 # 샘플 개수
    T = 1 / sampling_rate           # 샘플링 주기
    t = np.arange(0, L) * T         # 시간 벡터 (선택사항, 시각화용)

    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(L, T)

    # 양의 주파수만 필터링
    pos_mask = fft_freqs >= 0
    positive_freqs = fft_freqs[pos_mask]
    positive_magnitudes = np.abs(fft_vals[pos_mask])

    return positive_freqs, positive_magnitudes
