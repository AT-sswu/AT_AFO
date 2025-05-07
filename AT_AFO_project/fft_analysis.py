import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob('data/*.csv')

for file in csv_files:
    df = pd.read_csv(file)

    sample_rate = 1000  # 1초당 샘플 수 -> 추후 조율 필요
    duration = len(df) / sample_rate
    n = int(sample_rate * duration)


    # FFT 분석 함수 정의
    def fft_analysis(column_name):
        data = df[column_name].dropna().to_numpy()  # NaN 제거 및 배열로 변환
        x = np.fft.fftfreq(n, 1 / sample_rate)  # 주파수 축
        y = np.fft.fft(data) / len(data)  # FFT 후 정규화

        # 양의 주파수 부분만 -> 시각화
        mask = x >= 0
        x_plot = x[mask]
        y_plot = np.abs(y[mask])

        return x_plot, y_plot


    # 각 축에 대해 FFT 분석 및 시각화
    axes = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    plt.figure(figsize=(18, 12))  # 한 번에 여러 그래프 출력할 수 있도록 넓게 설정

    for i, axis in enumerate(axes):
        x_plot, y_plot = fft_analysis(axis)

        # 서브플롯에 -> 그래프 
        plt.subplot(3, 2, i + 1)
        plt.plot(x_plot, y_plot)
        plt.title(f'FFT of {axis}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
