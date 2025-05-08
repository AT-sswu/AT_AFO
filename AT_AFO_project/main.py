import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fft_analysis import fft_analysis
from model_train import train_and_predict

# CSV FFT 분석 및 공진 주파수 저장
def process_csv_files(data_dir='data', sample_rate=4096): #sample_rate 수정
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    for file in csv_files:
        df = pd.read_csv(file)

        axes = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
        peak_frequencies = []

        # 축별 FFT 분석
        for axis in axes:
            if axis in df.columns:
                data = df[axis].dropna().to_numpy()
                if len(data) == 0:
                    continue
                x_fft, y_fft = fft_analysis(data, sample_rate)
                peak_freq = x_fft[np.argmax(y_fft)]
                peak_frequencies.append(peak_freq)

                plt.plot(x_fft, y_fft, label=f'{axis}')

        # 공진 주파수 계산
        if peak_frequencies:
            resonant_freq = np.mean(peak_frequencies)
            df['Resonant_Frequency'] = resonant_freq
            df.to_csv(file, index=False)
            print(f" {os.path.basename(file)} → 공진 주파수: {resonant_freq:.2f} Hz")
        else:
            print(f" {os.path.basename(file)}: 데이터 없음")

        # FFT 결과
        plt.title(f'FFT for {os.path.basename(file)}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    process_csv_files()     # FFT 분석 + CSV 업데이트
    train_and_predict()
