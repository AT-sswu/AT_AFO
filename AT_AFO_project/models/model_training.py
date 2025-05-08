import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from fft_analysis import fft_analysis

# ML 모델 학습 + 예측 함수
def train_and_predict(data_dir='data', sample_rate=4096):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    X, y = [], []

    for file in csv_files:
        df = pd.read_csv(file)

        if 'Resonant_Frequency' not in df.columns or pd.isnull(df['Resonant_Frequency'].iloc[0]):
            continue

        axes = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

        # 각 축에 대해 FFT 특징 추출
        for axis in axes:
            if axis not in df.columns:
                continue
            data = df[axis].dropna().to_numpy()
            if len(data) == 0:
                continue
            x_fft, y_fft = fft_analysis(data, sample_rate)
            peak_frequency = x_fft[np.argmax(y_fft)]
            peak_amplitude = np.max(y_fft)
            X.append([peak_frequency, peak_amplitude])
            y.append(df['Resonant_Frequency'].iloc[0])

    X = np.array(X)
    y = np.array(y)

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'\n Mean Squared Error: {mse:.3f}')

        plt.scatter(y_test, y_pred)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('True Resonant Frequency')
        plt.ylabel('Predicted Resonant Frequency')
        plt.title('True vs Predicted Resonant Frequency')
        plt.grid(True)
        plt.show()
    else:
        print("데이터 부족")
