import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from AT_data.fft_analysis import fft_analysis

# ML 모델 학습 및 예측 함수
def train_and_predict(data_dir='data', sample_rate=4096):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    X, y = [], []
    axes = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    for file in csv_files:
        df = pd.read_csv(file)

        if 'Resonant_Frequency' not in df.columns or pd.isnull(df['Resonant_Frequency'].iloc[0]):
            continue

        features = []
        for axis in axes:
            if axis not in df.columns:
                features.extend([0, 0])  # 없는 축 -> 0으로
                continue

            data = df[axis].dropna().to_numpy()
            if len(data) == 0:
                features.extend([0, 0])  # 데이터가 없을 경우 -> 0으로
                continue

            _, y_fft, peak_freq = fft_analysis(data, sample_rate, plot=False)
            peak_amp = np.max(y_fft)
            features.extend([peak_freq, peak_amp])

        X.append(features)
        y.append(df['Resonant_Frequency'].iloc[0])

    X = np.array(X)
    y = np.array(y)

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'\nMean Squared Error: {mse:.3f}')

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('True Resonant Frequency')
        plt.ylabel('Predicted Resonant Frequency')
        plt.title('True vs Predicted Resonant Frequency')
        plt.grid(True)
        plt.show()
    else:
        print("데이터가 충분하지 않습니다.")
