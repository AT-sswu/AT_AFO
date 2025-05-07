import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import glob


# FFT 분석 함수
def fft_analysis(data, sample_rate):
    n = len(data)
    x = np.fft.fftfreq(n, 1 / sample_rate)
    y = np.fft.fft(data) / n
    mask = x >= 0
    return x[mask], np.abs(y[mask])

# 데이터 로딩 및 전처리
csv_files = glob.glob('data/*.csv')
X, y = [], []

# 각 파일에 대해 처리
for file in csv_files:
    df = pd.read_csv(file)
    sample_rate = 1000  # 샘플링 주파수 설정 -> 추후 변동
    axes = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    plt.figure(figsize=(18, 12))

    for i, axis in enumerate(axes):
        data = df[axis].dropna().to_numpy()
        x_plot, y_plot = fft_analysis(data, sample_rate)

        # 특징: 피크 위치와 강도
        peak_frequency = x_plot[np.argmax(y_plot)]  # 최대 진폭 주파수
        peak_amplitude = np.max(y_plot)  # 피크 진폭

        # 공진 주파수 예측을 위한 타겟 (최적 공진 주파수)
        resonant_frequency = df['Resonant_Frequency'].iloc[0]  # ex) 첫 번째 값 사용 -> 추후 변동

        X.append([peak_frequency, peak_amplitude])
        y.append(resonant_frequency)

        # FFT 분석 그래프
        plt.subplot(3, 2, i + 1)
        plt.plot(x_plot, y_plot)
        plt.title(f'FFT of {axis}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# 데이터셋 분할
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 예측 결과
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Resonant Frequency')
plt.ylabel('Predicted Resonant Frequency')
plt.title('True vs Predicted Resonant Frequency')
plt.show()
