import os
import pandas as pd
from scripts.data_loader import load_data
from scripts.feature_extractor import extract_dominant_freq

def build_dataset():
    data_dir = 'data'
    label_map = {
        'static_data.csv': 0,
        'fan_data.csv': 1,
        'motor_data.csv': 2
    }

    X = []
    y = []

    for filename, label in label_map.items():
        file_path = os.path.join(data_dir, filename)
        df = load_data(file_path)

        for i in range(0, len(df)-256, 128):  # 슬라이딩 윈도우
            window = df.iloc[i:i+256]  # 약 1.28초치 데이터 (샘플링 200Hz 가정)
            fx, _ = extract_dominant_freq(window['Accel_X'].values)
            fy, _ = extract_dominant_freq(window['Accel_Y'].values)
            fz, _ = extract_dominant_freq(window['Accel_Z'].values)
            gx, _ = extract_dominant_freq(window['Gyro_X'].values)
            gy, _ = extract_dominant_freq(window['Gyro_Y'].values)
            gz, _ = extract_dominant_freq(window['Gyro_Z'].values)

            X.append([fx, fy, fz, gx, gy, gz])
            y.append(label)

    return pd.DataFrame(X, columns=['fx', 'fy', 'fz', 'gx', 'gy', 'gz']), pd.Series(y)
