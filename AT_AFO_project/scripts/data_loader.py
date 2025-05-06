import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)

    # 열 추출
    selected_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    df = df[selected_columns]

    return df
