import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    # 필요한 열 추출
    accel_columns = ['Accel_X', 'Accel_Y', 'Accel_Z']
    df = df[accel_columns]
    return df
