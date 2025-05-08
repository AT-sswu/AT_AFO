import serial
import time
import pandas as pd
import numpy as np

ser = serial.Serial('COM3', 115200)  # 포트 번호 수정

data = []
start_time = time.time()
duration = 10  # 데이터 수집 시간

while time.time() - start_time < duration:
    line = ser.readline().decode().strip()
    if line and "Accel" not in line:
        parts = line.split(",")
        if len(parts) == 7:
            data.append(parts)

df = pd.DataFrame(data, columns=[
    'Time(ms)', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'
])

# 공진 주파수 컬럼
df['Resonant_Frequency'] = np.nan

# 데이터 저장
df.to_csv(".csv", index=False) #파일명 수정

ser.close()

print("데이터 수집 및 저장 완료")
