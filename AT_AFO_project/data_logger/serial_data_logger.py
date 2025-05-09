import serial
import csv
from datetime import datetime

# 시리얼 포트 확인 필요
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
DURATION_SECONDS = 1 # 수집 시간 설정

filename = f"mpu6050_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 시리얼 연결
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

print("데이터 수집 시작")

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])

    start_time = datetime.now()

    while (datetime.now() - start_time).total_seconds() < DURATION_SECONDS:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = line.split(',')
                if len(data) == 6:
                    data.append('')
                    writer.writerow(data)
        except Exception as e:
            print("오류", e)
            continue

print("데이터 수집 완료")
ser.close()
