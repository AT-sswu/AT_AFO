import asyncio
import serial_asyncio
import csv
from datetime import datetime

# 설정
SERIAL_PORT = 'COM3'  # 포트 확인 필요
BAUD_RATE = 250000
DURATION_SECONDS = 10  # 수집 시간

# 파일 이름
filename = f"mpu6050_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

class MPU6050Reader(asyncio.Protocol):
    def __init__(self, loop):
        self.loop = loop
        self.buffer = ''
        self.start_time = datetime.now()
        self.csv_file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])

    def connection_made(self, transport):
        self.transport = transport
        print("데이터 수집 시작...")

    def data_received(self, data):
        self.buffer += data.decode('utf-8', errors='ignore')
        lines = self.buffer.split('\n')
        self.buffer = lines[-1]  # 마지막 줄은 아직 완성 안된 줄일 수 있음

        for line in lines[:-1]:
            values = line.strip().split(',')
            if len(values) == 6:
                print(values)
                self.writer.writerow(values)

        # 시간 체크하여 종료
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed >= DURATION_SECONDS:
            print("데이터 수집 완료.")
            self.csv_file.close()
            self.transport.close()
            self.loop.stop()

    def connection_lost(self, exc):
        print("시리얼 연결 종료")
        if not self.csv_file.closed:
            self.csv_file.close()
        self.loop.stop()

# 비동기 실행
loop = asyncio.get_event_loop()
coro = serial_asyncio.create_serial_connection(
    loop, lambda: MPU6050Reader(loop), SERIAL_PORT, baudrate=BAUD_RATE
)
loop.run_until_complete(coro)
loop.run_forever()
