'''
바탕화면에 yolo_test 폴더 만들고 그 폴더 안에 이 파이썬 파일과 영상 이동 (코드 파일과 동영상 파이 이름 변경 필요)
vscode 터미널에
pip install ultralytics
cd Desktop/yolo_test
python main1.py
순서대로 실행
'''
from ultralytics import YOLO

# OBB 모델 불러오기 (처음 실행 시 알아서 다운로드 됨)
model = YOLO('yolov8n-obb.pt') 

# 내 컴퓨터에 있는 동영상 파일 경로 넣기 (예: 'my_video.mp4')
# 웹캠을 쓰려면 'my_video.mp4' 대신 0 을 넣으세요.
results = model.track(source='my_car.mp4', show=True, classes=[9, 10], conf=0.15, save=True, save_txt=True)