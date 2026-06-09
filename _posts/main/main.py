'''
https://docs.ultralytics.com/tasks/obb#visual-samples 의 데이터셋을 사용했다고 가정한 코드입니다
그 중에서도, dota8.yaml 이라는 Ultralytics YOLO 공식 문서(Oriented Bounding Boxes Docs)에서 사용자를 위해 제공하는 샘플 데이터셋을 사용했습니다.


시작하기 전 아래 프로그램을 설치해야 합니다
pip install ultralytics
pip install matplotlib numpy
pip install opencv-python



실제 고속도로 CCTV 사진을 사용해 데이터셋을 직접 구축할 때 참고할 수 있는 방법:

1단계: CCTV 이미지 수집
고속도로 실시간 CCTV 스트리밍 화면을 주기적으로 캡처하거나, 국토교통부 등에서 제공하는 고속도로 차량 사진 데이터를 수집합니다. (예: 100장~300장 정도로 시작해보는 것을 추천합니다.)

2단계: 무료 라벨링 툴 사용하기 
좌표값을 직접 메모장에 적을 필요가 전혀 없습니다. 마우스 클릭 몇 번으로 YOLO 포맷 텍스트 파일을 만들어주는 훌륭한 무료 툴들이 있습니다.

추천 도구: Roboflow (로보플로우) 
웹사이트에 사진을 올리고 마우스로 드래그해서 차에 네모 박스를 치는 도구입니다.

다 만든 후 "YOLOv11 포맷으로 다운로드" 버튼만 누르면, images 폴더, labels 폴더, 심지어 highway.yaml 파일까지 완벽하게 압축파일로 만들어 줍니다.

3단계: 마우스 드래그 및 태깅
사진을 한 장씩 넘기면서 화면에 보이는 차량에 네모 박스를 치고, 미리 지정해 둔 라벨(0: Sedan, 1: Bus, 2: Truck)을 선택합니다.

요즘은 Roboflow 내부에 AI 어시스턴트(Auto-Label) 기능이 있어서, 마우스를 갖다 대면 AI가 대략적으로 박스를 먼저 쳐주고 사람은 수정만 하면 되기 때문에 작업 속도가 매우 빠릅니다.

4단계: 데이터셋 분할 및 내보내기
작업이 끝나면 툴 내부에서 버튼 하나로 Train / Val / Test 세트를 8 : 1 : 1 비율로 알아서 쪼개줍니다. 그대로 다운로드해서 프로젝트 폴더에 넣으면 끝납니다.

직접 데이터셋을 만드셔도 좋고 이미 있는 yolo 데이터셋을 쓰셔도 괜찮다고 생각합니다 위 방법이 잘 작동하는지는 시험해보지 않았습니다




best.pt 가중치 파일 획득법:

yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=50 imgsz=640 workers=0 device=cpu 
위 코드를 터미널에서 바로 실행



1. 모델 자체의 검증 성능 평가 (Validation Metrics)
'''

from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':

    # 1. 모델 자체의 검증 성능 평가 (Validation Metrics)
    print("\n[단계 1] 모델 자체의 검증 성능 평가 모델 로드 및 실행...")
    
    # 모델 로드
    model = YOLO("runs/obb/train/weights/best.pt") 
    
    # 검증 실행 (plots=False로 두면 시각화 이미지 생성을 생략하여 속도가 빨라집니다)
    metrics = model.val(data="dota8.yaml", plots=True, workers=0, device='cpu')
    save_dir = Path(metrics.save_dir)

    print("\n" + "="*30)
    print("📊 내 커스텀 모델(best.pt) 최종 검증 성적표")
    print("="*30)
    # 안전하게 .box 속성으로 수치 출력
    print(f"mAP50-95 (종합 정확도) : {metrics.box.map:.4f}")
    print(f"mAP50 (느슨한 기준 정확도): {metrics.box.map50:.4f}")
    print(f"Precision (정밀도)      : {metrics.box.mp:.4f}")
    print(f"Recall (재현율)         : {metrics.box.mr:.4f}")
    print("="*30)


'2. 시각화'  


from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. 이미지 출력을 위한 함수 정의 (최상단 배치)
def show_metric_image(save_dir, image_path, title):
    full_path = Path(save_dir) / image_path
    if full_path.exists():
        img = mpimg.imread(str(full_path))
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.show()
    else:
        print(f"파일을 찾을 수 없습니다: {image_path}. 오타가 있거나 plots=True 옵션이 꺼져있을 수 있습니다.")


# 2. 실제 실행 메인 루프
if __name__ == '__main__':

    # --------------------------------------------------------------------------
    # [단계 1] 모델 로드 및 검증 성능 평가
    # --------------------------------------------------------------------------
    print("\n[단계 1] 모델 자체의 검증 성능 평가 모델 로드 및 실행...")
    
    model = YOLO("runs/obb/train/weights/best.pt") 
    metrics = model.val(data="dota8.yaml", plots=True, workers=0, device='cpu')
    save_dir = Path(metrics.save_dir)

    print("\n" + "="*30)
    print("📊 내 커스텀 모델(best.pt) 최종 검증 성적표")
    print("="*30)
    print(f"mAP50-95 (종합 정확도) : {metrics.box.map:.4f}")
    print(f"mAP50 (느슨한 기준 정확도): {metrics.box.map50:.4f}")
    print(f"Precision (정밀도)      : {metrics.box.mp:.4f}")
    print(f"Recall (재현율)         : {metrics.box.mr:.4f}")
    print("="*30)

    # --------------------------------------------------------------------------
    # [단계 2] 시각화 및 세부 수치 출력
    # --------------------------------------------------------------------------
    print("\n[단계 2] 시각화 그래프 및 클래스별 세부 수치 출력...")

    # 1. 정밀도(Precision)와 재현율(Recall) 곡선
    show_metric_image(save_dir, "PR_curve.png", "Precision-Recall Curve (차종별 분류 성능)")

    # 보고서용 텍스트 수치 출력
    print(f"\n전체 클래스 종합 mAP50: {metrics.box.map50:.4f}")
    print(f"전체 클래스 종합 mAP50-95(엄격한 기준): {metrics.box.map:.4f}")
    for class_name, map50 in zip(metrics.names.values(), metrics.maps):
        print(f" - [{class_name}]의 mAP50 정확도: {map50:.4f}")

    print("\n---")

    # 2. 혼동 행렬 (Confusion Matrix)
    # 어떤 차종을 잘 맞추고 어떤 차종끼리 헷갈려하는지 백분율(%)로 확인
    show_metric_image(save_dir, "confusion_matrix_normalized.png", "Normalized Confusion Matrix (오분류 경향 분석)")

    # 3. F1-Score와 최적 임계값 선정 (F1_curve.png)
    # 실무 관제 시스템 적용 시 최적의 신뢰도(Confidence) 커트라인을 찾는 그래프
    show_metric_image(save_dir, "F1_curve.png", "F1-Confidence Curve (최적 신뢰도 임계값 탐색)")
    
    print(f"\n🎉 모든 시각화 그래프 파일이 저장된 폴더: {save_dir}")



#3. 정량 평가

from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    # 1. 우리가 앞서 dota8 데이터셋으로 학습시켜 만든 나만의 가중치 파일 로드
    model = YOLO("runs/obb/train/weights/best.pt") 

    # 2. 탭과 동일한 조건(IoU 0.5 기준 채점)으로 검증 데이터셋 평가 수행
    # plots=True 옵션을 켜야 탭에서 말한 PR 곡선, 혼동 행렬 그래프가 자동 생성됩니다.
    metrics = model.val(data="dota8.yaml", plots=True, workers=0, device='cpu')

    # 3. 탭의 [4. Evaluation & Analysis] 양식에 맞추어 콘솔창에 성적표 출력
    print("\n" + "="*50)
    print("📊 [참고 탭 대응] 내 교통량 분석 OBB 모델 최종 성능 지표")
    print("="*50)
    # ⚠️ 내부 설계 규칙에 맞추어 .obb 대신 .box 속성으로 안전하게 수정했습니다.
    print(f"• mAP50 (IoU 50% 기준 종합 정확도) : {metrics.box.map50:.4f} (1.0에 가까울수록 완벽)")
    print(f"• Precision (정밀도 - 오탐지 제어력) : {metrics.box.mp:.4f}")
    print(f"• Recall (재현율 - 미탐지 최소화율)  : {metrics.box.mr:.4f}")
    print(f"• mAP50-95 (엄격한 기준 종합 정확도): {metrics.box.map:.4f}")
    print("="*50)
    
    # 4. 결과 파일들이 저장된 폴더 경로 안내
    print(f"🎉 보고서용 시각화 그래프 파일들이 저장된 폴더: {Path(metrics.save_dir)}")    
    # 4. 결과 파일들이 저장된 폴더 경로 안내
    print(f"🎉 보고서용 시각화 그래프 파일들이 저장된 폴더: {Path(metrics.save_dir)}")




#4. 실제 차량 수(Ground Truth)와 예측 수(Predicted) 비교 평가 (<- cctv영상을 사용할 때 쓸 수 있는 방법)

import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("path/to/best.pt")

# 테스트할 고속도로 CCTV 영상
video_path = "highway_sample.mp4"
cap = cv2.VideoCapture(video_path)

# 안전장치: 비디오 파일이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print(f"❌ 에러: '{video_path}' 동영상 파일을 열 수 없습니다. 경로를 확인해 주세요.")
    exit()

# 해당 영상의 실제 차량 대수 (사람이 직접 전수조사한 값)
ground_truth_count = 150 

# 프로그램이 추적한 차량 ID를 저장할 세트 (중복 제거용)
tracked_vehicle_ids = set()

print("🚀 고속도로 CCTV 영상 분석 시작...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 객체 추적(Track) 수행 (persist=True로 프레임 간 ID 유지)
    results = model.track(frame, persist=True, verbose=False)

    # ⚠️ OBB 모델 기준 예외 처리 및 ID 확보
    if results[0].obb is not None:
        # obb 속성이 있고, 그 안에 추적 ID(id)가 부여되어 있는지 확인
        if results[0].obb.id is not None:
            ids = results[0].obb.id.cpu().numpy().astype(int)
            
            for obj_id in ids:
                # 영상에 등장한 고유 차량 ID를 세트에 누적 (자동 중복 제거)
                tracked_vehicle_ids.add(obj_id)

cap.release()

# --- 정량 평가 지표 계산 ---
predicted_count = len(tracked_vehicle_ids)
error_count = abs(ground_truth_count - predicted_count)

# 제로 디비전(0으로 나누기) 및 마이너스 정확도 방지 안전장치
if ground_truth_count > 0:
    accuracy = (1 - (error_count / ground_truth_count)) * 100
    # 만약 예측을 너무 많이 해서 오차가 정답보다 커지면 정확도가 음수가 되므로 최소값 0 보정
    accuracy = max(0.0, accuracy)
else:
    accuracy = 0.0

print("\n=======================================")
print("📊 고속도로 교통량 계수 평가 결과")
print("=======================================")
print(f"• 실제 통행량 (GT)       : {ground_truth_count} 대")
print(f"• 프로그램 측정량 (Pred)  : {predicted_count} 대")
print(f"• 오차 대수              : {error_count} 대")
print(f"• 계수 정확도 (Accuracy)  : {accuracy:.2f}%")
print("=======================================")