import os
from ultralytics import YOLO
import cv2

# 압축 해제한 데이터셋 경로
dataset_dir = "/data"
data_yaml_path = os.path.join(dataset_dir, "data.yaml")

# 1. 모델 선택 (예시: YOLOv8s)
model_name = 'yolov8s.pt'  # yolov8~11 및 n/s/m/l/x 중 선택 가능

# 2. 모델 로드 및 학습
model = YOLO(model_name)

results = model.train(
    data=data_yaml_path,
    epochs=10,
    imgsz=640,
    batch=16,
    name='yolov8s_custom'
)

best_weights = results.best

# 3. 검증(Validation) 결과 도출
val_results = model.val(data=data_yaml_path)
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")

# 4. 테스트 이미지에 대한 결과 이미지 도출 및 저장
test_images_dir = os.path.join(dataset_dir, 'test', 'images')
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)

test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 학습된 모델 가중치 경로
best_weights = "runs/detect/yolov8s_custom72"
trained_model = YOLO(best_weights)

for img_file in test_images:
    img_path = os.path.join(test_images_dir, img_file)
    results = trained_model(img_path)
    result_img = results[0].plot()
    cv2.imwrite(os.path.join(results_dir, f'result_{img_file}'), result_img)

print(f"테스트 결과 이미지가 {results_dir} 폴더에 저장되었습니다.")
