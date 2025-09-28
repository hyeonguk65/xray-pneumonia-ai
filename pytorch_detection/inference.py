import torch
import pydicom
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

num_classes = 2
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# 체크포인트 로드
checkpoint = torch.load(
    r"C:\Users\user\Desktop\project\checkpoint\checkpoint_epoch6.pth",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# 1) DICOM 이미지 읽기
# 정상
# img_path = r"C:\Users\user\Downloads\rsna-pneumonia-detection-challenge\stage_2_test_images\0246ec39-4938-4c8f-bfb6-c750d0313ee4.dcm"
# 폐렴
# img_path = r"C:\Users\user\Downloads\rsna-pneumonia-detection-challenge\stage_2_test_images\0d4f4313-566e-4de4-96c2-e2879a2846ce.dcm"

img_path = r"C:\Users\user\Downloads\rsna-pneumonia-detection-challenge\stage_2_test_images\0da4074c-fa4b-4450-81b3-74282d6c49a6.dcm"
dcm = pydicom.dcmread(img_path)
pixel_array = dcm.pixel_array

# 정규화 (0~255)
pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
pixel_array = pixel_array.astype(np.uint8)

# PIL 이미지로 변환
image = Image.fromarray(pixel_array).convert("RGB")


# 2) 모델 추론
transform = T.Compose([T.ToTensor()])
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)

boxes = outputs[0]["boxes"].cpu().numpy()
scores = outputs[0]["scores"].cpu().numpy()


# 3) 시각화
img_np = np.array(image).copy()  # numpy array 복사
conf_threshold = 0.2

results = []  # 로그용 저장

for box, score in zip(boxes, scores):
    if score >= conf_threshold:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 바운딩 박스 위에 confidence 표시
        label_text = f"PNEUMONIA {score:.2f}"
        cv2.putText(
            img_np, label_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        results.append({
            "label": "PNEUMONIA",
            "confidence": float(score),
            "box": (x1, y1, x2, y2)
        })


# 결과 표시
plt.figure(figsize=(8, 8))
plt.imshow(img_np)
plt.axis("off")

plt.subplots_adjust(bottom=0.2)  # 아래쪽 여백 확보

if len(results) == 0:
    # 정상일 경우
    text = "Prediction: NORMAL"
else:
    # 탐지된 결과 있을 경우
    text = "\n".join([f"{r['label']} ({r['confidence']:.2f})" for r in results])

plt.figtext(
    0.5, 0.05, text,
    ha="center", va="top", fontsize=12,
    bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3")
)

plt.show()


