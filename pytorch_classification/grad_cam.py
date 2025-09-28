import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model import get_model
from data_load import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 클래스 이름 & 모델 불러오기
_, _, _, _, class_names = get_dataloaders()
model = get_model(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(r"C:\Users\user\Desktop\project\pneumonia_resnet18.pth", map_location=device))
model.eval()


# Grad-CAM 준비
finalconv_name = 'layer4'  # ResNet18의 마지막 conv block

features = []
gradients = []

def save_features(module, input, output):
    features.append(output)

def save_gradients(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer = dict([*model.named_modules()])[finalconv_name]
target_layer.register_forward_hook(save_features)
target_layer.register_full_backward_hook(save_gradients)


# 단일 이미지 로드 & 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# 예시 이미지 경로
# image_path = r"C:\Users\user\Desktop\PNEUMONIA.png"
image_path = r"C:\Users\user\Desktop\test.png"

img = Image.open(image_path).convert("L")
input_tensor = transform(img).unsqueeze(0).to(device)


# Forward & Backward
output = model(input_tensor)
probs = torch.softmax(output, dim=1)[0]
pred_class = output.argmax(dim=1).item()
confidence = probs[pred_class].item()

# 예측한 클래스의 score만 backward
model.zero_grad()
score = output[0, pred_class]
score.backward()

# Grad-CAM 계산
grads = gradients[-1].cpu().data.numpy()[0]          # [C,H,W]
feature = features[-1].cpu().data.numpy()[0]         # [C,H,W]

weights = np.mean(grads, axis=(1,2))                 # GAP
cam = np.zeros(feature.shape[1:], dtype=np.float32)  # [H,W]
for i, w in enumerate(weights):
    cam += w * feature[i, :, :]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224,224))
cam = cam - np.min(cam)
cam = cam / np.max(cam)


# 시각화
img_np = np.array(img.resize((224,224)))
if len(img_np.shape) == 2:
    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)


plt.figure(figsize=(8, 4))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

# Grad-CAM 이미지
plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(superimposed)
plt.axis("off")

# 텍스트를 위한 공간 확보
plt.subplots_adjust(bottom=0.2)

# 전체 그림 아래 중앙에 텍스트 추가
plt.figtext(
    0.5, 0.12,  # (x=0.5: 중앙, y=0.05: 그림 아래쪽의 여유 공간)
    f"Prediction : {class_names[pred_class]} ({confidence:.4f})",
    ha="center", va="top", fontsize=12,
    bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3")
)

plt.show()

