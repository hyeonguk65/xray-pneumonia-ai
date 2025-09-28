# 단일 이미지 예측

import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from data_load import get_dataloaders   # 클래스 이름만 쓰려고 불러옴


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)


# 클래스 이름 불러오기
_, _, _, _, class_names = get_dataloaders()
print("클래스:", class_names)


# 모델 불러오기
model = get_model(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(r"C:\Users\user\Desktop\project\pneumonia_resnet18.pth", map_location=device))
model.eval()


# 전처리 정의 (테스트용과 동일)
# 훈련할 때와 동일한 입력 형태로 맞춰서 모델이 제대로 인식할 수 있도록 하기 위해서
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# 단일 이미지 예측 함수
def predict_image(image_path, model, transform, class_names, device):
    # 흑백으로 불러오기
    # 전처리에서 다시 3채널로 맞추는 게 훈련과 동일한 조건 유지를 위해서
    img = Image.open(image_path).convert("L")   
    x = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = probs.argmax().item()
    return class_names[pred_class], probs[pred_class].item()


# 실행 예시
test_normal_image = r"C:\Users\user\Desktop\project\chest_xray\test\NORMAL\NORMAL2-IM-0372-0001.jpeg"  # 예시 파일
test_pneumonia_image = r"C:\Users\user\Desktop\project\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"  # 예시 파일
normal_label, normal_confidence = predict_image(test_normal_image, model, transform, class_names, device)
pneumonia_label, pneumonia_confidence = predict_image(test_pneumonia_image, model, transform, class_names, device)

print(f"이미지: {test_normal_image}")
print(f"예측 결과: {normal_label} (확률: {normal_confidence:.4f})")

print(f"이미지: {test_pneumonia_image}")
print(f"예측 결과: {pneumonia_label} (확률: {pneumonia_confidence:.4f})")
