import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_load import get_dataloaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)


# 데이터셋 불러오기 (test만 사용)
_, _, test_loader, _, class_names = get_dataloaders()


# 모델 불러오기
model = get_model(num_classes=2, pretrained=False).to(device)
model.load_state_dict(torch.load(r"C:\Users\user\Desktop\project\pneumonia_resnet18.pth", map_location=device))
model.eval()


# 테스트셋 평가
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())


#               예측: NORMAL   예측: PNEUMONIA
# 실제: NORMAL       TN              FP
# 실제: PNEUMONIA    FN              TP

print("\n모델 평가 결과")
print("--------------------")

cm = confusion_matrix(y_true, y_pred)
print("혼동 행렬:")
print(cm)

# 정밀도(Precision) → 내가 "폐렴!"이라고 한 게 얼마나 진짜 폐렴인지 (내 말이 맞았나?)
# 재현율(Recall) → 실제 폐렴 환자를 내가 얼마나 놓치지 않고 잡았는지 (놓친 사람 있나?)
# F1-score → 둘 사이의 균형 (정밀도·재현율 둘 다 잘해야 높음)
cr = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n분류 보고서:")
print(cr)
