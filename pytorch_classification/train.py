# ResNet18

import torch
import torch.nn as nn
import torch.optim as optim
import time, copy
from sklearn.metrics import classification_report, confusion_matrix

from data_load import get_dataloaders
from model import get_model

# 1) 데이터 불러오기
# get_dataloaders() → train, val, test 데이터로더를 반환.
# dataset_sizes → 각 split별 데이터 개수. (에폭 평균 loss/acc 계산에 사용)
# class_names → 클래스 이름들 (['NORMAL', 'PNEUMONIA'])
train_loader, val_loader, test_loader, dataset_sizes, class_names = get_dataloaders()
dataloaders = {"train": train_loader, "val": val_loader}

# 2) 디바이스 설정: CUDA가 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

# 3) 모델 정의: ResNet18 + 출력층(2클래스)로 교체
# get_model 함수 → ResNet18 불러오고 출력층을 2개 클래스에 맞게 교체.
# pretrained=True → ImageNet 사전학습 가중치 활용 → 작은 데이터셋에서도 성능 향상 기대
model = get_model(num_classes=2, pretrained=True).to(device)

# 4) 손실함수/옵티마이저/스케줄러
# 손실함수   : CrossEntropyLoss: 다중분류 표준, 입력은 "logits"(softmax 하지 않은 값)
# 옵티마이저 : AdamW (Adam + weight decay 적용 → 과적합 완화) 빠르게 수렴, weight_decay(정규화) 주면 과적합 방지에 도움
# 스케줄러   : 검증 정확도가 2 epoch 동안 향상되지 않으면 학습률(LR)을 절반으로 줄임.
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 검증 정확도가 정체되면 LR을 낮추는 스케줄러
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

# 5) 학습 함수: train/val 한 에폭씩 돌며 최고 성능 가중치를 보관
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # 최고 성능일 때 가중치 스냅샷 저장.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:  # 두 번 돌면서 학습/평가 모두 진행.
            model.train() if phase == "train" else model.eval()
            loader = dataloaders[phase]

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # train일 때만 gradient 계산/역전파
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)               # [B, 2] logits
                    _, preds = torch.max(outputs, 1)      # 예측 클래스 인덱스
                    loss = criterion(outputs, labels)     # CE loss

                    # 학습 단계(train)에서는 역전파 + 가중치 업데이트
                    # 검증 단계(val)에서는 단순 forward만 실행
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 배치 누적 지표
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # 에폭 평균 지표
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 스케줄러 갱신(검증 기준)
            if phase == "val" and scheduler is not None:
                scheduler.step(epoch_acc)

            # 최고 성능이면 가중치 스냅샷 갱신
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"훈련 완료: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초")
    print(f"최고 검증 정확도: {best_acc:.4f}")

    # 최고 성능 가중치로 복원
    model.load_state_dict(best_model_wts)
    return model

# 6) 학습 실행 
# 10 epoch 학습 후, 최고 성능 모델(model_ft)을 얻음.
model_ft = train_model(model, criterion, optimizer, scheduler=scheduler, num_epochs=10)

# 7) 모델 저장 (가중치만 저장: 추론 시 같은 모델 구조 필요)
torch.save(model_ft.state_dict(), "pneumonia_resnet18.pth")
print("모델 저장 완료!")

